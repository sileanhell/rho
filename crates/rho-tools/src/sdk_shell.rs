use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use rho_sdk::{
    tool::{
        OperationKind, PreparedToolInvocation, Tool, ToolContext, ToolError, ToolErrorKind,
        ToolInvocation, ToolMetadata, ToolOutput, ToolPreparationContext, ToolPrepareFuture,
        ToolProgress, ToolSecurity,
    },
    CapabilityKind, CapabilityRequest, CapabilitySource, ProcessEnvironment, ProcessExecution,
    ProcessInvocation, ProcessOutputLimits, ResolvedWorkspacePath, Workspace,
};
use serde_json::Value;

use crate::{
    cancellation::RunCancellation,
    shell_process::ShellArgs,
    tool::{Tool as AppTool, ToolError as AppToolError},
    DEFAULT_MAX_OUTPUT_BYTES,
};

use super::sdk_support::{map_app_error, map_invalid_app_error};

/// Options for the host-facing shell tool adapter.
#[derive(Clone)]
pub struct ShellToolOptions {
    max_output_bytes: usize,
    environment: ProcessEnvironment,
    mutation_observer: Option<Arc<dyn crate::WorkspaceMutationObserver>>,
    host_terminals: Option<Arc<dyn crate::HostTerminals>>,
    shell_executable: Option<PathBuf>,
}

impl Default for ShellToolOptions {
    fn default() -> Self {
        Self {
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            environment: ProcessEnvironment::InheritAll,
            mutation_observer: None,
            host_terminals: None,
            shell_executable: None,
        }
    }
}

impl ShellToolOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_output_bytes(mut self, max_output_bytes: usize) -> Self {
        self.max_output_bytes = max_output_bytes.max(1);
        self
    }

    pub fn environment(mut self, environment: ProcessEnvironment) -> Self {
        self.environment = environment;
        self
    }

    pub fn mutation_observer(
        mut self,
        observer: Arc<dyn crate::WorkspaceMutationObserver>,
    ) -> Self {
        self.mutation_observer = Some(observer);
        self
    }

    /// Terminals the host runs `bash` commands in, in place of local child
    /// processes; see [`crate::HostTerminals`].
    pub fn host_terminals(mut self, terminals: Arc<dyn crate::HostTerminals>) -> Self {
        self.host_terminals = Some(terminals);
        self
    }

    /// Executable the tool spawns instead of the kind's PATH-resolved name.
    /// Hosts resolving a non-PATH shell (Git's `usr/bin` bash, a managed
    /// download) pass the absolute path here so spawn never relies on PATH.
    pub fn shell_executable(mut self, executable: impl Into<PathBuf>) -> Self {
        self.shell_executable = Some(executable.into());
        self
    }
}

/// Which shell the [`shell_tool`] runs commands through.
///
/// Both variants compile into every binary; the host's shell configuration
/// picks the one a session advertises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShellKind {
    Bash,
    PowerShell,
}

struct SdkShellTool {
    kind: ShellKind,
    executable: PathBuf,
    max_output_bytes: usize,
    environment: ProcessEnvironment,
    mutation_observer: Option<Arc<dyn crate::WorkspaceMutationObserver>>,
    host_terminals: Option<Arc<dyn crate::HostTerminals>>,
}

impl SdkShellTool {
    fn new(kind: ShellKind, options: ShellToolOptions) -> Self {
        Self {
            kind,
            executable: options
                .shell_executable
                .unwrap_or_else(|| PathBuf::from(kind.default_executable())),
            max_output_bytes: options.max_output_bytes,
            environment: options.environment,
            mutation_observer: options.mutation_observer,
            // Host terminals run `bash` scripts; PowerShell stays local.
            host_terminals: match kind {
                ShellKind::Bash => options.host_terminals,
                ShellKind::PowerShell => None,
            },
        }
    }
}

struct ShellPlan {
    execution: ProcessExecution,
    resolved_cwd: ResolvedWorkspacePath,
}

impl ShellPlan {
    fn parse(
        kind: ShellKind,
        executable: &Path,
        arguments: Value,
        workspace: Option<&Workspace>,
        max_output_bytes: usize,
        environment: ProcessEnvironment,
    ) -> Result<Self, ToolError> {
        let arguments = ShellArgs::parse(arguments).map_err(map_invalid_app_error)?;
        let timeout = arguments.timeout().map_err(map_invalid_app_error)?;
        let workspace = workspace.ok_or_else(|| {
            ToolError::new(
                ToolErrorKind::Execution,
                "workspace is required for shell tools",
            )
        })?;
        let resolved_cwd = workspace
            .resolve_for_read(workspace.root())
            .map_err(|error| ToolError::new(ToolErrorKind::PolicyDenied, error.to_string()))?;
        let execution = ProcessExecution::new(
            resolved_cwd.path(),
            kind.invocation(executable, arguments.command),
            environment,
            ProcessOutputLimits::new(max_output_bytes, timeout),
        );
        Ok(Self {
            execution,
            resolved_cwd,
        })
    }

    /// The process facts approved before this plan executes.
    fn capability(&self, kind: ShellKind) -> CapabilityRequest {
        CapabilityRequest::process(
            self.execution.clone(),
            CapabilitySource::built_in_tool(kind.name()),
        )
    }

    /// Runs the plan in a host terminal when `host_terminals` can take it,
    /// else as a local child process.
    async fn execute(
        self,
        kind: ShellKind,
        host_terminals: Option<&dyn crate::HostTerminals>,
        context: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let workspace = context.workspace().ok_or_else(|| {
            ToolError::new(
                ToolErrorKind::Execution,
                "workspace is required for shell tools",
            )
        })?;
        workspace
            .revalidate(&self.resolved_cwd)
            .map_err(|error| ToolError::new(ToolErrorKind::PolicyDenied, error.to_string()))?;
        let max_output_bytes = self.execution.output_limits().max_output_bytes();
        let mut metadata = ToolMetadata::new().operation(OperationKind::Execute);
        // Only `bash` runs in host terminals; there it must, see `host_command`.
        let hosted = match (kind, host_terminals) {
            #[cfg(unix)]
            (ShellKind::Bash, Some(terminals)) => Some((
                terminals,
                crate::host_terminals::host_command(&self.execution).map_err(map_app_error)?,
            )),
            _ => None,
        };
        let run = match hosted {
            Some((terminals, command)) => {
                let (run, terminal) = crate::host_terminals::run(
                    terminals,
                    command,
                    self.execution.output_limits().timeout(),
                    context.cancellation(),
                    context.progress(),
                )
                .await
                .map_err(map_app_error)?;
                metadata = metadata.host_terminal(terminal);
                run
            }
            None => execute_with_progress(kind, self.execution, context).await?,
        };
        // Taken before `limit_data`, which may drop the structured view.
        let process = run.data().map(|outcome| outcome.process.clone());
        let mut output = run
            .limit_data(max_output_bytes)?
            .into_tool_output(metadata)?;
        if let Some(process) = process {
            output = output.with_process_result(process);
        }
        Ok(output)
    }
}

impl ShellKind {
    /// The shell a host offers when the user picks no preference.
    pub const fn platform_default() -> Self {
        if cfg!(windows) {
            Self::PowerShell
        } else {
            Self::Bash
        }
    }

    /// The other shell kind, for hosts that fall back when the configured
    /// one is not in a session's capability set.
    pub const fn other(self) -> Self {
        match self {
            Self::Bash => Self::PowerShell,
            Self::PowerShell => Self::Bash,
        }
    }

    /// Tool name the model sees.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Bash => "bash",
            Self::PowerShell => "powershell",
        }
    }

    /// Executable name resolved through PATH when the host supplies no
    /// override.
    pub const fn default_executable(self) -> &'static str {
        match self {
            Self::Bash => "bash",
            Self::PowerShell => "powershell.exe",
        }
    }

    /// Builds this shell's process invocation for `command`.
    ///
    /// `executable` is usually [`Self::default_executable`]; hosts may pass a
    /// resolved absolute path for shells that are not on PATH.
    pub fn invocation(
        self,
        executable: impl Into<PathBuf>,
        command: impl Into<String>,
    ) -> ProcessInvocation {
        let command = command.into();
        match self {
            Self::Bash => {
                ProcessInvocation::shell_from_path(executable, vec!["-lc".into()], command)
            }
            Self::PowerShell => ProcessInvocation::shell_from_path(
                executable,
                vec![
                    "-NoProfile".into(),
                    "-NonInteractive".into(),
                    "-Command".into(),
                ],
                super::powershell::wrapped_command(&command),
            ),
        }
    }

    async fn execute(
        self,
        execution: ProcessExecution,
        cancellation: RunCancellation,
        on_update: &mut (dyn FnMut(Vec<String>) + Send),
    ) -> Result<crate::Rendered<crate::shell_process::ShellOutcome>, AppToolError> {
        match self {
            Self::Bash => super::bash::execute_process(execution, cancellation, on_update).await,
            Self::PowerShell => {
                super::powershell::execute_process(execution, cancellation, on_update).await
            }
        }
    }
}

impl Tool for SdkShellTool {
    fn spec(&self) -> rho_sdk::model::ToolSpec {
        match self.kind {
            ShellKind::Bash => super::bash::Bash::new(/*rtk_enabled*/ false).spec(),
            ShellKind::PowerShell => {
                super::powershell::PowerShell::new(/*rtk_enabled*/ false).spec()
            }
        }
    }

    fn security(&self) -> ToolSecurity {
        ToolSecurity::built_in([CapabilityKind::Process])
    }

    fn output_schema(&self) -> Option<Value> {
        Some(crate::output_schema::<crate::shell_process::ShellOutcome>())
    }

    /// Carries the command line, so hosts can show what runs.
    fn start_metadata(&self, arguments: &Value) -> ToolMetadata {
        let metadata = ToolMetadata::new().operation(OperationKind::Execute);
        match arguments.get("command").and_then(Value::as_str) {
            Some(command) => metadata.command_line(command),
            None => metadata,
        }
    }

    /// Declares the process facts up front, so the runtime authorizes them
    /// before the call waits for its exclusive execution slot. Concurrent
    /// shell calls are then approved together instead of each approval
    /// waiting for the previous command to finish.
    fn prepare<'a>(
        &'a self,
        invocation: ToolInvocation,
        context: ToolPreparationContext,
    ) -> ToolPrepareFuture<'a> {
        Box::pin(async move {
            if context.cancellation().is_cancelled() {
                return Err(ToolError::cancelled());
            }
            let metadata = self.start_metadata(invocation.arguments());
            let plan = ShellPlan::parse(
                self.kind,
                &self.executable,
                invocation.into_arguments(),
                context.workspace(),
                self.max_output_bytes,
                self.environment.clone(),
            )?;
            Ok(PreparedToolInvocation::exclusive_with_capabilities(
                [plan.capability(self.kind)],
                metadata,
                move |context| {
                    Box::pin(async move {
                        if let Some(observer) = self.mutation_observer.as_ref() {
                            observer.mark_untracked_effect(
                                crate::UntrackedWorkspaceEffect::ShellCommand,
                                self.kind.name(),
                            );
                        }
                        plan.execute(self.kind, self.host_terminals.as_deref(), &context)
                            .await
                    })
                },
            ))
        })
    }
}

async fn execute_with_progress(
    kind: ShellKind,
    execution: ProcessExecution,
    context: &ToolContext,
) -> Result<crate::Rendered<crate::shell_process::ShellOutcome>, ToolError> {
    let (update_sender, mut updates) = tokio::sync::mpsc::unbounded_channel::<Vec<String>>();
    let mut on_update = move |lines: Vec<String>| {
        let _ = update_sender.send(lines);
    };
    let call = kind.execute(execution, context.cancellation().clone(), &mut on_update);
    tokio::pin!(call);
    let mut updates_open = true;
    let result = loop {
        tokio::select! {
            result = &mut call => break result,
            update = updates.recv(), if updates_open => {
                match update {
                    Some(lines) => {
                        let _ = context
                            .progress()
                            .send(ToolProgress::message(lines.join("\n")))
                            .await;
                    }
                    None => updates_open = false,
                }
            }
        }
    };
    while let Ok(lines) = updates.try_recv() {
        let _ = context
            .progress()
            .send(ToolProgress::message(lines.join("\n")))
            .await;
    }
    result.map_err(map_app_error)
}

/// Returns the workspace shell tool for `kind` (`bash` or `powershell`) as an
/// SDK tool trait object. Both kinds ship in every build; hosts pick one per
/// session from their shell configuration.
///
/// The tool does not grant capabilities by itself. Hosts must attach a
/// workspace and a non-default policy on the runtime before commands run.
/// Default environment inheritance is [`ProcessEnvironment::InheritAll`];
/// security-sensitive hosts should pass a stricter policy through
/// [`ShellToolOptions::environment`].
pub fn shell_tool(kind: ShellKind, options: ShellToolOptions) -> std::sync::Arc<dyn Tool> {
    std::sync::Arc::new(SdkShellTool::new(kind, options))
}
