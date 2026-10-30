use std::sync::Arc;

use super::host_io::HostIo;
use crate::agent::{AgentCapabilities, ToolCapability};
use rho_sdk::{tool::Tool, ProcessEnvironment};

/// Build the mid-session/startup edit tool with one shared options policy.
pub(super) fn edit_tool(
    edit_format: rho_tools::EditFormat,
    max_output_bytes: usize,
    mutation_observer: Arc<dyn rho_tools::WorkspaceMutationObserver>,
    host_io: &HostIo,
) -> Arc<dyn Tool> {
    rho_tools::coding_tool(
        rho_tools::CodingToolKind::Edit,
        host_io.coding_options(
            rho_tools::CodingToolOptions::new()
                .max_output_bytes(max_output_bytes)
                .edit_tool(edit_format)
                .mutation_observer(mutation_observer),
        ),
    )
}

/// The capability that advertises `kind` to the model.
fn shell_capability(kind: rho_tools::ShellKind) -> ToolCapability {
    match kind {
        rho_tools::ShellKind::Bash => ToolCapability::Bash,
        rho_tools::ShellKind::PowerShell => ToolCapability::Powershell,
    }
}

pub(super) fn sdk_bundle(
    capabilities: &AgentCapabilities,
    preferred_shell: rho_tools::ShellKind,
    max_output_bytes: usize,
    process_environment: ProcessEnvironment,
    mutation_observer: Arc<dyn rho_tools::WorkspaceMutationObserver>,
    file_view: rho_tools::FileViewPolicy,
    host_io: &HostIo,
) -> super::sdk_registry::StaticToolBundle {
    use rho_tools::CodingToolKind;

    let options = host_io.coding_options(
        rho_tools::CodingToolOptions::new()
            .max_output_bytes(max_output_bytes)
            .file_view(file_view)
            .mutation_observer(Arc::clone(&mutation_observer)),
    );
    let mut tools = Vec::new();
    for (capability, kind) in [
        (ToolCapability::ListDir, CodingToolKind::ListDir),
        (ToolCapability::ReadFile, CodingToolKind::ReadFile),
        (ToolCapability::WriteFile, CodingToolKind::WriteFile),
        (ToolCapability::Edit, CodingToolKind::Edit),
        (ToolCapability::Grep, CodingToolKind::Grep),
        (ToolCapability::Glob, CodingToolKind::Glob),
    ] {
        if !capabilities.contains(&capability) {
            continue;
        }
        tools.push(rho_tools::coding_tool(kind, options.clone()));
    }
    // The configured shell wins; an explicit allowlist may still name the
    // other one when the host kept both capabilities.
    let shell_kind = [preferred_shell, preferred_shell.other()]
        .into_iter()
        .find(|kind| capabilities.contains(&shell_capability(*kind)));
    if let Some(shell_kind) = shell_kind {
        // RTK stays disabled here. Authorization and execution must use the same
        // immutable process description.
        tools.push(rho_tools::shell_tool(
            shell_kind,
            host_io.shell_options(
                rho_tools::ShellToolOptions::new()
                    .max_output_bytes(max_output_bytes)
                    .environment(process_environment)
                    .mutation_observer(mutation_observer)
                    .shell_executable(super::shell_executable(shell_kind)),
            ),
        ));
    }
    super::sdk_registry::StaticToolBundle::new(tools)
}
