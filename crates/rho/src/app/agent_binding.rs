use std::sync::Arc;

use rho_providers::credentials::CredentialStore;

use crate::{
    agent::{
        AgentCapabilities, AgentDefinition, AgentFingerprint, AgentId, AgentRuntimeSpec,
        AntigravityTool, CursorTool, ModelPolicy, PromptPolicy, ReasoningLevel, ToolCapability,
        ToolPolicy,
    },
    config::Config,
};

#[path = "agent_binding_acp.rs"]
mod acp;
use acp::{
    bind_antigravity_runtime, bind_cursor_runtime, frozen_antigravity_runtime,
    frozen_cursor_runtime,
};
#[path = "agent_binding_model_override.rs"]
mod model_override;
pub(crate) use model_override::ModelOverride;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AgentRole {
    InteractiveRoot,
    AutomationRoot,
    Delegated,
    Workflow,
}

#[derive(Clone, Debug)]
pub(crate) struct AgentInvocation {
    pub(crate) role: AgentRole,
    pub(crate) available_tools: AgentCapabilities,
}

/// Capacity class for nested concurrency pools.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CapacityClass {
    Rho,
    Cursor,
    Antigravity,
}

/// Runtime-specific values produced by binding.
///
/// Callers must match exhaustively so Rho-shaped config and external spawn
/// data stay separate after bind.
#[derive(Clone, Debug)]
pub(crate) enum BoundRuntime {
    Rho {
        config: Box<Config>,
        capabilities: AgentCapabilities,
    },
    Cursor {
        /// Cursor `--model` value, byte-for-byte from the definition when set.
        /// `None` means inherit Cursor's own default (no `--model` flag).
        model: Option<String>,
        tools: Vec<CursorTool>,
        /// Snapshot of the parent permission mode at bind time. Cursor spawn
        /// maps this; Auto / Allow edits / Supervised already failed at bind.
        permission_mode: crate::permission::PermissionMode,
    },
    Antigravity {
        /// Session `model` config option. `None` keeps the server default.
        model: Option<String>,
        tools: Vec<AntigravityTool>,
        /// Snapshot of the parent permission mode at bind time; Auto / Allow
        /// edits / Supervised already failed at bind.
        permission_mode: crate::permission::PermissionMode,
    },
}

impl BoundRuntime {
    pub(crate) fn capacity_class(&self) -> CapacityClass {
        match self {
            Self::Rho { .. } => CapacityClass::Rho,
            Self::Cursor { .. } => CapacityClass::Cursor,
            Self::Antigravity { .. } => CapacityClass::Antigravity,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BoundAgent {
    definition: Arc<AgentDefinition>,
    fingerprint: AgentFingerprint,
    runtime: BoundRuntime,
    step_limit: u64,
}

impl BoundAgent {
    pub(crate) fn id(&self) -> &AgentId {
        &self.definition.id
    }

    pub(crate) fn fingerprint(&self) -> AgentFingerprint {
        self.fingerprint
    }

    pub(crate) fn definition(&self) -> &AgentDefinition {
        &self.definition
    }

    pub(crate) fn runtime(&self) -> &BoundRuntime {
        &self.runtime
    }

    /// Rho-bound config. Claude-cli agents have no Rho provider/model config.
    pub(crate) fn rho_config(&self) -> Option<&Config> {
        match &self.runtime {
            BoundRuntime::Rho { config, .. } => Some(config.as_ref()),
            BoundRuntime::Cursor { .. } | BoundRuntime::Antigravity { .. } => None,
        }
    }

    /// The model this launch will actually run on.
    ///
    /// Taken from the bound runtime rather than the definition, so a pinned
    /// model, an inherited one, and a Claude pass-through all report what this
    /// launch settled on.
    pub(crate) fn prompt_model(&self) -> crate::model_identity::PromptModel {
        use crate::model_identity::PromptModel;
        match &self.runtime {
            BoundRuntime::Rho { config, .. } => PromptModel::from_config(config),

            BoundRuntime::Cursor { model, .. } => PromptModel::ExternalCli {
                runtime: crate::agent::AgentRuntime::Cursor,
                requested: model.clone(),
                resolved: None,
            },
            BoundRuntime::Antigravity { model, .. } => PromptModel::ExternalCli {
                runtime: crate::agent::AgentRuntime::Antigravity,
                requested: model.clone(),
                resolved: None,
            },
        }
    }

    /// Carries the host session's `/fast` onto a Rho bind that inherits the
    /// host model, when the bound selection can serve it.
    ///
    /// For synthetic host-owned asides such as `/side`, which follow the
    /// conversation rather than an agent file. Catalog agents must not use
    /// this: their definition's `fast` is authoritative.
    pub(crate) fn keep_host_fast_mode(&mut self, host_config: &Config) {
        if let BoundRuntime::Rho { config, .. } = &mut self.runtime {
            config.fast_mode = host_config.fast_mode
                && rho_providers::providers::fast_mode::supports_fast_mode(
                    &config.provider,
                    &config.model,
                    &config.auth,
                );
        }
    }

    /// Rho-bound capabilities. Claude-cli agents do not bind host tools.
    pub(crate) fn rho_capabilities(&self) -> Option<&AgentCapabilities> {
        match &self.runtime {
            BoundRuntime::Rho { capabilities, .. } => Some(capabilities),
            BoundRuntime::Cursor { .. } | BoundRuntime::Antigravity { .. } => None,
        }
    }

    pub(crate) fn step_limit(&self) -> u64 {
        self.step_limit
    }

    pub(crate) fn prompt(&self) -> &PromptPolicy {
        &self.definition.prompt
    }

    /// Identity stamped onto `result.json` at the Starting boundary.
    ///
    /// Reasoning is what bind settled on. Claude inherit is `None`.
    pub(crate) fn artifact_identity(&self) -> crate::run_artifacts::RunArtifactIdentity {
        match self.runtime() {
            BoundRuntime::Rho { config, .. } => crate::run_artifacts::RunArtifactIdentity {
                agent_id: self.id().to_string(),
                agent_fingerprint: self.fingerprint().to_string(),
                provider: config.provider.clone(),
                model: Some(config.model.clone()),
                runtime: crate::agent::AgentRuntime::Rho,
                reasoning: Some(config.reasoning),
            },
            BoundRuntime::Cursor { model, .. } => crate::run_artifacts::RunArtifactIdentity {
                agent_id: self.id().to_string(),
                agent_fingerprint: self.fingerprint().to_string(),
                provider: crate::cursor_runtime::models::CURSOR_SOURCE_LABEL.into(),
                // `None` means no `--model` pin; Cursor chooses.
                model: model.clone(),
                runtime: crate::agent::AgentRuntime::Cursor,
                reasoning: None,
            },
            BoundRuntime::Antigravity { model, .. } => crate::run_artifacts::RunArtifactIdentity {
                agent_id: self.id().to_string(),
                agent_fingerprint: self.fingerprint().to_string(),
                provider: crate::antigravity_runtime::ANTIGRAVITY_LABEL_NAME.into(),
                // `None` means no model pin; the server chooses.
                model: model.clone(),
                runtime: crate::agent::AgentRuntime::Antigravity,
                reasoning: None,
            },
        }
    }

    /// Build the Cursor session request for a bound Cursor runtime.
    pub(crate) fn into_cursor_session(
        self,
        prompt: String,
        output_file: std::path::PathBuf,
        cwd: std::path::PathBuf,
        cancellation: rho_tools::cancellation::RunCancellation,
        status_tx: Option<tokio::sync::watch::Sender<crate::subagent::RunStatus>>,
        started_status: Option<crate::subagent::RunStatus>,
    ) -> Option<crate::cursor_runtime::session::CursorSessionRequest> {
        let identity = self.artifact_identity();
        let BoundRuntime::Cursor {
            tools,
            permission_mode,
            ..
        } = self.runtime
        else {
            return None;
        };
        Some(crate::cursor_runtime::session::CursorSessionRequest {
            system_prompt: self.definition.prompt.clone(),
            identity,
            tools,
            prompt,
            output_file,
            cwd,
            permission_mode,
            cancellation,
            status_tx,
            started_status,
            parent_messages: None,
            // No home dir leaves only `CURSOR_CONFIG_DIR` / XDG to find the
            // user's config; a missing file derives the managed config from `{}`.
            config_paths: crate::cursor_runtime::acp_config::CursorConfigPaths::from_env(
                crate::paths::home_dir().unwrap_or_default(),
            ),
            overrides: Default::default(),
        })
    }

    /// Build the Antigravity session request for a bound Antigravity runtime.
    pub(crate) fn into_antigravity_session(
        self,
        prompt: String,
        output_file: std::path::PathBuf,
        cwd: std::path::PathBuf,
        cancellation: rho_tools::cancellation::RunCancellation,
        status_tx: Option<tokio::sync::watch::Sender<crate::subagent::RunStatus>>,
        started_status: Option<crate::subagent::RunStatus>,
    ) -> Option<crate::antigravity_runtime::session::AntigravitySessionRequest> {
        let identity = self.artifact_identity();
        let BoundRuntime::Antigravity {
            tools,
            permission_mode,
            ..
        } = self.runtime
        else {
            return None;
        };
        Some(
            crate::antigravity_runtime::session::AntigravitySessionRequest {
                system_prompt: self.definition.prompt.clone(),
                identity,
                tools,
                prompt,
                output_file,
                cwd,
                permission_mode,
                cancellation,
                status_tx,
                started_status,
                parent_messages: None,
                home: crate::antigravity_runtime::home::AntigravityHome::from_env(
                    &crate::paths::home_dir().unwrap_or_default(),
                ),
                overrides: Default::default(),
            },
        )
    }
}

pub(crate) struct AgentBinder;

impl AgentBinder {
    pub(crate) fn bind(
        definition: Arc<AgentDefinition>,
        invocation: AgentInvocation,
        host_config: &Config,
    ) -> anyhow::Result<BoundAgent> {
        Self::bind_with_model_override(
            definition,
            invocation,
            host_config,
            /*model_override*/ None,
        )
    }

    /// Per-call selection changes only the launch model, never the definition
    /// identity, prompt, capabilities, or host permission ceiling.
    pub(crate) fn bind_with_model_override(
        definition: Arc<AgentDefinition>,
        invocation: AgentInvocation,
        host_config: &Config,
        model_override: Option<ModelOverride<'_>>,
    ) -> anyhow::Result<BoundAgent> {
        Self::bind_with_credentials(
            definition,
            invocation,
            host_config,
            model_override,
            &crate::credential_store::AppCredentialStore,
        )
    }

    fn bind_with_credentials(
        definition: Arc<AgentDefinition>,
        invocation: AgentInvocation,
        host_config: &Config,
        model_override: Option<ModelOverride<'_>>,
        store: &dyn CredentialStore,
    ) -> anyhow::Result<BoundAgent> {
        if model_override.is_some()
            && definition.runtime.runtime() != crate::agent::AgentRuntime::Rho
        {
            anyhow::bail!(
                "agent '{}': per-call model overrides require runtime rho, not {}",
                definition.id,
                definition.runtime.runtime()
            );
        }
        let fingerprint = definition.fingerprint();
        // Only Rho agents take a configured `[internal_agents.<id>]` entry; an
        // entry for any other runtime fails here instead of being ignored.
        let configured = crate::agent::configured_model(&definition, &host_config.internal_agents)?;
        let runtime = match &definition.runtime {
            AgentRuntimeSpec::Rho {
                tools,
                model,
                reasoning,
            } => {
                // The configured entry replaces the definition's model; an
                // explicit per-call selection still wins over both.
                let (model, reasoning) =
                    crate::agent::effective_rho_model(model, *reasoning, configured.as_ref());
                let mut config = bind_rho_config(
                    definition.id.as_str(),
                    model,
                    reasoning,
                    host_config,
                    model_override,
                    store,
                )?;
                apply_fast_mode(&definition.id, model, invocation.role, &mut config)?;
                BoundRuntime::Rho {
                    capabilities: bind_rho_capabilities(&definition, tools, &invocation)?,
                    config: Box::new(config),
                }
            }
            AgentRuntimeSpec::Cursor(config) => {
                bind_cursor_runtime(&definition, config, &invocation, host_config)?
            }
            AgentRuntimeSpec::Antigravity(config) => {
                bind_antigravity_runtime(&definition, config, &invocation, host_config)?
            }
        };
        Ok(BoundAgent {
            definition,
            fingerprint,
            runtime,
            step_limit: super::sdk_config::run_step_limit().get() as u64,
        })
    }

    /// Rebuilds a launch object only from metadata stored in a frozen graph.
    ///
    /// This path does not discover, open, or bind an agent definition. Current
    /// config supplies credentials and can narrow permission mode, but every
    /// provider, model, prompt, capability, and step choice comes from `frozen`.
    pub(crate) fn bind_frozen(
        frozen: &crate::workflow::ResolvedAgent,
        current_config: &Config,
        current_tools: &AgentCapabilities,
    ) -> anyhow::Result<BoundAgent> {
        let id = AgentId::new(frozen.agent_id.clone())?;
        let fingerprint = frozen.fingerprint.parse::<AgentFingerprint>()?;
        let prompt = decode_frozen_prompt_policy(&frozen.prompt_policy)?;
        let permission_mode = narrower_permission_mode(
            parse_permission_mode(&frozen.permission_ceiling)?,
            current_config.permission_mode,
        );
        let definition = Arc::new(AgentDefinition {
            id,
            description: "frozen workflow agent".into(),
            prompt,
            runtime: AgentRuntimeSpec::default(),
        });
        let runtime = match frozen.runtime {
            crate::workflow::AgentRuntime::Rho => {
                let mut config = current_config.clone();
                if let Some(provider) = &frozen.provider {
                    config.provider.clone_from(provider);
                }
                if let Some(model) = &frozen.model {
                    config.model.clone_from(model);
                }
                if let Some(reasoning) = &frozen.reasoning {
                    config.reasoning = reasoning.parse().map_err(|_| {
                        anyhow::anyhow!("frozen agent reasoning is invalid: '{reasoning}'")
                    })?;
                }
                if let Some(auth) = &frozen.auth_profile {
                    config.auth.clone_from(auth);
                }
                // Plan time already validated support; never inherit the
                // current session's /fast.
                config.fast_mode = frozen.fast;
                config.permission_mode = permission_mode;
                let capabilities = frozen_capabilities(frozen, current_tools);
                BoundRuntime::Rho {
                    config: Box::new(config),
                    capabilities,
                }
            }
            crate::workflow::AgentRuntime::Cursor => {
                frozen_cursor_runtime(frozen, permission_mode)?
            }
            crate::workflow::AgentRuntime::Antigravity => {
                frozen_antigravity_runtime(frozen, permission_mode)?
            }
        };
        Ok(BoundAgent {
            definition,
            fingerprint,
            runtime,
            step_limit: frozen.step_limit,
        })
    }
}

fn decode_frozen_prompt_policy(encoded: &str) -> anyhow::Result<PromptPolicy> {
    if let Some(text) = encoded.strip_prefix("extend:") {
        Ok(PromptPolicy::Extend(text.to_owned()))
    } else if let Some(text) = encoded.strip_prefix("replace:") {
        Ok(PromptPolicy::Replace(text.to_owned()))
    } else {
        anyhow::bail!("frozen prompt policy is invalid")
    }
}

fn frozen_capabilities(
    frozen: &crate::workflow::ResolvedAgent,
    current_tools: &AgentCapabilities,
) -> AgentCapabilities {
    let mut tools = crate::agent::ToolCapabilitySet::new();
    for name in &frozen.capabilities {
        let capability = ToolCapability::parse(name.clone());
        if current_tools.contains(&capability) {
            tools.insert(capability);
        }
    }
    let mut capabilities = AgentCapabilities::new(tools);
    for forbidden in [
        ToolCapability::Advisor,
        ToolCapability::Agent,
        ToolCapability::Agents,
        ToolCapability::Questionnaire,
        ToolCapability::Rho,
        ToolCapability::Workflow,
    ] {
        capabilities.remove(&forbidden);
    }
    capabilities
}

/// Keeps only the configured model shell in `capabilities`, so a bound run
/// advertises exactly one shell tool.
///
/// This is what the old per-platform `cfg` removal did, now driven by
/// `behavior.shell_tool`. Resolution happens at bind time: changing the
/// preference affects the next bound run, never a running session's tool
/// list.
pub(crate) fn retain_configured_shell(
    capabilities: &mut AgentCapabilities,
    config: &crate::config::Config,
) {
    let withheld = match config.resolved_shell_tool() {
        rho_tools::ShellKind::Bash => ToolCapability::Powershell,
        rho_tools::ShellKind::PowerShell => ToolCapability::Bash,
    };
    capabilities.remove(&withheld);
}

fn parse_permission_mode(value: &str) -> anyhow::Result<crate::permission::PermissionMode> {
    value
        .parse()
        .map_err(|error| anyhow::anyhow!("frozen permission ceiling is invalid: {error}"))
}

pub(crate) fn narrower_permission_mode(
    frozen: crate::permission::PermissionMode,
    current: crate::permission::PermissionMode,
) -> crate::permission::PermissionMode {
    if current.restrictiveness_rank() < frozen.restrictiveness_rank() {
        current
    } else {
        frozen
    }
}

fn bind_rho_capabilities(
    definition: &AgentDefinition,
    tools: &ToolPolicy,
    invocation: &AgentInvocation,
) -> anyhow::Result<AgentCapabilities> {
    let mut capabilities = invocation.available_tools.clone();
    if matches!(invocation.role, AgentRole::Delegated | AgentRole::Workflow) {
        // Keep questionnaire when the host offers it. The executor strips it
        // before bind when no parent bridge can answer.
        capabilities.remove(&ToolCapability::Agent);
        capabilities.remove(&ToolCapability::Agents);
        // The advisor reviews the root session. A child run has its own
        // session and nothing to review.
        capabilities.remove(&ToolCapability::Advisor);
    }
    if invocation.role == AgentRole::Workflow {
        for capability in [
            ToolCapability::Advisor,
            ToolCapability::Agent,
            ToolCapability::Agents,
            ToolCapability::Questionnaire,
            ToolCapability::Rho,
            ToolCapability::Workflow,
        ] {
            capabilities.remove(&capability);
        }
    }

    match tools {
        ToolPolicy::All => {
            capabilities.remove(&ToolCapability::Shell);
            Ok(capabilities)
        }
        ToolPolicy::Allow(requested) => {
            let mut resolved = crate::agent::ToolCapabilitySet::new();
            let mut unavailable = Vec::new();
            for tool in requested {
                if tool == &ToolCapability::Shell {
                    let shell = if capabilities.contains(&ToolCapability::Bash) {
                        Some(ToolCapability::Bash)
                    } else if capabilities.contains(&ToolCapability::Powershell) {
                        Some(ToolCapability::Powershell)
                    } else {
                        None
                    };
                    if let Some(shell) = shell {
                        resolved.insert(shell);
                    } else {
                        unavailable.push(tool.to_string());
                    }
                } else if capabilities.contains(tool) {
                    resolved.insert(tool.clone());
                } else if may_omit_unavailable_tool(tool, invocation.role) {
                    // Role and host gates strip some built-ins for this launch.
                    // Definitions may still list them for other invocations.
                } else {
                    unavailable.push(tool.to_string());
                }
            }
            if !unavailable.is_empty() {
                anyhow::bail!(
                    "agent '{}': requested tools are unavailable for {:?}: {}",
                    definition.id,
                    invocation.role,
                    unavailable.join(", ")
                );
            }
            Ok(AgentCapabilities::new(resolved))
        }
    }
}

/// Built-ins that a definition may list even when this launch cannot offer them.
///
/// Host and role policy remove these before bind (no parent questionnaire bridge,
/// no nested agents, workflow isolation). Soft-omit keeps the allowlist valid so
/// the same definition still binds when those tools are present.
fn may_omit_unavailable_tool(tool: &ToolCapability, role: AgentRole) -> bool {
    match tool {
        ToolCapability::Questionnaire => true,
        ToolCapability::Agent | ToolCapability::Agents | ToolCapability::Advisor => {
            matches!(role, AgentRole::Delegated | AgentRole::Workflow)
        }
        ToolCapability::Rho | ToolCapability::Workflow => matches!(role, AgentRole::Workflow),
        _ => false,
    }
}

/// The config a fresh launch of `definition` would run with on top of
/// `host_config`: its effective model policy and reasoning (its configured
/// entry, else the file's) applied, as startup does after
/// CLI overrides. External-runtime and failing definitions keep `host_config`.
pub(crate) fn relaunch_config(definition: &AgentDefinition, host_config: &Config) -> Config {
    let AgentRuntimeSpec::Rho {
        model, reasoning, ..
    } = &definition.runtime
    else {
        return host_config.clone();
    };
    let Ok(configured) = crate::agent::configured_model(definition, &host_config.internal_agents)
    else {
        return host_config.clone();
    };
    let (model, reasoning) =
        crate::agent::effective_rho_model(model, *reasoning, configured.as_ref());
    bind_rho_config(
        definition.id.as_str(),
        model,
        reasoning,
        host_config,
        /*model_override*/ None,
        &crate::credential_store::AppCredentialStore,
    )
    .unwrap_or_else(|_| host_config.clone())
}

fn bind_rho_config(
    agent_id: &str,
    model: &ModelPolicy,
    reasoning: Option<ReasoningLevel>,
    host_config: &Config,
    model_override: Option<ModelOverride<'_>>,
    store: &dyn CredentialStore,
) -> anyhow::Result<Config> {
    let mut config = match model_override {
        Some(reference) => model_override::bind_config(agent_id, reference, host_config, store)?,
        None => {
            let mut config = host_config.clone();
            apply_rho_model_policy(agent_id, model, &mut config, store)?;
            config
        }
    };
    if let Some(reasoning) = reasoning {
        config.reasoning = reasoning;
    }
    Ok(config)
}

/// Settles fast mode for a bound Rho agent.
///
/// Delegated and workflow runs use exactly the agent's `fast` value, so the
/// parent's `/fast` never leaks into children. Roots keep the user's saved
/// preference and `fast: true` can only turn it on. An explicit request must
/// be servable by the bound provider, model, and auth. `model` is the agent's
/// effective policy (its configured entry, else the file) and supplies only
/// `fast`, including when a per-call selection chose the provider and model.
fn apply_fast_mode(
    agent_id: &AgentId,
    model: &ModelPolicy,
    role: AgentRole,
    config: &mut Config,
) -> anyhow::Result<()> {
    let fast = model.selection().is_some_and(|selection| selection.fast);
    if fast
        && !rho_providers::providers::fast_mode::supports_fast_mode(
            &config.provider,
            &config.model,
            &config.auth,
        )
    {
        anyhow::bail!(
            "agent '{agent_id}': fast mode is not available for {}/{} with auth '{}'",
            config.provider,
            config.model,
            config.auth
        );
    }
    config.fast_mode = match role {
        AgentRole::Delegated | AgentRole::Workflow => fast,
        AgentRole::InteractiveRoot | AgentRole::AutomationRoot => config.fast_mode || fast,
    };
    Ok(())
}

/// Applies an agent's Rho model policy onto a host config clone.
fn apply_rho_model_policy(
    agent_id: &str,
    model: &ModelPolicy,
    config: &mut Config,
    store: &dyn CredentialStore,
) -> anyhow::Result<()> {
    match model {
        ModelPolicy::Inherit => Ok(()),
        ModelPolicy::Prefer(selection)
        | ModelPolicy::Require(selection)
        | ModelPolicy::Select(selection) => {
            // Resolve before provider or model-specific handling so all
            // downstream code sees the concrete target.
            let resolved = config
                .model_aliases
                .resolve(&selection.model)
                .map_err(|error| anyhow::anyhow!("agent '{agent_id}': {error}"))?;
            match (&selection.provider, &resolved.provider, &resolved.alias) {
                (Some(pinned), Some(alias_provider), Some(_)) if pinned != alias_provider => {
                    anyhow::bail!(
                        "agent '{agent_id}': model alias '{}' resolves to provider '{alias_provider}', which conflicts with the agent's provider '{pinned}'",
                        selection.model,
                    );
                }
                _ => {}
            }
            config.model_alias = resolved.alias;
            let provider = resolved.provider.or_else(|| selection.provider.clone());
            apply_bound_provider_auth(
                agent_id,
                config,
                provider.as_deref(),
                selection.auth.as_deref(),
                store,
            )?;
            // Config load and CLI overrides already refused a decision-model
            // host; a pin replaces the provider after both.
            crate::config::ensure_serves_chat(&config.provider)
                .map_err(|error| anyhow::anyhow!("agent '{agent_id}': {error}"))?;
            config.model = resolved.model;
            Ok(())
        }
    }
}

/// Applies optional provider/auth pins from an agent definition onto a host clone.
///
/// - Explicit `auth` wins and must resolve to a known profile. When `provider` is
///   also set, it must accept that auth.
/// - Provider without `auth` keeps compatible host auth; otherwise it selects
///   an available target-provider login using the catalog's preference order,
///   falling back to the provider default when none is available.
/// - Auth without `provider` sets both from the auth profile.
fn apply_bound_provider_auth(
    agent_id: &str,
    config: &mut Config,
    provider: Option<&str>,
    auth: Option<&str>,
    store: &dyn CredentialStore,
) -> anyhow::Result<()> {
    use rho_providers::provider::{
        legacy_provider_alias, resolve_auth_mode, resolve_profile_exact,
        resolve_provider_reference_with_credentials,
    };

    match (provider, auth) {
        (None, None) => Ok(()),
        (None, Some(auth)) => {
            let (descriptor, mode) = resolve_auth_mode(auth).ok_or_else(|| {
                anyhow::anyhow!("agent '{agent_id}': unknown auth profile '{auth}'")
            })?;
            config.provider = descriptor.name.to_string();
            config.auth = mode.id.to_string();
            Ok(())
        }
        (Some(provider), None) => {
            // Only canonical references can inherit host auth. Legacy aliases
            // keep their fixed auth through the credential-aware resolver below.
            if let Some((_, host_mode)) = resolve_auth_mode(&config.auth)
                .filter(|_| legacy_provider_alias(provider).is_none())
            {
                if let Ok(profile) = resolve_profile_exact(provider, host_mode.id) {
                    config.provider = profile.provider_name().to_string();
                    config.auth = profile.auth_id().to_string();
                    return Ok(());
                }
            }
            let profile = resolve_provider_reference_with_credentials(provider, store)
                .map_err(|error| bind_profile_error(agent_id, provider, None, error))?;
            config.provider = profile.provider_name().to_string();
            config.auth = profile.auth_id().to_string();
            Ok(())
        }
        (Some(provider), Some(auth)) => {
            let profile = resolve_profile_exact(provider, auth)
                .map_err(|error| bind_profile_error(agent_id, provider, Some(auth), error))?;
            config.provider = profile.provider_name().to_string();
            config.auth = profile.auth_id().to_string();
            Ok(())
        }
    }
}

fn bind_profile_error(
    agent_id: &str,
    provider: &str,
    auth: Option<&str>,
    error: rho_providers::provider::ProfileResolutionError,
) -> anyhow::Error {
    match auth {
        Some(auth) => {
            anyhow::anyhow!("agent '{agent_id}': provider '{provider}' auth '{auth}': {error}")
        }
        None => anyhow::anyhow!("agent '{agent_id}': provider '{provider}': {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ShellTool;

    #[test]
    fn configured_shell_withholds_the_other_capability() {
        let mut config = Config::default();
        config.shell_tool = ShellTool::Bash;
        let mut capabilities = AgentCapabilities::all_host_tools();
        retain_configured_shell(&mut capabilities, &config);
        assert!(capabilities.contains(&ToolCapability::Bash));
        assert!(!capabilities.contains(&ToolCapability::Powershell));

        config.shell_tool = ShellTool::PowerShell;
        let mut capabilities = AgentCapabilities::all_host_tools();
        retain_configured_shell(&mut capabilities, &config);
        assert!(capabilities.contains(&ToolCapability::Powershell));
        assert!(!capabilities.contains(&ToolCapability::Bash));
    }

    #[test]
    fn auto_shell_keeps_the_platform_default() {
        let config = Config::default();
        assert_eq!(config.shell_tool, ShellTool::Auto);
        let mut capabilities = AgentCapabilities::all_host_tools();
        retain_configured_shell(&mut capabilities, &config);
        #[cfg(windows)]
        {
            assert!(capabilities.contains(&ToolCapability::Powershell));
            assert!(!capabilities.contains(&ToolCapability::Bash));
        }
        #[cfg(not(windows))]
        {
            assert!(capabilities.contains(&ToolCapability::Bash));
            assert!(!capabilities.contains(&ToolCapability::Powershell));
        }
    }
}

#[cfg(test)]
mod shell_selection_tests {
    use crate::config::ShellTool;

    #[test]
    fn powershell_is_only_selectable_on_windows() {
        let all = ShellTool::all();
        #[cfg(windows)]
        assert!(all.contains(&ShellTool::PowerShell));
        #[cfg(not(windows))]
        assert!(!all.contains(&ShellTool::PowerShell));
    }

    #[test]
    fn powershell_pin_degrades_to_bash_off_windows() {
        #[cfg(windows)]
        assert_eq!(
            ShellTool::PowerShell.resolved(),
            rho_tools::ShellKind::PowerShell
        );
        #[cfg(not(windows))]
        assert_eq!(
            ShellTool::PowerShell.resolved(),
            rho_tools::ShellKind::Bash
        );
    }
}
