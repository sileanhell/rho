//! Runtime selection model for internal agents: which harness runs an agent,
//! and the settings only that harness understands.
//!
//! The on-disk shape of these selections lives in `config_format`.

use {crate::model_aliases::ModelAliases, rho_providers::reasoning::ReasoningLevel};

/// How a decision-model entry, such as the permission screen, asks its model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelKind {
    /// Over a decision API (System One or OpenAI's Decisions), on a host
    /// that serves decision models.
    Decision,
    /// As a chat model, through the text adapter.
    Text,
}

/// Provider selection for an internal agent that runs on Rho's own stack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RhoInternalAgentModel {
    pub provider: String,
    pub model: String,
    pub auth: String,
    /// The `kind` key. Only decision-model entries read it; `None` takes the
    /// entry's default.
    pub kind: Option<ModelKind>,
    /// The `allow_threshold_percent` key. Only a decision-model permission
    /// screen reads it; `None` takes the screen's default.
    pub allow_threshold_percent: Option<u8>,
    pub(super) model_alias: Option<String>,
}

/// Which harness runs an internal agent, together with the settings only that
/// harness understands.
///
/// The runtime axis travels as one value, mirroring `AgentRuntimeSpec`, so a
/// stored selection cannot pair one harness with another harness's model
/// vocabulary or auth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InternalAgentTarget {
    /// Rho's own provider stack, with an explicit provider, model, and auth.
    Rho(RhoInternalAgentModel),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InternalAgentModelConfig {
    pub target: InternalAgentTarget,
    /// Per-agent reasoning override. `None` keeps the agent definition default.
    pub reasoning: Option<ReasoningLevel>,
}

impl InternalAgentModelConfig {
    /// A selection on Rho's own runtime.
    pub fn new(provider: String, model: String, auth: String) -> Self {
        Self {
            target: InternalAgentTarget::Rho(RhoInternalAgentModel {
                provider,
                model,
                auth,
                kind: None,
                allow_threshold_percent: None,
                model_alias: None,
            }),
            reasoning: None,
        }
    }

    /// The Rho provider selection.
    pub fn rho(&self) -> Option<&RhoInternalAgentModel> {
        let InternalAgentTarget::Rho(model) = &self.target;
        Some(model)
    }

    /// How the selection reads on a status line, badge, or picker row.
    pub fn display_reference(&self) -> String {
        let InternalAgentTarget::Rho(selection) = &self.target;
        rho_providers::provider::model_reference(&selection.provider, &selection.model)
    }

    /// The Rho selection, for assertions that only make sense on Rho's stack.
    #[cfg(test)]
    pub fn expect_rho(&self) -> &RhoInternalAgentModel {
        self.rho().expect("selection runs on the rho runtime")
    }

    /// The Rho selection for mutation in tests.
    #[cfg(test)]
    pub fn expect_rho_mut(&mut self) -> &mut RhoInternalAgentModel {
        let InternalAgentTarget::Rho(model) = &mut self.target;
        model
    }

    pub(super) fn current_alias<'a>(&'a self, aliases: &'a ModelAliases) -> Option<&'a str> {
        let selection = self.rho()?;
        let name = selection.model_alias.as_deref()?;
        let target = aliases.get(name)?;
        (target.model == selection.model
            && target.provider.as_deref().unwrap_or(&selection.provider) == selection.provider)
            .then_some(name)
    }
}
