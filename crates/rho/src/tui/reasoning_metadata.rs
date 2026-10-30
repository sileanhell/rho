use super::{config_picker, App, ComposerMode, Entry, InteractiveRuntime};
use rho_providers::{
    model::{
        models_dev, ModelMetadata, ReasoningCapabilities, ReasoningRequestSource,
        ReasoningResolution,
    },
    reasoning::ReasoningLevel,
};

pub(super) struct FetchedReasoningResolution {
    pub(super) effective: ReasoningLevel,
    pub(super) rejected: Option<ReasoningLevel>,
}

impl super::TuiBootstrap {
    pub(super) fn set_reasoning(&mut self, level: ReasoningLevel, source: ReasoningRequestSource) {
        self.runtime.reasoning = level;
        self.runtime.reasoning_source = source;
        self.services.diagnostics.update_identity(
            &self.runtime.provider,
            &self.runtime.model,
            level,
        );
    }
}

pub(super) fn cached_metadata(provider: &str, model: &str) -> Option<(ModelMetadata, bool)> {
    let metadata = models_dev::cached_model_metadata(provider, model)?;
    let is_current = !models_dev::model_metadata_needs_refresh(provider, model);
    Some((metadata, is_current))
}

pub(super) use crate::app::conversation_switch::{
    resolve_model_switch_reasoning, ModelSwitchReasoningPolicy, ModelSwitchReasoningResolution,
};

pub(super) fn resolve_fetched_reasoning(
    capabilities: &ReasoningCapabilities,
    current: ReasoningLevel,
    at_fetch_start: Option<(ReasoningLevel, ReasoningRequestSource)>,
) -> FetchedReasoningResolution {
    let source = match at_fetch_start {
        Some((reasoning, _)) if reasoning != current => ReasoningRequestSource::Explicit,
        Some((_, source)) => source,
        None => ReasoningRequestSource::PersistedOrDefault,
    };
    let resolution = capabilities.resolve(current, source);
    if let ReasoningResolution::UnsupportedExplicit(requested) = resolution {
        let effective = at_fetch_start
            .and_then(|(reasoning, _)| {
                capabilities
                    .resolve(reasoning, ReasoningRequestSource::PersistedOrDefault)
                    .effective()
            })
            .unwrap_or(current);
        return FetchedReasoningResolution {
            effective,
            rejected: Some(requested),
        };
    }
    FetchedReasoningResolution {
        effective: resolution.effective().unwrap_or(current),
        rejected: None,
    }
}

fn reasoning_picker(
    capabilities: &ReasoningCapabilities,
    current: ReasoningLevel,
) -> super::UiPicker {
    let levels = capabilities.selectable_levels(&ReasoningLevel::ALL, None);
    let items = levels
        .iter()
        .map(|level| super::PickerItem {
            badge: (*level == current).then(|| super::PickerBadge {
                text: "selected".into(),
                tone: super::PickerBadgeTone::Selected,
            }),
            ..super::PickerItem::new(
                level.to_string(),
                format!("{}{level}", config_picker::REASONING_PREFIX),
            )
        })
        .collect();
    let mut picker = super::UiPicker::config("select reasoning", items);
    picker.selected = levels
        .iter()
        .position(|level| *level == current)
        .unwrap_or(0);
    picker
}

impl App {
    pub(super) fn open_reasoning_picker(&mut self, from_config: bool) {
        let capabilities = models_dev::current_reasoning_capabilities(
            &self.info.runtime.provider,
            &self.info.runtime.model,
        );
        let picker = reasoning_picker(&capabilities, self.info.runtime.reasoning);
        if from_config {
            self.open_child_picker(picker);
        } else {
            self.input_ui.set_composer(ComposerMode::Picker(picker));
        }
    }

    /// Apply catalog normalization only when it changes the effective reasoning.
    pub(super) async fn apply_fetched_reasoning(
        &mut self,
        agent: &mut InteractiveRuntime,
        capabilities: &ReasoningCapabilities,
        at_fetch_start: Option<(ReasoningLevel, ReasoningRequestSource)>,
    ) {
        let resolved =
            resolve_fetched_reasoning(capabilities, self.info.runtime.reasoning, at_fetch_start);
        let reasoning = resolved.effective;
        if let Some(requested) = resolved.rejected {
            self.insert_entry(&Entry::Error(format!(
                "reasoning level '{requested}' is not supported by {}/{}; restored '{reasoning}'",
                self.info.runtime.provider, self.info.runtime.model
            )));
        }
        // A metadata refresh is not a provider switch. Replacing an unchanged
        // provider discards the session's successful context calibration.
        if reasoning == self.info.runtime.reasoning {
            return;
        }
        let provider = match self
            .build_provider_for_selection(
                &self.info.runtime.provider,
                &self.info.runtime.model,
                reasoning,
                &self.info.runtime.auth,
            )
            .await
        {
            Ok(provider) => provider,
            Err(err) => {
                self.insert_entry(&Entry::Error(format!(
                    "could not apply model reasoning metadata: {err}"
                )));
                return;
            }
        };
        if let Err(err) = agent.replace_provider(provider, reasoning, &self.info.runtime.auth) {
            self.insert_entry(&Entry::Error(format!(
                "could not apply model reasoning metadata: {err}"
            )));
            return;
        }
        self.cache_warm_reasoning_changed(agent);
        self.info
            .set_reasoning(reasoning, ReasoningRequestSource::PersistedOrDefault);
        if let Err(err) = self.info.services.config_repository.update(|config| {
            config.reasoning = reasoning;
        }) {
            self.insert_entry(&Entry::Error(format!(
                "could not save normalized reasoning: {err}"
            )));
        }
    }

    pub(super) async fn cycle_reasoning(
        &mut self,
        agent: &mut InteractiveRuntime,
    ) -> anyhow::Result<()> {
        let capabilities = models_dev::current_reasoning_capabilities(
            &self.info.runtime.provider,
            &self.info.runtime.model,
        );
        if capabilities == ReasoningCapabilities::NotConfigurable {
            return Ok(());
        }
        let reasoning = capabilities.next_level(self.info.runtime.reasoning);
        self.select_reasoning(reasoning, agent).await
    }

    pub(super) async fn select_reasoning(
        &mut self,
        reasoning: ReasoningLevel,
        agent: &mut InteractiveRuntime,
    ) -> anyhow::Result<()> {
        let capabilities = models_dev::current_reasoning_capabilities(
            &self.info.runtime.provider,
            &self.info.runtime.model,
        );
        if capabilities
            .resolve(reasoning, ReasoningRequestSource::Explicit)
            .effective()
            .is_none()
        {
            self.set_status("reasoning level is not supported by this model");
            return Ok(());
        }
        let provider = match self
            .build_provider_for_selection(
                &self.info.runtime.provider,
                &self.info.runtime.model,
                reasoning,
                &self.info.runtime.auth,
            )
            .await
        {
            Ok(provider) => provider,
            Err(err) => {
                self.insert_entry(&Entry::Error(format!(
                    "could not update reasoning to {reasoning}: {err}"
                )));
                self.set_status("reasoning change failed");
                return Ok(());
            }
        };
        agent.replace_provider(provider, reasoning, &self.info.runtime.auth)?;
        self.cache_warm_reasoning_changed(agent);
        self.info
            .set_reasoning(reasoning, ReasoningRequestSource::Explicit);
        let save_result = self.info.services.config_repository.update(|config| {
            config.reasoning = reasoning;
        });
        let choosing_reasoning = matches!(
            self.input_ui.composer(),
            ComposerMode::Picker(picker) if picker.items.iter().any(|item| item.value.starts_with(config_picker::REASONING_PREFIX))
        );
        if choosing_reasoning {
            if !self.pop_picker_level() {
                self.input_ui.set_composer(ComposerMode::Input);
            }
        } else if matches!(
            self.input_ui.composer(),
            ComposerMode::Picker(picker) if picker.is_config()
        ) {
            let config = self
                .info
                .services
                .config_repository
                .load()
                .unwrap_or_default();
            self.info.runtime.show_reasoning_output = config.show_reasoning_output;
            self.info.runtime.zen_mode = config.zen_mode;
            self.refresh_main_config_picker(config_picker::REASONING_VALUE)?;
        }
        match save_result {
            Ok(()) => self.set_status(format!("reasoning: {reasoning}")),
            Err(err) => {
                self.insert_entry(&Entry::Error(format!(
                    "reasoning set to {reasoning} for this session, but saving config failed: {err}"
                )));
                self.set_status("config save failed");
            }
        }
        Ok(())
    }
}
