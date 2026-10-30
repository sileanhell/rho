//! Anthropic `/v1/models` discovery.
//!
//! Capability projection (thinking mode, effort, Off encoding) lives in
//! [`policy`] so fetch transport and wire/picker semantics stay separate.

use reqwest::Url;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::{credentials::CredentialStore, model::ModelError};

use super::{load_api_key_auth, provider_models_client, ProviderModel};

#[path = "anthropic_policy.rs"]
mod policy;

#[cfg(test)]
pub(crate) use policy::thinking_mode_from_value;
pub(crate) use policy::{
    cached_thinking_mode, capabilities_json_is_known, dated_parent_model, is_pre_thinking_haiku,
    supports_per_message_effort, AnthropicThinkingMode, OffThinking,
};

/// Upper bound on `/v1/models` pages so a misbehaving cursor cannot hang the
/// startup refresh.
const MAX_MODEL_PAGES: usize = 20;

#[derive(Deserialize)]
struct AnthropicModelsResponse {
    data: Vec<AnthropicModel>,
    #[serde(default)]
    has_more: bool,
    last_id: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicModel {
    id: String,
    display_name: Option<String>,
    max_input_tokens: Option<u64>,
    max_tokens: Option<u64>,
    #[serde(default)]
    capabilities: Option<Value>,
}

enum ModelListContinuation {
    Done,
    Next { after_id: String },
}

fn model_list_continuation(
    has_more: bool,
    last_id: Option<String>,
    after_id: Option<&str>,
) -> ModelListContinuation {
    if !has_more {
        return ModelListContinuation::Done;
    }
    let Some(next_after_id) = last_id else {
        return ModelListContinuation::Done;
    };
    if after_id == Some(next_after_id.as_str()) {
        // The cursor did not advance; stop instead of refetching the page.
        return ModelListContinuation::Done;
    }
    ModelListContinuation::Next {
        after_id: next_after_id,
    }
}

fn model_list_truncated(max_pages: usize) -> ModelError {
    ModelError::InvalidResponse(format!(
        "Anthropic /v1/models exceeded {max_pages} pages while more results remain"
    ))
}

fn records_from_listed_models(
    provider: &str,
    response: AnthropicModelsResponse,
) -> Vec<super::ProviderModelRecord> {
    response
        .data
        .into_iter()
        .map(|model| {
            let raw_json = policy::capabilities_json(model.capabilities);
            // `capabilities_json` always yields a parseable object, including
            // `{}` when the API omitted capabilities.
            let reasoning_capabilities = policy::AnthropicModelCapabilities::from_value(&raw_json)
                .expect("capabilities_json always stores a parseable object")
                .reasoning_capabilities(&model.id);
            super::ProviderModelRecord {
                model: ProviderModel {
                    provider: provider.to_string(),
                    display_name: model.display_name.unwrap_or_else(|| model.id.clone()),
                    context_window: model.max_input_tokens.filter(|window| *window > 0),
                    max_output_tokens: model.max_tokens,
                    model: model.id,
                    reasoning_capabilities,
                },
                raw_json,
            }
        })
        .collect()
}

fn add_listed_page(
    models: &mut Vec<super::ProviderModelRecord>,
    provider: &str,
    response: AnthropicModelsResponse,
    after_id: Option<&str>,
) -> ModelListContinuation {
    let has_more = response.has_more;
    let last_id = response.last_id.clone();
    models.extend(records_from_listed_models(provider, response));
    model_list_continuation(has_more, last_id, after_id)
}

fn finalize_models(mut models: Vec<super::ProviderModelRecord>) -> Vec<super::ProviderModelRecord> {
    models.sort_by(|left, right| left.model.model.cmp(&right.model.model));
    models.dedup_by(|left, right| left.model.model == right.model.model);
    models
}

/// Static subscription catalog used when the credential is an OAuth grant.
///
/// OAuth access tokens (`sk-ant-oat*`, `cc-*`, JWT) are inference-scoped and
/// cannot call `/v1/models`, so the pickable set is the fixed subscription
/// menu — the same reason Claude Code hardcodes its model list. Capabilities
/// use the Models API shape so the shared projection still derives thinking
/// mode, effort levels, and Off encoding per family.
/// 5.x/4.6+ rows: adaptive thinking plus the full effort vocabulary.
/// `off` drives the `disabled` leaf: `Some(false)` marks always-thinking
/// families; `None` omits it so the family shim picks the encoding (the
/// Sonnet 5.5 Off is `between_tools`, not `disabled`).
fn adaptive_capabilities(off: Option<bool>) -> Value {
    let mut types = json!({ "adaptive": { "supported": true } });
    if let Some(selectable) = off {
        types["disabled"] = json!({ "supported": selectable });
    }
    json!({
        "thinking": { "types": types },
        "effort": {
            "supported": true,
            "low": { "supported": true },
            "medium": { "supported": true },
            "high": { "supported": true },
            "xhigh": { "supported": true },
            "max": { "supported": true },
        },
    })
}

/// 4.x rows: legacy budget-token thinking.
fn budget_capabilities() -> Value {
    json!({
        "thinking": {
            "types": {
                "enabled": { "supported": true },
                "disabled": { "supported": true },
            },
        },
    })
}

struct StaticModel {
    id: &'static str,
    display_name: &'static str,
    max_input_tokens: u64,
    max_tokens: u64,
    capabilities: Value,
}

fn static_subscription_models() -> Vec<StaticModel> {
    vec![
        StaticModel {
            id: "claude-fable-5-1",
            display_name: "Claude Fable 5.1",
            max_input_tokens: 1_000_000,
            max_tokens: 128_000,
            capabilities: adaptive_capabilities(Some(false)),
        },
        StaticModel {
            id: "claude-fable-5",
            display_name: "Claude Fable 5",
            max_input_tokens: 1_000_000,
            max_tokens: 128_000,
            capabilities: adaptive_capabilities(Some(false)),
        },
        StaticModel {
            id: "claude-opus-5-5",
            display_name: "Claude Opus 5.5",
            max_input_tokens: 1_000_000,
            max_tokens: 128_000,
            capabilities: adaptive_capabilities(Some(false)),
        },
        StaticModel {
            id: "claude-opus-5",
            display_name: "Claude Opus 5",
            max_input_tokens: 1_000_000,
            max_tokens: 128_000,
            capabilities: adaptive_capabilities(Some(true)),
        },
        StaticModel {
            id: "claude-opus-5-fast",
            display_name: "Claude Opus 5 Fast",
            max_input_tokens: 1_000_000,
            max_tokens: 128_000,
            capabilities: adaptive_capabilities(Some(true)),
        },
        StaticModel {
            id: "claude-opus-4-8",
            display_name: "Claude Opus 4.8",
            max_input_tokens: 1_000_000,
            max_tokens: 128_000,
            capabilities: adaptive_capabilities(Some(true)),
        },
        StaticModel {
            id: "claude-opus-4-8-fast",
            display_name: "Claude Opus 4.8 Fast",
            max_input_tokens: 1_000_000,
            max_tokens: 128_000,
            capabilities: adaptive_capabilities(Some(true)),
        },
        StaticModel {
            id: "claude-sonnet-5-5",
            display_name: "Claude Sonnet 5.5",
            max_input_tokens: 1_000_000,
            max_tokens: 128_000,
            // No `disabled` leaf: Sonnet 5.5's documented Off is
            // `between_tools`, resolved by the family shim.
            capabilities: adaptive_capabilities(None),
        },
        StaticModel {
            id: "claude-sonnet-5",
            display_name: "Claude Sonnet 5",
            max_input_tokens: 1_000_000,
            max_tokens: 128_000,
            capabilities: adaptive_capabilities(Some(true)),
        },
        StaticModel {
            id: "claude-mythos-5-1",
            display_name: "Claude Mythos 5.1",
            max_input_tokens: 1_000_000,
            max_tokens: 128_000,
            capabilities: adaptive_capabilities(Some(false)),
        },
        StaticModel {
            id: "claude-haiku-5-5",
            display_name: "Claude Haiku 5.5",
            max_input_tokens: 1_000_000,
            max_tokens: 128_000,
            capabilities: adaptive_capabilities(Some(true)),
        },
        StaticModel {
            id: "claude-sonnet-4-5",
            display_name: "Claude Sonnet 4.5",
            max_input_tokens: 200_000,
            max_tokens: 64_000,
            capabilities: budget_capabilities(),
        },
        StaticModel {
            id: "claude-opus-4-1",
            display_name: "Claude Opus 4.1",
            max_input_tokens: 200_000,
            max_tokens: 32_000,
            capabilities: budget_capabilities(),
        },
        StaticModel {
            id: "claude-haiku-4-5",
            display_name: "Claude Haiku 4.5",
            max_input_tokens: 200_000,
            max_tokens: 64_000,
            // Haiku through 4.5 accepts no thinking parameter; no control
            // surface is advertised (is_pre_thinking_haiku gates the wire).
            capabilities: json!({}),
        },
    ]
}

/// Static fallback: context window, output cap and capabilities for `id`,
/// matched by exact id then longest id-prefix (dated snapshots inherit the
/// parent row). `None` for models the static menu does not cover.
fn static_model_for(id: &str) -> Option<StaticModel> {
    let mut best: Option<StaticModel> = None;
    let mut best_len = 0usize;
    for model in static_subscription_models() {
        let matches = id == model.id
            || id
                .strip_prefix(model.id)
                .is_some_and(|suffix| suffix.starts_with('-'));
        if matches && model.id.len() > best_len {
            best_len = model.id.len();
            best = Some(model);
        }
    }
    best
}

fn oauth_fallback_records(provider: &str) -> Vec<super::ProviderModelRecord> {
    records_from_listed_models(
        provider,
        AnthropicModelsResponse {
            data: static_subscription_models()
                .into_iter()
                .map(|model| AnthropicModel {
                    id: model.id.to_string(),
                    display_name: Some(model.display_name.to_string()),
                    max_input_tokens: Some(model.max_input_tokens),
                    max_tokens: Some(model.max_tokens),
                    capabilities: Some(model.capabilities),
                })
                .collect(),
            has_more: false,
            last_id: None,
        },
    )
}

pub(super) async fn fetch(
    provider: &str,
    store: &dyn CredentialStore,
) -> Result<Vec<super::ProviderModelRecord>, ModelError> {
    let token = crate::auth::provider_credentials::resolve_anthropic_request_auth(store).await?;
    // The subscription menu lives on the bootstrap endpoint, not
    // `/v1/models`; keep the static menu as the fallback when the fetch
    // cannot run or yields nothing pickable.
    Ok(fetch_bootstrap(provider, &token)
        .await
        .filter(|records| !records.is_empty())
        .unwrap_or_else(|| oauth_fallback_records(provider)))
}

/// Subscription model menu from `GET /api/claude_cli/bootstrap`, mirroring
/// Claude Code's OAuth discovery. Answers `additional_model_options`
/// (display names, disabled reasons) plus `model_access` entitlements;
/// capability data is not served there, so context/capabilities come from
/// [`static_model_for`], defaulting unknown families to adaptive thinking.
#[derive(Deserialize)]
struct BootstrapResponse {
    #[serde(default)]
    additional_model_options: Vec<BootstrapModelOption>,
    #[serde(default)]
    model_access: Vec<BootstrapModelAccess>,
}

#[derive(Deserialize)]
struct BootstrapModelOption {
    model: String,
    name: Option<String>,
    disabled_reason: Option<String>,
}

#[derive(Deserialize)]
struct BootstrapModelAccess {
    api_name: String,
    entitled: Option<bool>,
}

async fn fetch_bootstrap(
    provider: &str,
    token: &str,
) -> Option<Vec<super::ProviderModelRecord>> {
    let response: BootstrapResponse = provider_models_client()
        .ok()?
        .get("https://api.anthropic.com/api/claude_cli/bootstrap")
        .bearer_auth(token)
        .header("anthropic-version", "2023-06-01")
        .header(
            "user-agent",
            format!(
                "claude-code/{} (external, cli)",
                crate::auth::anthropic_oauth::claude_code_version()
            ),
        )
        .header("x-app", "cli")
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
        .ok()?
        .error_for_status()
        .ok()?
        .json()
        .await
        .ok()?;
    let entitled: std::collections::HashMap<&str, bool> = response
        .model_access
        .iter()
        .map(|access| (access.api_name.as_str(), access.entitled.unwrap_or(true)))
        .collect();
    let mut models = Vec::new();
    for option in response.additional_model_options {
        if option.disabled_reason.is_some() {
            continue;
        }
        if entitled.get(option.model.as_str()) == Some(&false) {
            continue;
        }
        let static_row = static_model_for(&option.model);
        let raw_json = policy::capabilities_json(
            static_row
                .as_ref()
                .map(|row| row.capabilities.clone())
                .or_else(|| Some(adaptive_capabilities(None))),
        );
        let reasoning_capabilities = policy::AnthropicModelCapabilities::from_value(&raw_json)
            .expect("capabilities_json always stores a parseable object")
            .reasoning_capabilities(&option.model);
        models.push(super::ProviderModelRecord {
            model: ProviderModel {
                provider: provider.to_string(),
                display_name: option.name.unwrap_or_else(|| option.model.clone()),
                context_window: static_row
                    .as_ref()
                    .map(|row| row.max_input_tokens)
                    .filter(|window| *window > 0),
                max_output_tokens: static_row.as_ref().map(|row| row.max_tokens),
                model: option.model,
                reasoning_capabilities,
            },
            raw_json,
        });
    }
    // Bootstrap rows not covered by additional_model_options still come
    // through model_access.
    let covered: std::collections::HashSet<String> = models
        .iter()
        .map(|record| record.model.model.clone())
        .collect();
    for access in response.model_access {
        if access.entitled == Some(false) || covered.contains(access.api_name.as_str()) {
            continue;
        }
        let static_row = static_model_for(&access.api_name);
        let raw_json = policy::capabilities_json(
            static_row
                .as_ref()
                .map(|row| row.capabilities.clone())
                .or_else(|| Some(adaptive_capabilities(None))),
        );
        let reasoning_capabilities = policy::AnthropicModelCapabilities::from_value(&raw_json)
            .expect("capabilities_json always stores a parseable object")
            .reasoning_capabilities(&access.api_name);
        models.push(super::ProviderModelRecord {
            model: ProviderModel {
                provider: provider.to_string(),
                display_name: access.api_name.clone(),
                context_window: static_row
                    .as_ref()
                    .map(|row| row.max_input_tokens)
                    .filter(|window| *window > 0),
                max_output_tokens: static_row.as_ref().map(|row| row.max_tokens),
                model: access.api_name,
                reasoning_capabilities,
            },
            raw_json,
        });
    }
    Some(models)
}

pub(super) async fn fetch_compatible(
    provider: &str,
    api_base: &Url,
    store: &dyn CredentialStore,
) -> Result<Vec<super::ProviderModelRecord>, ModelError> {
    let base = Url::parse(&format!(
        "{}/models",
        api_base.as_str().trim_end_matches('/')
    ))
    .map_err(|err| ModelError::InvalidResponse(format!("invalid Anthropic models URL: {err}")))?;
    fetch_models(provider, store, base).await
}

async fn fetch_models(
    provider: &str,
    store: &dyn CredentialStore,
    base: Url,
) -> Result<Vec<super::ProviderModelRecord>, ModelError> {
    let key = load_api_key_auth(provider, store)?;
    let client = provider_models_client()?;
    let mut models = Vec::new();
    let mut after_id = None::<String>;
    for _ in 0..MAX_MODEL_PAGES {
        let mut url = base.clone();
        if let Some(after_id) = &after_id {
            url.query_pairs_mut().append_pair("after_id", after_id);
        }
        let response: AnthropicModelsResponse = client
            .get(url)
            .header("x-api-key", &key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        match add_listed_page(&mut models, provider, response, after_id.as_deref()) {
            ModelListContinuation::Done => return Ok(finalize_models(models)),
            ModelListContinuation::Next {
                after_id: next_after_id,
            } => after_id = Some(next_after_id),
        }
    }
    Err(model_list_truncated(MAX_MODEL_PAGES))
}
