use std::sync::{Mutex, OnceLock};

use crate::{
    auth::{
        anthropic_oauth::{self, is_native_anthropic_host},
        anthropic_token::{AnthropicAuthManager, AnthropicAuthMaterial},
    },
    model::ModelIdentity,
    protocol::anthropic_messages::{
        collect_anthropic_sse_response, convert_anthropic_response, split_system_and_messages,
        to_anthropic_tool, AnthropicCacheControl, AnthropicContentBlock, AnthropicMessage,
        AnthropicRequest, AnthropicResponse, AnthropicRole, AnthropicSystemBlock,
        AnthropicThinkingConfig, ProviderContextReplay,
    },
    provider_backend::{ModelError, ModelEvent, ModelRequest, ModelResponse},
};

#[cfg(test)]
use crate::provider_backend::stream_timeout::provider_client;

#[cfg(test)]
const ANTHROPIC_API_BASE: &str = "https://api.anthropic.com/v1";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Beta headers safe on first-party Anthropic requests. GA on Claude 4.6+
/// (harmless no-op there) but older Claude still gates on them.
const COMMON_BETAS: &str = "interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14";
/// Betas required for OAuth/subscription auth, matching Claude Code.
const OAUTH_BETAS: &str = "claude-code-20250219,oauth-2025-04-20";
/// First line of the system prompt on the OAuth wire: Anthropic's classifier
/// expects the Claude Code identity behind OAuth tokens.
const CLAUDE_CODE_SYSTEM_PREFIX: &str = "You are Claude Code, Anthropic's official CLI for Claude.";
/// OAuth tool-name prefix so the wire fingerprint matches Claude Code's MCP
/// tools; reversed on response.
const MCP_TOOL_PREFIX: &str = "mcp__";
/// Tool names Anthropic's OAuth billing classifier fingerprints as a
/// third-party app; aliased on the OAuth wire only.
const OAUTH_TOOL_NAME_ALIASES: &[(&str, &str)] = &[
    ("session_search", "chat_history_lookup"),
    ("memory", "context_notes"),
];
/// Tool names that are also safe to rewrite inside prose (tool descriptions,
/// system text). `memory` is ordinary English inside its own parameter docs,
/// so only `session_search` is prose-safe — same split as Hermes.
const OAUTH_PROSE_ALIASES: &[(&str, &str)] = &[("session_search", "chat_history_lookup")];
/// Product-name replacements on the OAuth wire: harness branding in prose is
/// the other signal Anthropic's billing classifier uses to reroute the
/// request to metered extra usage.
const OAUTH_PRODUCT_REPLACEMENTS: &[(&str, &str)] = &[("Rho", "Claude Code")];
/// `speed: "fast"` beta, accepted by Opus 4.8 / 5 / 5.5 on the first-party
/// host. The `-fast` catalog rows map onto their base id plus this speed.
const FAST_MODE_BETA: &str = "fast-mode-2026-02-01";
/// Overrides the wire `cache_control.ttl` on the native route
/// (`"5m"`/`"1h"`). Mirrors Claude Code's `CLAUDE_CODE_PROMPT_CACHE_TTL`.
const CACHE_TTL_ENV: &str = "RHO_ANTHROPIC_CACHE_TTL";
/// `cache_control.scope` value for account-shared prefixes.
const GLOBAL_CACHE_SCOPE: &str = "global";
pub const DEFAULT_MAX_TOKENS: u32 = 4096;
/// First-party output ceiling while neither catalog knows the model. Thinking
/// counts against `max_tokens` even when its text is omitted, so a
/// thinking-off-sized fallback truncates always-thinking Claude models.
///
/// Sized to the smallest `max_output_tokens` on any current `/v1/models` row
/// (claude-opus-4-1: 32,000; Claude 4.5+ and 5.x rows report 64k-128k), so a
/// later hydrate raises rather than lowers it. Hosted Messages adapters keep
/// [`DEFAULT_MAX_TOKENS`] because their caps are unknown.
const ANTHROPIC_COLD_CATALOG_MAX_TOKENS: u32 = 32_000;
pub(crate) const ANTHROPIC_ANSWER_RESERVE_TOKENS: u32 = 1_024;

mod latches;
mod per_message_effort;
mod thinking;

/// Credential material accepted by [`AnthropicProvider`].
///
/// `ApiKey` authenticates as `x-api-key`. `OAuth` authenticates as
/// `Authorization: Bearer` with the Claude Code identity headers and wire
/// transforms Anthropic expects behind subscription tokens; on the native
/// host it also refreshes the single-use grant on expiry or 401.
pub enum AnthropicAuth {
    ApiKey(String),
    OAuth(AnthropicAuthManager),
}

impl AnthropicAuth {
    pub fn api_key(key: impl Into<String>) -> Self {
        Self::ApiKey(key.into())
    }

    pub(crate) fn oauth(manager: AnthropicAuthManager) -> Self {
        Self::OAuth(manager)
    }
}

impl std::fmt::Debug for AnthropicAuth {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self {
            Self::ApiKey(_) => "api-key",
            Self::OAuth(_) => "oauth",
        };
        formatter
            .debug_struct("AnthropicAuth")
            .field("kind", &kind)
            .field("secret", &"[REDACTED]")
            .finish()
    }
}

/// Resolved auth for one request: the credential plus whether the Claude
/// Code OAuth identity applies (native first-party route only — hosted
/// Anthropic-compatible endpoints never get it).
struct RequestAuth {
    access_token: String,
    oauth: bool,
}

pub struct AnthropicProvider {
    client: reqwest::Client,
    auth: AnthropicAuth,
    api_base: String,
    identity_provider: &'static str,
    model: String,
    /// `max_tokens` sent while neither model catalog knows the model.
    cold_max_tokens: u32,
    /// Fixed max-tokens for tests; `None` resolves from the model catalog per request.
    max_tokens_override: Option<u32>,
    /// Test-only thinking snapshot. Production resolves per request so a
    /// catalog hydrate that finishes after construction still applies.
    thinking_override: Option<thinking::ThinkingSource>,
    /// First live `max_tokens` used to clamp a thinking budget. Later catalog
    /// hydrates may raise the request `max_tokens`, but the budget stays put so
    /// a hydrate cannot rewrite thinking params and bust the message cache. See
    /// [`Self::thinking_budget_ceiling`] for a hydrate that lowers it.
    thinking_budget_ceiling: OnceLock<u32>,
    /// Per-conversation prefix effort and later shifts, keyed by prompt cache
    /// key, so a mid-session change does not rewrite top-level effort.
    per_message_effort: Mutex<per_message_effort::PerMessageEffortState>,
    /// Sticky per-conversation downgrades latched after classified 400s.
    latches: Mutex<latches::ConversationLatches>,
    /// Explicit `cache_control.ttl` override (`"5m"`/`"1h"`): the env var on
    /// the native route, `providers.custom.*.prompt_cache_ttl` on hosted
    /// adapters. `None` resolves per request from the auth shape.
    configured_cache_ttl: Option<&'static str>,
    session_headers: super::opencode::SessionHeaders,
    compression: super::request_body::EndpointCompression,
}

impl AnthropicProvider {
    // Tests construct unresolved and inject capabilities explicitly via
    // `with_thinking` and a fixed max-tokens, so they never touch the on-disk
    // model cache.
    #[cfg(test)]
    pub fn new(model: String, api_key: String) -> Self {
        let thinking_override = Some(thinking::ThinkingSource::unresolved(&model));
        Self {
            client: provider_client(),
            auth: AnthropicAuth::ApiKey(api_key),
            api_base: ANTHROPIC_API_BASE.into(),
            identity_provider: "anthropic",
            model,
            cold_max_tokens: ANTHROPIC_COLD_CATALOG_MAX_TOKENS,
            max_tokens_override: Some(DEFAULT_MAX_TOKENS),
            thinking_override,
            session_headers: super::opencode::SessionHeaders::new("anthropic"),
            compression: super::request_body::EndpointCompression::Verified,
            thinking_budget_ceiling: OnceLock::new(),
            per_message_effort: Mutex::new(per_message_effort::PerMessageEffortState::default()),
            latches: Mutex::new(latches::ConversationLatches::default()),
            configured_cache_ttl: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_thinking(mut self, thinking: thinking::ThinkingSource) -> Self {
        self.thinking_override = Some(thinking);
        self
    }

    #[cfg(test)]
    pub(crate) fn set_max_tokens_override(&mut self, tokens: u32) {
        self.max_tokens_override = Some(tokens);
    }

    /// First-party Anthropic API.
    pub(crate) fn new_with_transport(
        model: String,
        auth: AnthropicAuth,
        client: reqwest::Client,
        api_base: String,
    ) -> Self {
        Self::with_cold_max_tokens(
            model,
            auth,
            client,
            api_base,
            "anthropic",
            ANTHROPIC_COLD_CATALOG_MAX_TOKENS,
        )
        .with_cache_ttl(env_cache_ttl())
    }

    /// Hosted Anthropic Messages adapter registered under `identity_provider`.
    pub(crate) fn new_with_identity(
        model: String,
        api_key: String,
        client: reqwest::Client,
        api_base: String,
        identity_provider: &'static str,
    ) -> Self {
        Self::with_cold_max_tokens(
            model,
            AnthropicAuth::ApiKey(api_key),
            client,
            api_base,
            identity_provider,
            DEFAULT_MAX_TOKENS,
        )
    }

    /// Replaces the built-in endpoint allowlist for Messages request bodies.
    pub(crate) fn with_request_compression(
        mut self,
        compression: super::request_body::EndpointCompression,
    ) -> Self {
        self.compression = compression;
        self
    }

    /// Overrides the `cache_control.ttl` sent on breakpoints. Anthropic only
    /// documents `"5m"` and `"1h"`, so other durations snap to the nearest
    /// tier; `None` keeps the current setting so overrides can layer over
    /// the env fallback.
    pub(crate) fn with_cache_ttl(mut self, ttl: Option<std::time::Duration>) -> Self {
        if let Some(ttl) = ttl {
            self.configured_cache_ttl = Some(if ttl <= std::time::Duration::from_secs(300) {
                "5m"
            } else {
                "1h"
            });
        }
        self
    }

    fn with_cold_max_tokens(
        model: String,
        auth: AnthropicAuth,
        client: reqwest::Client,
        api_base: String,
        identity_provider: &'static str,
        cold_max_tokens: u32,
    ) -> Self {
        Self {
            client,
            auth,
            api_base,
            identity_provider,
            model,
            cold_max_tokens,
            max_tokens_override: None,
            thinking_override: None,
            session_headers: super::opencode::SessionHeaders::new(identity_provider),
            compression: super::request_body::EndpointCompression::Verified,
            thinking_budget_ceiling: OnceLock::new(),
            per_message_effort: Mutex::new(per_message_effort::PerMessageEffortState::default()),
            latches: Mutex::new(latches::ConversationLatches::default()),
            configured_cache_ttl: None,
        }
    }

    fn thinking_source(&self) -> thinking::ThinkingSource {
        self.thinking_override.clone().unwrap_or_else(|| {
            thinking::ThinkingSource::resolve(self.identity_provider, &self.model)
        })
    }

    /// Resolved per request so a catalog hydrate that finishes after
    /// construction still supplies the real output budget.
    fn max_tokens(&self) -> u32 {
        if let Some(tokens) = self.max_tokens_override {
            return tokens;
        }
        crate::model::provider_models::cached_provider_model(self.identity_provider, &self.model)
            .and_then(|metadata| metadata.max_output_tokens)
            .or_else(|| {
                crate::model::models_dev::cached_model_metadata(self.identity_provider, &self.model)
                    .and_then(|metadata| metadata.max_output_tokens)
            })
            .and_then(|tokens| u32::try_from(tokens).ok())
            .unwrap_or(self.cold_max_tokens)
    }

    /// Ceiling used to clamp `budget_tokens`. Latched on the first request so a
    /// later catalog hydrate can still raise `max_tokens` without rewriting the
    /// thinking params that sit next to the cached message prefix. A hydrate
    /// that lowers `max_tokens` below the latch wins: a budget at or above
    /// `max_tokens` is a 400, which is worse than one cache miss.
    fn thinking_budget_ceiling(&self, live_max_tokens: u32) -> u32 {
        (*self.thinking_budget_ceiling.get_or_init(|| live_max_tokens)).min(live_max_tokens)
    }

    /// Breakpoint TTL for this request. Mirrors Claude Code's policy: the
    /// subscription (OAuth) route defaults to `"1h"` because plan-priced cache
    /// writes make the longer TTL strictly better; API-key requests keep the
    /// explicit `"5m"` default since an hour write costs 2x input. Hosted
    /// adapters send a TTL only when one was configured — a proxy that does
    /// not know the field latches `ttl_refused` and retries plain. A latched
    /// refusal drops the leaf everywhere.
    fn wire_ttl(&self, oauth: bool, latches: &latches::Latches) -> Option<&'static str> {
        if latches.ttl_refused {
            return None;
        }
        if let Some(ttl) = self.configured_cache_ttl {
            return Some(ttl);
        }
        if !self.native_route() {
            return None;
        }
        Some(if oauth { "1h" } else { "5m" })
    }

    /// Whether breakpoints may be sent at all. Requests without a prompt
    /// cache key have no stable conversation identity, so they opt out like
    /// Claude Code's aux queries (`enablePromptCaching: false`) and do not
    /// pay write premiums on throwaway prefixes.
    fn breakpoints_allowed(&self, request: &ModelRequest<'_>, latches: &latches::Latches) -> bool {
        request.prompt_cache_key.is_some() && !latches.cache_control_refused
    }

    fn request_latches(&self, cache_key: Option<&str>) -> latches::Latches {
        self.latches
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .latches(cache_key)
            .clone()
    }

    fn request_body(
        &self,
        request: ModelRequest<'_>,
        stream: bool,
        oauth: bool,
        options: &rho_sdk::provider::ModelRequestOptions,
    ) -> Result<AnthropicRequest, ModelError> {
        let latches = self.request_latches(request.prompt_cache_key);
        let target = self.model_identity();
        let max_tokens = options
            .max_output_tokens()
            .map_or_else(|| self.max_tokens(), |cap| self.max_tokens().min(cap.get()));
        let (thinking, output_config) = thinking::thinking_config_for(
            &self.thinking_source(),
            request.reasoning_level,
            self.thinking_budget_ceiling(max_tokens),
        )?;
        let (system, mut messages) = split_system_and_messages(
            request.messages,
            &target,
            provider_context_replay(thinking.as_ref()),
        )?;
        let output_config = {
            let mut state = self
                .per_message_effort
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            per_message_effort::apply(
                &self.model,
                &mut state,
                request.prompt_cache_key,
                output_config,
                &mut messages,
                !latches.mid_conv_system_refused,
            )
        };
        let cacheable = self.breakpoints_allowed(&request, &latches);
        let ttl = self.wire_ttl(oauth, &latches);
        if cacheable {
            mark_cache_control_points(&mut messages, ttl);
        }
        let mut tools = request
            .tools
            .iter()
            .cloned()
            .map(to_anthropic_tool)
            .collect::<Vec<_>>();
        if oauth {
            apply_oauth_wire_names(&mut tools, &mut messages);
            for tool in tools.iter_mut() {
                tool.description = sanitize_oauth_prose(&tool.description);
            }
        }
        if cacheable {
            if let Some(tool) = tools.last_mut() {
                tool.cache_control = Some(AnthropicCacheControl::ephemeral().with_ttl(ttl));
            }
        }
        let mut system_blocks = system.map(|text| {
            vec![AnthropicSystemBlock::text(
                text,
                cacheable.then(|| AnthropicCacheControl::ephemeral().with_ttl(ttl)),
            )]
        });
        if oauth {
            for block in system_blocks.iter_mut().flatten() {
                block.text = sanitize_oauth_prose(&block.text);
            }
            // Anthropic's OAuth classifier expects the Claude Code identity
            // line first; the app's own system text keeps the second (cached)
            // breakpoint. The identity line is byte-identical for every Rho
            // session on the account, so it carries `scope: "global"` — a new
            // or resumed conversation can read the shared prefix instead of
            // rewriting it, matching Claude Code's classifier block.
            let identity_control = if cacheable {
                Some(
                    AnthropicCacheControl::ephemeral()
                        .with_ttl(ttl)
                        .with_scope((!latches.scope_refused).then_some(GLOBAL_CACHE_SCOPE)),
                )
            } else {
                None
            };
            let scope = (!latches.scope_refused).then_some(GLOBAL_CACHE_SCOPE);
            let claude_code = AnthropicSystemBlock::text(
                CLAUDE_CODE_SYSTEM_PREFIX.to_string(),
                match system_blocks.as_mut().and_then(|blocks| blocks.first_mut()) {
                    Some(first) => first
                        .cache_control
                        .take()
                        .map(|control| control.with_scope(scope))
                        .or(identity_control),
                    None => identity_control,
                },
            );
            system_blocks
                .get_or_insert_with(Vec::new)
                .insert(0, claude_code);
        }
        // `-fast` catalog rows are a speed tier of the base model on the
        // native host, not a model id: strip the suffix and send `speed`. A
        // requested Priority service tier maps onto the same wire flag.
        let (model, mut speed) = match fast_mode_base(&self.model).filter(|_| self.native_route()) {
            Some(base) => (base.to_string(), Some("fast")),
            None => (self.model.clone(), None),
        };
        if speed.is_none()
            && self.native_route()
            && options.service_tier() == Some(rho_sdk::model::ServiceTier::Priority)
            && fast_mode_base(&format!("{}-fast", self.model)).is_some()
        {
            speed = Some("fast");
        }
        Ok(AnthropicRequest {
            model,
            max_tokens,
            system: system_blocks,
            messages,
            tools: (!tools.is_empty()).then_some(tools),
            cache_control: None,
            thinking,
            output_config,
            speed,
            stream,
        })
    }

    pub(crate) fn model_identity(&self) -> ModelIdentity {
        ModelIdentity::new(self.identity_provider, "anthropic-messages", &self.model)
    }

    /// Completes one turn using inherent async methods so the future is `Send`.
    pub(crate) async fn complete_turn(
        &self,
        request: ModelRequest<'_>,
    ) -> Result<ModelResponse, ModelError> {
        self.send_messages(request).await
    }

    /// Streams one turn through a `Send` callback for the public SDK adapter.
    pub(crate) async fn stream_turn(
        &self,
        request: ModelRequest<'_>,
        on_event: &mut (dyn FnMut(ModelEvent) -> Result<(), ModelError> + Send),
        on_request_event: &mut (dyn FnMut(rho_sdk::provider::ProviderRequestEvent) -> Result<(), ModelError>
                  + Send),
    ) -> Result<ModelResponse, ModelError> {
        self.stream_turn_with_options(
            request,
            rho_sdk::provider::ModelRequestOptions::default(),
            on_event,
            on_request_event,
        )
        .await
    }

    /// `max_tokens` is always sent, so every route honors an output cap —
    /// a keep-warm ping caps at one token. The TUI still verifies the cap
    /// from reported usage before repeating a capped request.
    pub(crate) fn applies_max_output_tokens(&self) -> bool {
        true
    }

    pub(crate) async fn stream_turn_with_options(
        &self,
        request: ModelRequest<'_>,
        options: rho_sdk::provider::ModelRequestOptions,
        on_event: &mut (dyn FnMut(ModelEvent) -> Result<(), ModelError> + Send),
        _on_request_event: &mut (dyn FnMut(rho_sdk::provider::ProviderRequestEvent) -> Result<(), ModelError>
                  + Send),
    ) -> Result<ModelResponse, ModelError> {
        self.send_messages_stream(request, &options, on_event).await
    }

    async fn send_messages(&self, request: ModelRequest<'_>) -> Result<ModelResponse, ModelError> {
        let response = self
            .send_with_refresh(
                request,
                false,
                &rho_sdk::provider::ModelRequestOptions::default(),
                &mut |_| Ok(()),
            )
            .await?;
        Ok(strip_oauth_tool_names(response, self.oauth_route()))
    }

    async fn send_messages_stream(
        &self,
        request: ModelRequest<'_>,
        options: &rho_sdk::provider::ModelRequestOptions,
        on_event: &mut (dyn FnMut(ModelEvent) -> Result<(), ModelError> + Send),
    ) -> Result<ModelResponse, ModelError> {
        let oauth = self.oauth_route();
        let response = self
            .send_with_refresh(request, true, options, &mut |event| {
                on_event(strip_oauth_event_tool_name(event, oauth))
            })
            .await?;
        Ok(strip_oauth_tool_names(response, oauth))
    }

    /// Sends one request, retrying once after a 401 with a force-refreshed
    /// OAuth token (an expired grant is the common cause) and replaying it
    /// degraded after each newly classified 400 — cache fields, mid-
    /// conversation system markers, or refused betas a hosted adapter
    /// rejects. Latches stick for the conversation, so a refused field is
    /// never sent twice.
    async fn send_with_refresh(
        &self,
        request: ModelRequest<'_>,
        stream: bool,
        options: &rho_sdk::provider::ModelRequestOptions,
        on_event: &mut (dyn FnMut(ModelEvent) -> Result<(), ModelError> + Send),
    ) -> Result<ModelResponse, ModelError> {
        let mut auth = self.request_auth().await?;
        let mut refreshed = false;
        // One auth retry plus up to two degradation replays bound the loop.
        for _ in 0..4 {
            match self
                .send_once(request.clone(), stream, &auth, options, &mut *on_event)
                .await
            {
                Err(error) if is_unauthorized(&error) && !refreshed => {
                    refreshed = true;
                    let AnthropicAuth::OAuth(manager) = &self.auth else {
                        return Err(error);
                    };
                    let Some(material) = manager.force_refresh(&auth.access_token).await? else {
                        return Err(error);
                    };
                    auth = RequestAuth {
                        access_token: material.access_token,
                        oauth: auth.oauth,
                    };
                }
                Err(error) => {
                    let Some(degrade) = self.classify_failure(&error, request.prompt_cache_key)
                    else {
                        return Err(error);
                    };
                    let newly = {
                        self.latches
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .latch(request.prompt_cache_key, degrade)
                    };
                    if !newly {
                        return Err(error);
                    }
                }
                result => return result,
            }
        }
        Err(ModelError::InvalidResponse(
            "provider rejected the request after every downgrade".into(),
        ))
    }

    /// Maps a failed attempt to the downgrade it asks for, or `None` when the
    /// error is not a classified field refusal.
    fn classify_failure(
        &self,
        error: &ModelError,
        cache_key: Option<&str>,
    ) -> Option<latches::Degrade> {
        let ModelError::HttpStatus { status, body, .. } = error else {
            return None;
        };
        if !matches!(
            *status,
            http::StatusCode::BAD_REQUEST | http::StatusCode::UNPROCESSABLE_ENTITY
        ) {
            return None;
        }
        let latches = self.request_latches(cache_key);
        let sent_betas = self.sent_betas(latches.mid_conv_system_refused, &latches.refused_betas);
        latches::classify_bad_request(body, &sent_betas)
    }

    /// The `anthropic-beta` values an attempt would send under `latches`.
    fn sent_betas(
        &self,
        mid_conv_system_refused: bool,
        refused: &std::collections::BTreeSet<&'static str>,
    ) -> Vec<&'static str> {
        let mut betas = Vec::new();
        if self.native_route() {
            betas.extend(COMMON_BETAS.split(','));
        }
        if fast_mode_base(&self.model).is_some() && self.native_route() {
            betas.push(FAST_MODE_BETA);
        }
        if !mid_conv_system_refused {
            betas.push(per_message_effort::BETA);
        }
        if matches!(self.auth, AnthropicAuth::OAuth(_)) && self.native_route() {
            betas.extend(OAUTH_BETAS.split(','));
        }
        betas.retain(|beta| !refused.contains(beta));
        betas
    }

    async fn send_once(
        &self,
        request: ModelRequest<'_>,
        stream: bool,
        auth: &RequestAuth,
        options: &rho_sdk::provider::ModelRequestOptions,
        on_event: &mut (dyn FnMut(ModelEvent) -> Result<(), ModelError> + Send),
    ) -> Result<ModelResponse, ModelError> {
        let cache_key = request.prompt_cache_key;
        let body = self.request_body(request, stream, auth.oauth, options)?;
        let latches = self.request_latches(cache_key);
        let headers = self.session_headers.headers(cache_key);
        let response = super::request_body::send(
            self.messages_request(&body, auth, &latches)
                .headers(headers)
                .json(&body),
            self.compression,
        )
        .await?;
        let response = crate::provider_backend::http_error::error_for_status(response).await?;
        if stream {
            collect_anthropic_sse_response(response, on_event).await
        } else {
            let response: AnthropicResponse = response.json().await?;
            convert_anthropic_response(response)
        }
    }

    /// Resolves this request's credential. OAuth applies only on the native
    /// first-party route; everywhere else the same credential degrades to a
    /// plain `x-api-key`/`Bearer` per token shape without the Claude Code
    /// identity (third-party endpoints 401/403 on it).
    async fn request_auth(&self) -> Result<RequestAuth, ModelError> {
        let native = self.native_route();
        match &self.auth {
            AnthropicAuth::ApiKey(key) => {
                // An OAuth-shaped stored key (setup token) still rides the
                // OAuth wire identity on the native host.
                let oauth = native && anthropic_oauth::is_oauth_token(key);
                if oauth {
                    anthropic_oauth::ensure_claude_code_version(&self.client).await?;
                }
                Ok(RequestAuth {
                    access_token: key.clone(),
                    oauth,
                })
            }
            AnthropicAuth::OAuth(manager) => {
                let AnthropicAuthMaterial { access_token, .. } = manager.auth_material().await?;
                if native {
                    anthropic_oauth::ensure_claude_code_version(&self.client).await?;
                }
                Ok(RequestAuth {
                    access_token,
                    oauth: native,
                })
            }
        }
    }

    /// True when this adapter serves the first-party Anthropic host — the
    /// only route the OAuth wire identity (betas, Claude Code headers, tool
    /// and system transforms) applies to.
    fn native_route(&self) -> bool {
        self.identity_provider == "anthropic" && is_native_anthropic_host(&self.api_base)
    }

    fn oauth_route(&self) -> bool {
        self.native_route() && matches!(self.auth, AnthropicAuth::OAuth(_))
    }

    fn messages_url(&self) -> String {
        format!("{}/messages", self.api_base.trim_end_matches('/'))
    }

    fn messages_request(
        &self,
        body: &AnthropicRequest,
        auth: &RequestAuth,
        latches: &latches::Latches,
    ) -> reqwest::RequestBuilder {
        let mut betas = Vec::new();
        if self.native_route() {
            betas.push(COMMON_BETAS);
        }
        if body.speed.is_some() && !latches.refused_betas.contains(FAST_MODE_BETA) {
            betas.push(FAST_MODE_BETA);
        }
        if let Some(beta) = per_message_effort::beta_header(&body.messages) {
            if !latches.refused_betas.contains(beta) {
                betas.push(beta);
            }
        }
        if auth.oauth {
            betas.push(OAUTH_BETAS);
        }
        let request = self.client.post(self.messages_url());
        let request = if auth.oauth {
            request
                .bearer_auth(&auth.access_token)
                .header(
                    "user-agent",
                    format!(
                        "claude-code/{} (external, cli)",
                        anthropic_oauth::claude_code_version()
                    ),
                )
                .header("x-app", "cli")
        } else {
            request.header("x-api-key", &auth.access_token)
        };
        let request = request.header("anthropic-version", ANTHROPIC_VERSION);
        if betas.is_empty() {
            request
        } else {
            request.header("anthropic-beta", betas.join(","))
        }
    }
}

fn is_unauthorized(error: &ModelError) -> bool {
    matches!(
        error,
        ModelError::HttpStatus { status, .. } if *status == http::StatusCode::UNAUTHORIZED
    )
}

/// `RHO_ANTHROPIC_CACHE_TTL` override on the native route: `"5m"` or `"1h"`.
/// Anything else is ignored so a typo cannot kill request construction.
fn env_cache_ttl() -> Option<std::time::Duration> {
    match std::env::var(CACHE_TTL_ENV).ok()?.as_str() {
        "5m" => Some(std::time::Duration::from_secs(5 * 60)),
        "1h" => Some(std::time::Duration::from_secs(60 * 60)),
        _ => None,
    }
}

/// OAuth wire name for one tool: `mcp__<name>`, with the classifier-aliased
/// names applied first. Collision-safe: an alias that would collide with a
/// real tool name is skipped.
fn apply_oauth_wire_names(
    tools: &mut [crate::protocol::anthropic_messages::AnthropicTool],
    messages: &mut [AnthropicMessage],
) {
    let claimed: std::collections::HashSet<String> = tools
        .iter()
        .map(|tool| oauth_wire_name(&tool.name))
        .collect();
    let wire = |name: &str| {
        if let Some(alias) = OAUTH_TOOL_NAME_ALIASES
            .iter()
            .find(|(from, _)| *from == name)
            .map(|(_, to)| *to)
        {
            let aliased = format!("{MCP_TOOL_PREFIX}{alias}");
            if !claimed.contains(&aliased) {
                return aliased;
            }
        }
        oauth_wire_name(name)
    };
    for tool in tools.iter_mut() {
        tool.name = wire(&tool.name);
    }
    for message in messages.iter_mut() {
        if message.role != AnthropicRole::Assistant {
            continue;
        }
        for block in message.content.iter_mut() {
            if let AnthropicContentBlock::ToolUse { name, .. } = block {
                *name = wire(name);
            }
        }
    }
}

fn oauth_wire_name(name: &str) -> String {
    if name.starts_with(MCP_TOOL_PREFIX) {
        return name.to_string();
    }
    format!(
        "{MCP_TOOL_PREFIX}{}",
        name.strip_prefix("mcp_").unwrap_or(name)
    )
}

/// Reverses the OAuth wire naming on a streamed event.
fn strip_oauth_event_tool_name(event: ModelEvent, oauth: bool) -> ModelEvent {
    if !oauth {
        return event;
    }
    match event {
        ModelEvent::ToolCallDelta {
            index,
            id,
            name,
            arguments,
        } => ModelEvent::ToolCallDelta {
            index,
            id,
            name: name.map(|name| oauth_response_tool_name(&name).to_string()),
            arguments,
        },
        event => event,
    }
}

/// Reverses the OAuth wire naming on a completed response.
fn strip_oauth_tool_names(mut response: ModelResponse, oauth: bool) -> ModelResponse {
    if !oauth {
        return response;
    }
    let ModelResponse::Assistant(blocks) = &mut response;
    for block in blocks.iter_mut() {
        if let rho_sdk::model::ContentBlock::ToolCall(call) = block {
            call.name = oauth_response_tool_name(&call.name).to_string();
        }
    }
    response
}

/// Prose rewrites applied to system text and tool descriptions on the OAuth
/// wire: product-name sanitizing plus the prose-safe tool aliases, matching
/// Hermes's `_apply_claude_code_identity` / `_apply_oauth_prose_aliases`.
fn sanitize_oauth_prose(text: &str) -> String {
    let mut text = text.to_string();
    for (from, to) in OAUTH_PRODUCT_REPLACEMENTS {
        text = replace_word_token(&text, from, to);
    }
    text = replace_slug_token(&text, "rho", "claude-code");
    for (from, to) in OAUTH_PROSE_ALIASES {
        text = replace_word_token(&text, from, to);
    }
    text
}

fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// Replaces occurrences of `token` that are not glued to a word byte on
/// either side (so `session_search_tool` and `the_session_search` survive).
fn replace_word_token(text: &str, token: &str, replacement: &str) -> String {
    replace_token(text, token, replacement, |before, after| {
        !is_word_byte(before) && !is_word_byte(after)
    })
}

/// Standalone-slug rewrite for the `rho` token: like the word-boundary rule
/// but also refusing to rewrite inside paths, hosts, mailboxes and quoted
/// identifiers (`GitHub/rho`, `rho.dev`, `user@rho`), mirroring Hermes's
/// `_OAUTH_SLUG_PATTERN`.
fn replace_slug_token(text: &str, token: &str, replacement: &str) -> String {
    const GLUE: &[u8] = b"./:@'\"`-";
    replace_token(text, token, replacement, |before, after| {
        (!is_word_byte(before) && !GLUE.contains(&before))
            && (!is_word_byte(after) && !GLUE.contains(&after))
    })
}

fn replace_token(
    text: &str,
    token: &str,
    replacement: &str,
    boundary: impl Fn(u8, u8) -> bool,
) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(index) = rest.find(token) {
        let before = if index == 0 {
            0
        } else {
            rest.as_bytes()[index - 1]
        };
        let after = *rest.as_bytes().get(index + token.len()).unwrap_or(&0);
        if boundary(before, after) {
            out.push_str(&rest[..index]);
            out.push_str(replacement);
        } else {
            out.push_str(&rest[..index + token.len()]);
        }
        rest = &rest[index + token.len()..];
    }
    out.push_str(rest);
    out
}

/// Base id for a `-fast` catalog row on a fast-mode family (Opus 4.8 / 5 /
/// 5.5). `None` keeps the id as-is; `Some` strips the suffix and the caller
/// adds `speed: "fast"` plus the fast-mode beta — on the native host only,
/// since third-party endpoints reject both.
fn fast_mode_base(model: &str) -> Option<&str> {
    let base = model.strip_suffix("-fast")?;
    const FAST_FAMILIES: [&str; 2] = ["claude-opus-4-8", "claude-opus-5"];
    FAST_FAMILIES
        .iter()
        .any(|prefix| {
            base == *prefix
                || base
                    .strip_prefix(prefix)
                    .is_some_and(|suffix| suffix.starts_with('-'))
        })
        .then_some(base)
}

/// `mcp__chat_history_lookup` -> `session_search`; `mcp__<name>` -> `<name>`.
fn oauth_response_tool_name(name: &str) -> &str {
    let Some(wire) = name.strip_prefix(MCP_TOOL_PREFIX) else {
        return name;
    };
    if let Some((from, _)) = OAUTH_TOOL_NAME_ALIASES.iter().find(|(_, to)| *to == wire) {
        return from;
    }
    wire
}

fn provider_context_replay(thinking: Option<&AnthropicThinkingConfig>) -> ProviderContextReplay {
    match thinking {
        Some(
            AnthropicThinkingConfig::Enabled { .. }
            | AnthropicThinkingConfig::Adaptive { .. }
            | AnthropicThinkingConfig::BetweenTools,
        ) => ProviderContextReplay::Enabled,
        Some(AnthropicThinkingConfig::Disabled) | None => ProviderContextReplay::Disabled,
    }
}

fn mark_cache_control_points(messages: &mut [AnthropicMessage], ttl: Option<&'static str>) {
    // Writes occur only at marked breakpoints. Anthropic looks 20 content
    // blocks upstream of each one, so a fat tool-result user turn can miss the
    // previous write if only the tail is marked. Mark the last shared block of
    // the previous user message (the prior write) and the new tail. Tools and
    // system still use the other two of the four breakpoints.
    // A trailing user text block is the per-request suffix and stays unmarked.
    let mut points = Vec::new();
    for (message_index, message) in messages.iter().enumerate() {
        if message.role != AnthropicRole::User {
            continue;
        }
        for (block_index, block) in message.content.iter().enumerate() {
            if is_cacheable_user_block(block) {
                points.push((message_index, block_index));
            }
        }
    }
    if let Some(&(message_index, block_index)) = points.last() {
        let last_message_points = points
            .iter()
            .filter(|(index, _)| *index == message_index)
            .count();
        if last_message_points > 1
            && is_user_text_block(&messages[message_index].content[block_index])
        {
            points.pop();
        }
    }
    if let Some(&(tail_message, tail_block)) = points.last() {
        set_user_breakpoint(&mut messages[tail_message].content[tail_block], ttl);
        if let Some(&(prefix_message, prefix_block)) = points
            .iter()
            .rev()
            .find(|(message_index, _)| *message_index < tail_message)
        {
            set_user_breakpoint(&mut messages[prefix_message].content[prefix_block], ttl);
        }
        return;
    }

    for message in messages.iter_mut().rev() {
        if message.role != AnthropicRole::Assistant {
            continue;
        }
        if let Some(AnthropicContentBlock::Text { cache_control, .. }) = message
            .content
            .iter_mut()
            .rev()
            .find(|block| matches!(block, AnthropicContentBlock::Text { .. }))
        {
            *cache_control = Some(AnthropicCacheControl::ephemeral().with_ttl(ttl));
            return;
        }
    }
}

fn set_user_breakpoint(block: &mut AnthropicContentBlock, ttl: Option<&'static str>) {
    match block {
        AnthropicContentBlock::Text { cache_control, .. }
        | AnthropicContentBlock::ToolResult { cache_control, .. } => {
            *cache_control = Some(AnthropicCacheControl::ephemeral().with_ttl(ttl));
        }
        AnthropicContentBlock::Thinking { .. }
        | AnthropicContentBlock::RedactedThinking { .. }
        | AnthropicContentBlock::Image { .. }
        | AnthropicContentBlock::ToolUse { .. } => {}
    }
}

fn is_cacheable_user_block(block: &AnthropicContentBlock) -> bool {
    matches!(
        block,
        AnthropicContentBlock::Text { .. } | AnthropicContentBlock::ToolResult { .. }
    )
}

fn is_user_text_block(block: &AnthropicContentBlock) -> bool {
    matches!(block, AnthropicContentBlock::Text { .. })
}

crate::impl_sdk_model_provider!(
    AnthropicProvider,
    request_options(max_output_tokens = Self::applies_max_output_tokens)
);

#[cfg(test)]
mod tests {
    use rho_sdk::{
        model::{Message, ToolSpec},
        provider::ModelRequestOptions,
        CancellationToken, ReasoningLevel,
    };
    use serde_json::{json, Value};

    use super::*;

    fn test_request<'a>(
        messages: &'a [Message],
        tools: &'a [ToolSpec],
        prompt_cache_key: Option<&'a str>,
    ) -> ModelRequest<'a> {
        ModelRequest {
            messages,
            tools,
            cancellation: CancellationToken::new(),
            reasoning_level: ReasoningLevel::Off,
            prompt_cache_key,
        }
    }

    fn tool() -> ToolSpec {
        ToolSpec {
            name: "read".into(),
            description: "read a file".into(),
            input_schema: json!({"type": "object"}),
        }
    }

    fn body(provider: &AnthropicProvider, request: ModelRequest<'_>, oauth: bool) -> Value {
        serde_json::to_value(
            provider
                .request_body(request, false, oauth, &ModelRequestOptions::default())
                .expect("request body"),
        )
        .expect("serializes")
    }

    /// Every `cache_control` leaf in the serialized body, for assertions
    /// that count breakpoints or inspect `ttl`/`scope`.
    fn cache_controls(body: &Value) -> Vec<&Value> {
        let mut found = Vec::new();
        let mut stack = vec![body];
        while let Some(value) = stack.pop() {
            match value {
                Value::Object(map) => {
                    if let Some(control) = map.get("cache_control") {
                        found.push(control);
                    }
                    stack.extend(map.values());
                }
                Value::Array(items) => stack.extend(items),
                _ => {}
            }
        }
        found
    }

    #[test]
    fn api_key_route_sends_5m_ttl() {
        let provider = AnthropicProvider::new("claude-sonnet-4-5".into(), "key".into());
        let messages = vec![Message::user_text("hello")];
        let tools = vec![tool()];
        let body = body(
            &provider,
            test_request(&messages, &tools, Some("conv")),
            false,
        );
        let controls = cache_controls(&body);
        assert!(!controls.is_empty());
        assert!(
            controls.iter().all(|control| control["ttl"] == "5m"),
            "every breakpoint carries the 5m ttl: {body}"
        );
    }

    #[test]
    fn oauth_route_sends_1h_ttl_and_global_scope_identity() {
        let provider = AnthropicProvider::new("claude-sonnet-4-5".into(), "key".into());
        let messages = vec![
            Message::System("you are helpful".into()),
            Message::user_text("hello"),
        ];
        let tools = vec![tool()];
        let body = body(
            &provider,
            test_request(&messages, &tools, Some("conv")),
            true,
        );
        let system = body["system"].as_array().expect("system blocks");
        let identity = &system[0];
        assert_eq!(identity["text"], CLAUDE_CODE_SYSTEM_PREFIX);
        assert_eq!(identity["cache_control"]["scope"], "global");
        assert_eq!(identity["cache_control"]["ttl"], "1h");
        // The single system breakpoint moves onto the identity block — the
        // byte-identical line Anthropic can share account-wide — so the
        // app's own system text is covered by the tool/message breakpoints
        // and the four-breakpoint cap is not exceeded.
        assert!(system[1].get("cache_control").is_none(), "{system:?}");
    }

    #[test]
    fn keyless_request_sends_no_cache_control() {
        let provider = AnthropicProvider::new("claude-sonnet-4-5".into(), "key".into());
        let messages = vec![Message::user_text("hello")];
        let tools = vec![tool()];
        let body = body(&provider, test_request(&messages, &tools, None), false);
        assert!(cache_controls(&body).is_empty(), "{body}");
    }

    #[test]
    fn latched_cache_control_strips_breakpoints() {
        let provider = AnthropicProvider::new("claude-sonnet-4-5".into(), "key".into());
        provider
            .latches
            .lock()
            .unwrap()
            .latch(Some("conv"), latches::Degrade::StripCacheControl);
        let messages = vec![Message::user_text("hello")];
        let tools = vec![tool()];
        let body = body(
            &provider,
            test_request(&messages, &tools, Some("conv")),
            false,
        );
        assert!(cache_controls(&body).is_empty(), "{body}");
    }

    #[test]
    fn latched_ttl_drops_leaf_only() {
        let provider = AnthropicProvider::new("claude-sonnet-4-5".into(), "key".into());
        provider
            .latches
            .lock()
            .unwrap()
            .latch(Some("conv"), latches::Degrade::DropTtl);
        let messages = vec![Message::user_text("hello")];
        let tools = vec![tool()];
        let body = body(
            &provider,
            test_request(&messages, &tools, Some("conv")),
            false,
        );
        let controls = cache_controls(&body);
        assert!(!controls.is_empty());
        assert!(controls.iter().all(|control| control.get("ttl").is_none()));
    }

    #[test]
    fn latched_scope_drops_leaf_only() {
        let provider = AnthropicProvider::new("claude-sonnet-4-5".into(), "key".into());
        provider
            .latches
            .lock()
            .unwrap()
            .latch(Some("conv"), latches::Degrade::DropScope);
        let messages = vec![Message::user_text("hello")];
        let tools = vec![tool()];
        let body = body(
            &provider,
            test_request(&messages, &tools, Some("conv")),
            true,
        );
        let system = body["system"].as_array().expect("system blocks");
        let control = &system[0]["cache_control"];
        assert_eq!(control["ttl"], "1h");
        assert!(control.get("scope").is_none(), "{control}");
    }

    #[test]
    fn latches_are_scoped_per_conversation() {
        let provider = AnthropicProvider::new("claude-sonnet-4-5".into(), "key".into());
        provider
            .latches
            .lock()
            .unwrap()
            .latch(Some("other"), latches::Degrade::StripCacheControl);
        let messages = vec![Message::user_text("hello")];
        let tools = vec![tool()];
        let body = body(
            &provider,
            test_request(&messages, &tools, Some("conv")),
            false,
        );
        assert!(!cache_controls(&body).is_empty());
    }

    #[test]
    fn configured_ttl_override_wins() {
        let provider = AnthropicProvider::new("claude-sonnet-4-5".into(), "key".into())
            .with_cache_ttl(Some(std::time::Duration::from_secs(301)));
        let messages = vec![Message::user_text("hello")];
        let tools = vec![tool()];
        let body = body(
            &provider,
            test_request(&messages, &tools, Some("conv")),
            false,
        );
        assert!(cache_controls(&body)
            .iter()
            .all(|control| control["ttl"] == "1h"));
    }

    #[test]
    fn max_output_tokens_caps_max_tokens() {
        let provider = AnthropicProvider::new("claude-sonnet-4-5".into(), "key".into());
        let messages = vec![Message::user_text("hello")];
        let tools = vec![tool()];
        let options =
            ModelRequestOptions::default().with_max_output_tokens(std::num::NonZeroU32::MIN);
        let request = provider
            .request_body(
                test_request(&messages, &tools, Some("conv")),
                false,
                false,
                &options,
            )
            .expect("request body");
        assert_eq!(request.max_tokens, 1);
    }
}
