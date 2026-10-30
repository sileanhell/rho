use std::{future::Future, marker::PhantomData, pin::Pin, time::SystemTime};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use {
    rho_providers::auth::{
        anthropic_oauth::{claude_code_version, refresh_anthropic_tokens},
        kimi_oauth::refresh_kimi_tokens,
        xai_token::refresh_xai_tokens,
    },
    rho_providers::credentials::{
        load_anthropic_tokens, load_codex_tokens, load_kimi_tokens, load_provider_api_key,
        load_xai_tokens, save_anthropic_tokens, save_kimi_tokens, save_xai_tokens,
        AnthropicTokens, CodexTokens, CredentialStore, KimiTokens, XaiTokens,
    },
    rho_providers::providers::openai::auth::{refresh_codex_token, CodexAuthSource},
};

const ANTHROPIC_USAGE_URL: &str = "https://api.anthropic.com/api/oauth/usage";
const ANTHROPIC_OAUTH_BETA: &str = "oauth-2025-04-20";
const CODEX_USAGE_URL: &str = "https://chatgpt.com/backend-api/wham/usage";
const CODEX_ACCOUNT_HEADER: &str = "ChatGPT-Account-Id";
const KIMI_USAGE_URL: &str = "https://api.kimi.com/coding/v1/usages";
const XAI_BILLING_URL: &str = "https://cli-chat-proxy.grok.com/v1/billing?format=credits";
const XAI_TOKEN_AUTH_HEADER: &str = "xai-grok-cli";
const XAI_CLIENT_VERSION: &str = "0.2.93";
const OPENCODE_GO_USAGE_URL: &str = "https://opencode.ai/zen/go/v1/usage";

#[derive(Clone, Debug, PartialEq)]
pub struct ProviderUsageLimits {
    pub provider: String,
    pub windows: Vec<UsageLimitWindow>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UsageLimitWindow {
    pub label: String,
    /// Remaining percent when the source reported utilization.
    pub remaining_percent: Option<f64>,
    /// Reset instant when the source reported one.
    pub resets_at_unix: Option<i64>,
    /// Optional status note (warning, overage, observation age, …).
    pub note: Option<String>,
}

/// One connected source `/limits` can query live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UsageProviderKind {
    Claude,
    Codex,
    KimiCode,
    OpenCodeGo,
    Xai,
}

/// Boxed future returned by [`UsageLimitsSource::fetch`].
type UsageLimitsFuture<'a> = Pin<
    Box<dyn Future<Output = Result<Option<ProviderUsageLimits>, UsageLimitsError>> + Send + 'a>,
>;

/// Why a live usage read did not return percentages. Surfaced in `/limits`
/// headings so a throttled read is not mistaken for a broken one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UsageFailure {
    /// The usage endpoint throttled the read; retrying later works.
    RateLimited,
    /// Any other failure (network, auth, partial data).
    Other,
}

#[derive(Debug, Error)]
pub enum UsageLimitsError {
    #[error("could not load credentials: {0}")]
    Credentials(#[from] rho_providers::credentials::CredentialError),
    #[error("{provider} usage request failed: {source}")]
    Request {
        provider: &'static str,
        #[source]
        source: reqwest::Error,
    },
    #[error("could not refresh {provider} OAuth credentials: {detail}")]
    Refresh {
        provider: &'static str,
        detail: String,
    },
    #[error("{provider} credentials are no longer valid; run {login}")]
    Unauthorized {
        provider: &'static str,
        login: &'static str,
    },
}

impl UsageLimitsError {
    /// Only an HTTP 429 from the usage endpoint is a rate limit.
    pub fn failure(&self) -> UsageFailure {
        match self {
            Self::Request { source, .. }
                if source.status() == Some(reqwest::StatusCode::TOO_MANY_REQUESTS) =>
            {
                UsageFailure::RateLimited
            }
            Self::Request { .. }
            | Self::Credentials(_)
            | Self::Refresh { .. }
            | Self::Unauthorized { .. } => UsageFailure::Other,
        }
    }

    /// Labels a transport failure with the provider that produced it.
    fn request(provider: &'static str) -> impl Fn(reqwest::Error) -> Self {
        move |source| Self::Request { provider, source }
    }
}

/// Rejects error statuses and decodes a provider usage payload.
async fn decode_usage_payload<T: serde::de::DeserializeOwned>(
    response: reqwest::Response,
    provider: &'static str,
) -> Result<T, UsageLimitsError> {
    response
        .error_for_status()
        .map_err(UsageLimitsError::request(provider))?
        .json::<T>()
        .await
        .map_err(UsageLimitsError::request(provider))
}

/// Finishes the request/refresh/retry ladder shared by every usage source.
///
/// `first` is the initial response. When it fails auth and `refresh_and_retry`
/// returns a new response, that response is used instead. A still-failing
/// status becomes [`UsageLimitsError::Unauthorized`].
async fn finish_auth_retry<RefreshFut>(
    provider: &'static str,
    login: &'static str,
    is_auth_failure: impl Fn(reqwest::StatusCode) -> bool,
    first: Result<reqwest::Response, reqwest::Error>,
    refresh_and_retry: impl FnOnce() -> RefreshFut,
) -> Result<reqwest::Response, UsageLimitsError>
where
    RefreshFut: Future<Output = Result<Option<reqwest::Response>, UsageLimitsError>>,
{
    let mut response = first.map_err(UsageLimitsError::request(provider))?;
    if is_auth_failure(response.status()) {
        if let Some(retry) = refresh_and_retry().await? {
            response = retry;
        }
    }
    if is_auth_failure(response.status()) {
        return Err(UsageLimitsError::Unauthorized { provider, login });
    }
    Ok(response)
}

fn unauthorized_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::UNAUTHORIZED
}

fn unauthorized_or_forbidden(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN
}

/// HTTP client plus the endpoint a usage source queries.
struct UsageEndpoint {
    client: reqwest::Client,
    url: String,
}

impl UsageEndpoint {
    fn new(client: reqwest::Client, url: &str) -> Self {
        Self {
            client,
            url: url.into(),
        }
    }

    #[cfg(test)]
    fn with_url(url: String) -> Self {
        Self {
            client: crate::reqwest_client(),
            url,
        }
    }

    fn client(&self) -> &reqwest::Client {
        &self.client
    }

    fn get(&self) -> reqwest::RequestBuilder {
        self.client.get(&self.url)
    }
}

/// Normalized usage windows for one connected provider. Missing windows must not be synthesized.
pub trait UsageLimitsSource {
    fn fetch<'a>(&'a self, store: &'a dyn CredentialStore) -> UsageLimitsFuture<'a>;
}

/// Credentials a provider resolved, paired with where they came from.
/// `Ok(None)` means the provider is not connected.
type ConfiguredTokens<T, S> = Result<Option<(T, S)>, UsageLimitsError>;

/// Where an env-or-store provider found its active credentials. Only stored
/// credentials may be refreshed and persisted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TokenAuthSource {
    Env,
    Store,
}

/// Boxed future returned by [`UsageProvider::refresh`].
type RefreshFuture<'a, T> =
    Pin<Box<dyn Future<Output = Result<Option<T>, UsageLimitsError>> + Send + 'a>>;

/// Token-authenticated usage endpoint. [`TokenUsageSource`] owns the shared ladder.
trait UsageProvider: Send + Sync + 'static {
    type Tokens: Send + Sync;
    /// Whether a refresh may replace stored tokens.
    type Source: Copy + Send + Sync;
    type Payload: serde::de::DeserializeOwned + Send;

    /// Name shown to the user and carried on [`UsageLimitsError`].
    const PROVIDER: &'static str;
    /// Command that reconnects this provider once its credentials expire.
    const LOGIN: &'static str;
    const URL: &'static str;

    /// Reports whether a status means the endpoint rejected the credentials.
    fn is_auth_failure(status: reqwest::StatusCode) -> bool {
        unauthorized_status(status)
    }

    /// Resolves credentials from the environment or the credential store.
    /// `Ok(None)` means the provider is not connected, not that it failed.
    fn configured_tokens(
        store: &dyn CredentialStore,
    ) -> ConfiguredTokens<Self::Tokens, Self::Source>;

    /// Applies provider authentication and headers to a prepared request.
    fn authorize(
        request: reqwest::RequestBuilder,
        tokens: &Self::Tokens,
    ) -> reqwest::RequestBuilder;

    /// Refresh rejected credentials; `Ok(None)` becomes unauthorized.
    fn refresh<'a>(
        client: &'a reqwest::Client,
        store: &'a dyn CredentialStore,
        tokens: &'a Self::Tokens,
        source: Self::Source,
    ) -> RefreshFuture<'a, Self::Tokens>;

    /// Decoded payload → reported windows.
    fn windows(payload: Self::Payload) -> Vec<UsageLimitWindow>;
}

/// Queries one [`UsageProvider`] endpoint, refreshing credentials once when
/// the first attempt is rejected.
struct TokenUsageSource<P: UsageProvider> {
    endpoint: UsageEndpoint,
    provider: PhantomData<fn() -> P>,
}

impl<P: UsageProvider> TokenUsageSource<P> {
    fn new(client: reqwest::Client) -> Self {
        Self {
            endpoint: UsageEndpoint::new(client, P::URL),
            provider: PhantomData,
        }
    }

    #[cfg(test)]
    fn with_endpoint(endpoint: String) -> Self {
        Self {
            endpoint: UsageEndpoint::with_url(endpoint),
            provider: PhantomData,
        }
    }

    async fn request(&self, tokens: &P::Tokens) -> Result<reqwest::Response, reqwest::Error> {
        P::authorize(self.endpoint.get(), tokens).send().await
    }

    async fn fetch_with_tokens(
        &self,
        store: &dyn CredentialStore,
        mut tokens: P::Tokens,
        source: P::Source,
    ) -> Result<ProviderUsageLimits, UsageLimitsError> {
        let response = finish_auth_retry(
            P::PROVIDER,
            P::LOGIN,
            P::is_auth_failure,
            self.request(&tokens).await,
            || async {
                let Some(refreshed) =
                    P::refresh(self.endpoint.client(), store, &tokens, source).await?
                else {
                    return Ok(None);
                };
                tokens = refreshed;
                self.request(&tokens)
                    .await
                    .map(Some)
                    .map_err(UsageLimitsError::request(P::PROVIDER))
            },
        )
        .await?;
        let payload: P::Payload = decode_usage_payload(response, P::PROVIDER).await?;
        Ok(ProviderUsageLimits {
            provider: P::PROVIDER.into(),
            windows: P::windows(payload),
        })
    }
}

impl<P: UsageProvider> UsageLimitsSource for TokenUsageSource<P> {
    fn fetch<'a>(&'a self, store: &'a dyn CredentialStore) -> UsageLimitsFuture<'a> {
        Box::pin(async move {
            let Some((tokens, source)) = P::configured_tokens(store)? else {
                return Ok(None);
            };
            self.fetch_with_tokens(store, tokens, source)
                .await
                .map(Some)
        })
    }
}

type ClaudeUsageLimitsSource = TokenUsageSource<ClaudeUsage>;
type CodexUsageLimitsSource = TokenUsageSource<CodexUsage>;
type KimiUsageLimitsSource = TokenUsageSource<KimiUsage>;
type XaiUsageLimitsSource = TokenUsageSource<XaiUsage>;
type OpenCodeGoUsageLimitsSource = TokenUsageSource<OpenCodeGoUsage>;

struct ClaudeUsage;

impl UsageProvider for ClaudeUsage {
    type Tokens = AnthropicTokens;
    type Source = TokenAuthSource;
    type Payload = ClaudeUsagePayload;

    const PROVIDER: &'static str = "Claude";
    const LOGIN: &'static str = "/login anthropic-oauth";
    const URL: &'static str = ANTHROPIC_USAGE_URL;

    fn configured_tokens(
        store: &dyn CredentialStore,
    ) -> ConfiguredTokens<AnthropicTokens, TokenAuthSource> {
        let env_token = ["ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"]
            .into_iter()
            .find_map(|name| {
                std::env::var(name)
                    .ok()
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty())
            });
        if let Some(access_token) = env_token {
            return Ok(Some((
                AnthropicTokens {
                    access_token,
                    refresh_token: None,
                    expires_at_ms: None,
                },
                TokenAuthSource::Env,
            )));
        }
        Ok(load_anthropic_tokens(store)?.map(|tokens| (tokens, TokenAuthSource::Store)))
    }

    fn authorize(
        request: reqwest::RequestBuilder,
        tokens: &AnthropicTokens,
    ) -> reqwest::RequestBuilder {
        request
            .bearer_auth(&tokens.access_token)
            .header("anthropic-beta", ANTHROPIC_OAUTH_BETA)
            .header(
                reqwest::header::USER_AGENT,
                format!("claude-code/{} (external, cli)", claude_code_version()),
            )
            .header(reqwest::header::ACCEPT, "application/json")
            .timeout(std::time::Duration::from_secs(5))
    }

    fn refresh<'a>(
        client: &'a reqwest::Client,
        store: &'a dyn CredentialStore,
        tokens: &'a AnthropicTokens,
        source: TokenAuthSource,
    ) -> RefreshFuture<'a, AnthropicTokens> {
        Box::pin(async move {
            let (TokenAuthSource::Store, Some(refresh_token)) =
                (source, tokens.refresh_token.clone())
            else {
                return Ok(None);
            };
            let refreshed = refresh_anthropic_tokens(client, &refresh_token)
                .await
                .map_err(|error| UsageLimitsError::Refresh {
                    provider: Self::PROVIDER,
                    detail: error.to_string(),
                })?;
            save_anthropic_tokens(store, &refreshed)?;
            Ok(Some(refreshed))
        })
    }

    fn windows(payload: ClaudeUsagePayload) -> Vec<UsageLimitWindow> {
        payload.windows()
    }
}

struct CodexUsage;

impl UsageProvider for CodexUsage {
    type Tokens = CodexTokens;
    type Source = CodexAuthSource;
    type Payload = CodexUsagePayload;

    const PROVIDER: &'static str = "Codex";
    const LOGIN: &'static str = "/login openai-codex";
    const URL: &'static str = CODEX_USAGE_URL;

    fn configured_tokens(
        store: &dyn CredentialStore,
    ) -> ConfiguredTokens<CodexTokens, CodexAuthSource> {
        if let Ok(access_token) = std::env::var("CODEX_ACCESS_TOKEN") {
            return Ok(Some((
                CodexTokens {
                    access_token,
                    refresh_token: None,
                    id_token: None,
                    account_id: std::env::var("CODEX_ACCOUNT_ID").ok(),
                },
                CodexAuthSource::Env,
            )));
        }
        Ok(load_codex_tokens(store)?.map(|tokens| (tokens, CodexAuthSource::Store)))
    }

    fn authorize(
        request: reqwest::RequestBuilder,
        tokens: &CodexTokens,
    ) -> reqwest::RequestBuilder {
        let request = request
            .bearer_auth(&tokens.access_token)
            .header(reqwest::header::CACHE_CONTROL, "no-store");
        match &tokens.account_id {
            Some(account_id) => request.header(CODEX_ACCOUNT_HEADER, account_id),
            None => request,
        }
    }

    fn refresh<'a>(
        client: &'a reqwest::Client,
        store: &'a dyn CredentialStore,
        tokens: &'a CodexTokens,
        source: CodexAuthSource,
    ) -> RefreshFuture<'a, CodexTokens> {
        Box::pin(async move {
            let Some(refresh_token) = tokens.refresh_token.clone() else {
                return Ok(None);
            };
            refresh_codex_token(client, store, &refresh_token, source, tokens)
                .await
                .map(Some)
                .map_err(|error| UsageLimitsError::Refresh {
                    provider: Self::PROVIDER,
                    detail: error.to_string(),
                })
        })
    }

    fn windows(payload: CodexUsagePayload) -> Vec<UsageLimitWindow> {
        payload.windows()
    }
}

struct KimiUsage;

impl KimiUsage {
    fn configured_tokens_from(
        store: &dyn CredentialStore,
        env_access_token: Option<String>,
    ) -> ConfiguredTokens<KimiTokens, TokenAuthSource> {
        if let Some(access_token) = env_access_token.filter(|token| !token.trim().is_empty()) {
            return Ok(Some((
                KimiTokens {
                    access_token,
                    refresh_token: None,
                    expires_at_unix: None,
                    scope: String::new(),
                    token_type: "Bearer".into(),
                    expires_in: None,
                },
                TokenAuthSource::Env,
            )));
        }
        Ok(load_kimi_tokens(store)?.map(|tokens| (tokens, TokenAuthSource::Store)))
    }
}

impl UsageProvider for KimiUsage {
    type Tokens = KimiTokens;
    type Source = TokenAuthSource;
    type Payload = KimiUsagePayload;

    const PROVIDER: &'static str = "Kimi Code";
    const LOGIN: &'static str = "/login kimi-code";
    const URL: &'static str = KIMI_USAGE_URL;

    fn configured_tokens(
        store: &dyn CredentialStore,
    ) -> ConfiguredTokens<KimiTokens, TokenAuthSource> {
        Self::configured_tokens_from(store, std::env::var("KIMI_ACCESS_TOKEN").ok())
    }

    fn authorize(request: reqwest::RequestBuilder, tokens: &KimiTokens) -> reqwest::RequestBuilder {
        request
            .bearer_auth(&tokens.access_token)
            .header(reqwest::header::ACCEPT, "application/json")
    }

    fn refresh<'a>(
        client: &'a reqwest::Client,
        store: &'a dyn CredentialStore,
        tokens: &'a KimiTokens,
        source: TokenAuthSource,
    ) -> RefreshFuture<'a, KimiTokens> {
        Box::pin(async move {
            let (TokenAuthSource::Store, Some(refresh_token)) =
                (source, tokens.refresh_token.clone())
            else {
                return Ok(None);
            };
            let refreshed = refresh_kimi_tokens(client, &refresh_token)
                .await
                .map_err(|error| UsageLimitsError::Refresh {
                    provider: Self::PROVIDER,
                    detail: error.to_string(),
                })?;
            save_kimi_tokens(store, &refreshed)?;
            Ok(Some(refreshed))
        })
    }

    fn windows(payload: KimiUsagePayload) -> Vec<UsageLimitWindow> {
        payload.windows()
    }
}

struct XaiUsage;

impl XaiUsage {
    fn configured_tokens_from(
        store: &dyn CredentialStore,
        env_access_token: Option<String>,
    ) -> ConfiguredTokens<XaiTokens, TokenAuthSource> {
        if let Some(access_token) = env_access_token.filter(|token| !token.trim().is_empty()) {
            return Ok(Some((
                XaiTokens {
                    access_token,
                    refresh_token: None,
                    expires_at_unix: None,
                    id_token: None,
                },
                TokenAuthSource::Env,
            )));
        }
        Ok(load_xai_tokens(store)?.map(|tokens| (tokens, TokenAuthSource::Store)))
    }
}

impl UsageProvider for XaiUsage {
    type Tokens = XaiTokens;
    type Source = TokenAuthSource;
    type Payload = XaiBillingPayload;

    const PROVIDER: &'static str = "SpaceXAI";
    const LOGIN: &'static str = "/login xai-oauth";
    const URL: &'static str = XAI_BILLING_URL;

    fn is_auth_failure(status: reqwest::StatusCode) -> bool {
        unauthorized_or_forbidden(status)
    }

    fn configured_tokens(
        store: &dyn CredentialStore,
    ) -> ConfiguredTokens<XaiTokens, TokenAuthSource> {
        Self::configured_tokens_from(store, std::env::var("XAI_ACCESS_TOKEN").ok())
    }

    fn authorize(request: reqwest::RequestBuilder, tokens: &XaiTokens) -> reqwest::RequestBuilder {
        request
            .bearer_auth(&tokens.access_token)
            .header("x-xai-token-auth", XAI_TOKEN_AUTH_HEADER)
            .header("x-grok-client-version", XAI_CLIENT_VERSION)
            .header(
                reqwest::header::USER_AGENT,
                format!(
                    "rho/{}/grok-shell/{XAI_CLIENT_VERSION}",
                    env!("CARGO_PKG_VERSION")
                ),
            )
            .header(reqwest::header::ACCEPT, "application/json")
    }

    fn refresh<'a>(
        client: &'a reqwest::Client,
        store: &'a dyn CredentialStore,
        tokens: &'a XaiTokens,
        source: TokenAuthSource,
    ) -> RefreshFuture<'a, XaiTokens> {
        Box::pin(async move {
            let (TokenAuthSource::Store, Some(refresh_token)) =
                (source, tokens.refresh_token.clone())
            else {
                return Ok(None);
            };
            refresh_xai_token(client, store, &refresh_token, tokens)
                .await
                .map(Some)
                .map_err(|detail| UsageLimitsError::Refresh {
                    provider: Self::PROVIDER,
                    detail,
                })
        })
    }

    fn windows(payload: XaiBillingPayload) -> Vec<UsageLimitWindow> {
        payload.windows()
    }
}

struct OpenCodeGoUsage;

#[derive(Clone, PartialEq, Eq)]
struct OpenCodeGoKey(String);

/// Redacted so assertion and log output never print the secret; credential
/// types in this codebase must not expose their value through `Debug`.
impl std::fmt::Debug for OpenCodeGoKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OpenCodeGoKey(redacted)")
    }
}

impl OpenCodeGoUsage {
    fn configured_tokens_from(
        store: &dyn CredentialStore,
        env_api_key: Option<String>,
    ) -> ConfiguredTokens<OpenCodeGoKey, TokenAuthSource> {
        if let Some(api_key) = env_api_key.filter(|key| !key.trim().is_empty()) {
            return Ok(Some((OpenCodeGoKey(api_key), TokenAuthSource::Env)));
        }
        Ok(load_provider_api_key(store, "opencode-go")?
            .filter(|key| !key.trim().is_empty())
            .map(|key| (OpenCodeGoKey(key), TokenAuthSource::Store)))
    }
}

impl UsageProvider for OpenCodeGoUsage {
    type Tokens = OpenCodeGoKey;
    type Source = TokenAuthSource;
    type Payload = OpenCodeGoUsagePayload;

    const PROVIDER: &'static str = "OpenCode Go";
    const LOGIN: &'static str = "/login opencode-go";
    const URL: &'static str = OPENCODE_GO_USAGE_URL;

    fn configured_tokens(
        store: &dyn CredentialStore,
    ) -> ConfiguredTokens<OpenCodeGoKey, TokenAuthSource> {
        Self::configured_tokens_from(store, std::env::var("OPENCODE_API_KEY").ok())
    }

    fn authorize(
        request: reqwest::RequestBuilder,
        tokens: &OpenCodeGoKey,
    ) -> reqwest::RequestBuilder {
        request
            .bearer_auth(&tokens.0)
            .header(reqwest::header::ACCEPT, "application/json")
    }

    fn refresh<'a>(
        _client: &'a reqwest::Client,
        _store: &'a dyn CredentialStore,
        _tokens: &'a OpenCodeGoKey,
        _source: TokenAuthSource,
    ) -> RefreshFuture<'a, OpenCodeGoKey> {
        Box::pin(async move { Ok(None) })
    }

    fn windows(payload: OpenCodeGoUsagePayload) -> Vec<UsageLimitWindow> {
        payload.windows()
    }
}

impl UsageProviderKind {
    pub const ALL: [Self; 5] = [
        Self::Claude,
        Self::Codex,
        Self::KimiCode,
        Self::OpenCodeGo,
        Self::Xai,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Claude => ClaudeUsage::PROVIDER,
            Self::Codex => CodexUsage::PROVIDER,
            Self::KimiCode => KimiUsage::PROVIDER,
            Self::OpenCodeGo => OpenCodeGoUsage::PROVIDER,
            Self::Xai => XaiUsage::PROVIDER,
        }
    }
}

pub fn usage_provider_is_connected(
    kind: UsageProviderKind,
    store: &dyn CredentialStore,
) -> Result<bool, UsageLimitsError> {
    match kind {
        UsageProviderKind::Claude => {
            ClaudeUsage::configured_tokens(store).map(|tokens| tokens.is_some())
        }
        UsageProviderKind::Codex => {
            CodexUsage::configured_tokens(store).map(|tokens| tokens.is_some())
        }
        UsageProviderKind::KimiCode => {
            KimiUsage::configured_tokens(store).map(|tokens| tokens.is_some())
        }
        UsageProviderKind::OpenCodeGo => {
            OpenCodeGoUsage::configured_tokens(store).map(|tokens| tokens.is_some())
        }
        UsageProviderKind::Xai => XaiUsage::configured_tokens(store).map(|tokens| tokens.is_some()),
    }
}

pub async fn fetch_usage_provider(
    kind: UsageProviderKind,
    store: &dyn CredentialStore,
    client: reqwest::Client,
) -> Result<Option<ProviderUsageLimits>, UsageLimitsError> {
    match kind {
        UsageProviderKind::Claude => ClaudeUsageLimitsSource::new(client).fetch(store).await,
        UsageProviderKind::Codex => CodexUsageLimitsSource::new(client).fetch(store).await,
        UsageProviderKind::KimiCode => KimiUsageLimitsSource::new(client).fetch(store).await,
        UsageProviderKind::OpenCodeGo => {
            OpenCodeGoUsageLimitsSource::new(client).fetch(store).await
        }
        UsageProviderKind::Xai => XaiUsageLimitsSource::new(client).fetch(store).await,
    }
}

/// `GET /api/oauth/usage` body. Unknown windows (`seven_day_oauth_apps`,
/// `cinder_cove`, `wattle_ember`, …) deserialize to absent fields and stay
/// unreported rather than breaking the read.
#[derive(Deserialize)]
struct ClaudeUsagePayload {
    five_hour: Option<ClaudeUsageWindow>,
    seven_day: Option<ClaudeUsageWindow>,
    seven_day_opus: Option<ClaudeUsageWindow>,
    seven_day_sonnet: Option<ClaudeUsageWindow>,
    #[serde(default)]
    model_scoped: Vec<ClaudeUsageWindow>,
    extra_usage: Option<ClaudeExtraUsage>,
    #[serde(default)]
    limits: Vec<ClaudeLimitRow>,
}

/// One subscription window. `display_name` is set only inside `model_scoped`.
#[derive(Deserialize)]
struct ClaudeUsageWindow {
    utilization: Option<f64>,
    #[serde(default)]
    resets_at: Option<ClaudeResetAt>,
    #[serde(default)]
    display_name: Option<String>,
}

/// `resets_at` comes back as ISO 8601, epoch seconds behind some gateways.
#[derive(Deserialize)]
#[serde(untagged)]
enum ClaudeResetAt {
    Unix(i64),
    Iso(String),
}

impl ClaudeResetAt {
    fn unix(&self) -> Option<i64> {
        match self {
            Self::Unix(seconds) => Some(*seconds),
            Self::Iso(value) => parse_unix_timestamp(value),
        }
    }
}

/// Paid overage block; `monthly_limit`/`used_credits` are cents.
#[derive(Deserialize)]
struct ClaudeExtraUsage {
    is_enabled: Option<bool>,
    monthly_limit: Option<f64>,
    used_credits: Option<f64>,
    utilization: Option<f64>,
    currency: Option<String>,
}

/// A `limits[]` row; only the active row's reset feeds the extra-usage window.
#[derive(Deserialize)]
struct ClaudeLimitRow {
    is_active: Option<bool>,
    resets_at: Option<ClaudeResetAt>,
}

impl ClaudeUsageWindow {
    /// `utilization` is the percent used; a null one means the account has no
    /// such window, not an exhausted one.
    fn into_window(self, label: String) -> Option<UsageLimitWindow> {
        let used = self.utilization?;
        Some(UsageLimitWindow {
            label,
            remaining_percent: Some((100.0 - used).clamp(0.0, 100.0)),
            resets_at_unix: self.resets_at.and_then(|resets_at| resets_at.unix()),
            note: None,
        })
    }
}

impl ClaudeExtraUsage {
    fn into_window(self, resets_at_unix: Option<i64>) -> Option<UsageLimitWindow> {
        if self.is_enabled == Some(false) {
            return None;
        }
        let used = self.utilization?;
        let note = match (self.used_credits, self.monthly_limit) {
            (Some(used_credits), Some(limit)) if limit > 0.0 => {
                let currency = self.currency.as_deref().unwrap_or("USD");
                if currency == "USD" {
                    Some(format!(
                        "${:.2} / ${:.2} spent",
                        used_credits / 100.0,
                        limit / 100.0
                    ))
                } else {
                    Some(format!(
                        "{:.2} / {:.2} {currency} spent",
                        used_credits / 100.0,
                        limit / 100.0
                    ))
                }
            }
            _ => None,
        };
        Some(UsageLimitWindow {
            label: "Extra usage".into(),
            remaining_percent: Some((100.0 - used).clamp(0.0, 100.0)),
            resets_at_unix,
            note,
        })
    }
}

impl ClaudeUsagePayload {
    fn windows(self) -> Vec<UsageLimitWindow> {
        let mut windows: Vec<UsageLimitWindow> = [
            ("5-hour", self.five_hour),
            ("Weekly", self.seven_day),
            ("Weekly (Opus)", self.seven_day_opus),
            ("Weekly (Sonnet)", self.seven_day_sonnet),
        ]
        .into_iter()
        .filter_map(|(label, window)| window.and_then(|window| window.into_window(label.into())))
        .collect();
        windows.extend(self.model_scoped.into_iter().filter_map(|window| {
            let name = window
                .display_name
                .clone()
                .filter(|name| !name.trim().is_empty())?;
            window.into_window(format!("Weekly ({name})"))
        }));
        let extra_resets_at = self
            .limits
            .iter()
            .find(|row| row.is_active == Some(true))
            .and_then(|row| row.resets_at.as_ref().and_then(ClaudeResetAt::unix));
        windows.extend(
            self.extra_usage
                .into_iter()
                .filter_map(|extra| extra.into_window(extra_resets_at)),
        );
        windows
    }
}

#[derive(Deserialize)]
struct CodexUsagePayload {
    rate_limit: Option<CodexRateLimit>,
}

#[derive(Deserialize)]
struct CodexRateLimit {
    primary_window: Option<CodexLimitWindow>,
    secondary_window: Option<CodexLimitWindow>,
}

#[derive(Deserialize)]
struct CodexLimitWindow {
    used_percent: f64,
    limit_window_seconds: i64,
    reset_at: i64,
}

impl CodexUsagePayload {
    fn windows(self) -> Vec<UsageLimitWindow> {
        self.rate_limit
            .into_iter()
            .flat_map(|limits| [limits.primary_window, limits.secondary_window])
            .flatten()
            .map(UsageLimitWindow::from)
            .collect()
    }
}

impl From<CodexLimitWindow> for UsageLimitWindow {
    fn from(window: CodexLimitWindow) -> Self {
        Self {
            label: window_label(window.limit_window_seconds),
            remaining_percent: Some((100.0 - window.used_percent).clamp(0.0, 100.0)),
            resets_at_unix: Some(window.reset_at),
            note: None,
        }
    }
}

#[derive(Deserialize)]
struct KimiUsagePayload {
    usage: Option<KimiUsageDetail>,
    #[serde(default)]
    limits: Vec<KimiUsageLimit>,
}

#[derive(Deserialize)]
struct KimiUsageLimit {
    window: Option<KimiUsageWindow>,
    detail: KimiUsageDetail,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct KimiUsageWindow {
    duration: i64,
    time_unit: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct KimiUsageDetail {
    limit: KimiNumber,
    used: Option<KimiNumber>,
    remaining: Option<KimiNumber>,
    reset_time: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum KimiNumber {
    Number(f64),
    String(String),
}

impl KimiNumber {
    fn value(&self) -> Option<f64> {
        match self {
            Self::Number(value) => Some(*value),
            Self::String(value) => value.parse().ok(),
        }
    }
}

impl KimiUsagePayload {
    fn windows(self) -> Vec<UsageLimitWindow> {
        self.limits
            .into_iter()
            .filter_map(|limit| {
                let label = limit
                    .window
                    .as_ref()
                    .and_then(KimiUsageWindow::label)
                    .unwrap_or_else(|| "Usage".into());
                limit.detail.into_window(label)
            })
            .chain(
                self.usage
                    .into_iter()
                    .filter_map(|detail| detail.into_window("Weekly".into())),
            )
            .collect()
    }
}

impl KimiUsageWindow {
    fn label(&self) -> Option<String> {
        let seconds = if self.time_unit.contains("MINUTE") {
            self.duration.checked_mul(60)?
        } else if self.time_unit.contains("HOUR") {
            self.duration.checked_mul(60 * 60)?
        } else if self.time_unit.contains("DAY") {
            self.duration.checked_mul(24 * 60 * 60)?
        } else {
            return None;
        };
        Some(window_label(seconds))
    }
}

impl KimiUsageDetail {
    fn into_window(self, label: String) -> Option<UsageLimitWindow> {
        let limit = self.limit.value()?;
        if limit <= 0.0 {
            return None;
        }
        let remaining = self
            .remaining
            .as_ref()
            .and_then(KimiNumber::value)
            .or_else(|| {
                self.used
                    .as_ref()
                    .and_then(KimiNumber::value)
                    .map(|used| limit - used)
            })?;
        Some(UsageLimitWindow {
            label,
            remaining_percent: Some((remaining / limit * 100.0).clamp(0.0, 100.0)),
            resets_at_unix: Some(parse_unix_timestamp(&self.reset_time)?),
            note: None,
        })
    }
}

#[derive(Deserialize)]
struct XaiBillingPayload {
    config: Option<XaiBillingConfig>,
    #[serde(flatten)]
    root: XaiBillingConfig,
}

#[derive(Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
struct XaiBillingConfig {
    credit_usage_percent: Option<f64>,
    current_period: Option<XaiBillingPeriod>,
    billing_period_end: Option<String>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct XaiBillingPeriod {
    #[serde(rename = "type")]
    kind: Option<String>,
    end: Option<String>,
}

impl XaiBillingPayload {
    fn windows(self) -> Vec<UsageLimitWindow> {
        let config = self.config.unwrap_or(self.root);
        let Some(used_percent) = config.credit_usage_percent else {
            return Vec::new();
        };
        let period = config.current_period;
        let label = period
            .as_ref()
            .and_then(|period| period.kind.as_deref())
            .map(xai_period_label)
            .unwrap_or("Usage");
        let Some(resets_at_unix) = period
            .and_then(|period| period.end)
            .or(config.billing_period_end)
            .and_then(|value| parse_unix_timestamp(&value))
        else {
            return Vec::new();
        };
        vec![UsageLimitWindow {
            label: label.into(),
            remaining_percent: Some((100.0 - used_percent).clamp(0.0, 100.0)),
            resets_at_unix: Some(resets_at_unix),
            note: None,
        }]
    }
}

#[derive(Deserialize)]
struct OpenCodeGoUsagePayload {
    usage: Option<OpenCodeGoUsageWindows>,
}

#[derive(Deserialize)]
struct OpenCodeGoUsageWindows {
    rolling: Option<OpenCodeGoUsageWindow>,
    weekly: Option<OpenCodeGoUsageWindow>,
    monthly: Option<OpenCodeGoUsageWindow>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct OpenCodeGoUsageWindow {
    status: Option<String>,
    percent: Option<f64>,
    resets_at: Option<String>,
}

impl OpenCodeGoUsagePayload {
    fn windows(self) -> Vec<UsageLimitWindow> {
        let Some(usage) = self.usage else {
            return Vec::new();
        };
        [
            ("5-hour", usage.rolling),
            ("Weekly", usage.weekly),
            ("Monthly", usage.monthly),
        ]
        .into_iter()
        .filter_map(|(label, window)| window.and_then(|window| window.into_window(label)))
        .collect()
    }
}

impl OpenCodeGoUsageWindow {
    fn into_window(self, label: &str) -> Option<UsageLimitWindow> {
        let used = self.percent?;
        Some(UsageLimitWindow {
            label: label.into(),
            remaining_percent: Some((100.0 - used).clamp(0.0, 100.0)),
            resets_at_unix: self.resets_at.as_deref().and_then(parse_unix_timestamp),
            note: match self.status.as_deref() {
                Some("rate-limited") => Some("rate-limited".into()),
                _ => None,
            },
        })
    }
}

fn xai_period_label(kind: &str) -> &'static str {
    match kind {
        "USAGE_PERIOD_TYPE_WEEKLY" => "Weekly",
        "USAGE_PERIOD_TYPE_MONTHLY" => "Monthly",
        "USAGE_PERIOD_TYPE_DAILY" => "Daily",
        _ => "Usage",
    }
}

async fn refresh_xai_token(
    client: &reqwest::Client,
    store: &dyn CredentialStore,
    refresh_token: &str,
    previous: &XaiTokens,
) -> Result<XaiTokens, String> {
    let refreshed = refresh_xai_tokens(client, refresh_token, previous)
        .await
        .map_err(|err| err.to_string())?;
    save_xai_tokens(store, &refreshed).map_err(|err| err.to_string())?;
    Ok(refreshed)
}

fn parse_unix_timestamp(value: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(value)
        .map(|value| value.timestamp())
        .ok()
        .or_else(|| {
            chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.fZ")
                .ok()
                .map(|value| value.and_utc().timestamp())
        })
}

fn window_label(seconds: i64) -> String {
    const HOUR: i64 = 60 * 60;
    const DAY: i64 = 24 * HOUR;
    const WEEK: i64 = 7 * DAY;
    if approximately(seconds, 5 * HOUR) {
        "5-hour".into()
    } else if approximately(seconds, WEEK) {
        "Weekly".into()
    } else if approximately(seconds, DAY) {
        "Daily".into()
    } else if seconds >= DAY && seconds % DAY == 0 {
        format!("{}-day", seconds / DAY)
    } else if seconds >= HOUR && seconds % HOUR == 0 {
        format!("{}-hour", seconds / HOUR)
    } else {
        "Usage".into()
    }
}

fn approximately(actual: i64, expected: i64) -> bool {
    actual >= expected * 95 / 100 && actual <= expected * 105 / 100
}

pub fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claude_payload_maps_subscription_windows() {
        let payload: ClaudeUsagePayload = serde_json::from_str(
            r#"{
                "five_hour": {"utilization": 42.0, "resets_at": "2026-10-01T12:00:00.000Z"},
                "seven_day": {"utilization": 10.5, "resets_at": "2026-10-08T00:00:00Z"},
                "seven_day_sonnet": {"utilization": null, "resets_at": null},
                "model_scoped": [
                    {"display_name": "Fable", "utilization": 25.0, "resets_at": 1791504000}
                ],
                "extra_usage": {
                    "is_enabled": true,
                    "monthly_limit": 50000.0,
                    "used_credits": 27140.0,
                    "utilization": 54.0,
                    "currency": "USD"
                },
                "limits": [
                    {"kind": "spend", "group": "monthly", "percent": 54.0,
                     "resets_at": "2026-11-01T00:00:00.000Z", "is_active": true}
                ]
            }"#,
        )
        .expect("payload parses");
        let windows = payload.windows();
        let labels: Vec<&str> = windows.iter().map(|w| w.label.as_str()).collect();
        assert_eq!(
            labels,
            ["5-hour", "Weekly", "Weekly (Fable)", "Extra usage"]
        );
        assert_eq!(windows[0].remaining_percent, Some(58.0));
        assert!(windows[0].resets_at_unix.is_some());
        assert_eq!(
            windows[3].note.as_deref(),
            Some("$271.40 / $500.00 spent")
        );
        assert!(windows[3].resets_at_unix.is_some());
    }

    #[test]
    fn claude_payload_tolerates_absent_windows() {
        let payload: ClaudeUsagePayload = serde_json::from_str("{}").expect("empty parses");
        assert!(payload.windows().is_empty());
        let payload: ClaudeUsagePayload =
            serde_json::from_str(r#"{"seven_day": {"utilization": 130.0}}"#)
                .expect("partial parses");
        let windows = payload.windows();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].remaining_percent, Some(0.0));
    }
}
