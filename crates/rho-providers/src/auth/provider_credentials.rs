use std::sync::Arc;

use rho_sdk::SecretString;

use crate::{
    auth::{
        anthropic_oauth,
        anthropic_token::AnthropicAuthManager,
        github_copilot_token::GitHubCopilotAuthManager,
        kimi_token::{KimiAuthManager, KimiAuthSource},
        meta_token::MetaAuthManager,
        ollama_device::OllamaDeviceKey,
        xai_token::{XaiAuthManager, XaiAuthSource},
    },
    credentials::{
        load_anthropic_tokens, load_codex_tokens, load_kimi_tokens, load_meta_tokens,
        load_provider_api_key, load_xai_tokens, CodexTokens, CredentialStore, KimiTokens,
        XaiTokens,
    },
    model::{
        registry::{
            missing_credential_error, missing_credentials_error, provider_runtime, ProviderRuntime,
        },
        ModelError,
    },
    provider::{self, OpenAiRuntimeAuth, ProviderAuthKind},
    providers::{
        builder::ProviderCredential,
        openai::auth::{Auth, CodexAuthSource},
        openai_compatible::CompatibleAuth,
    },
};

/// Opt-in application adapter for environment and credential-store lookup.
///
/// Provider builders never invoke this adapter implicitly. The application
/// bootstrap chooses when to acquire credentials and passes the returned value
/// into provider construction. Login and keychain UX therefore remain outside
/// provider execution and outside `rho-sdk`.
pub trait ProviderCredentialSource: Send + Sync {
    fn acquire(&self, provider: &str, auth: &str) -> Result<ProviderCredential, ModelError>;
}

/// Rho's first-party environment and OS-keychain credential adapter.
///
/// Environment overrides are evaluated only when [`Self::acquire`] is called.
/// The configured store is retained only by OAuth transports that need to
/// persist refreshed tokens; API-key transports receive an owned secret value.
#[derive(Clone)]
pub struct ApplicationCredentialSource {
    store: Arc<dyn CredentialStore>,
}

impl ApplicationCredentialSource {
    pub fn new(store: Arc<dyn CredentialStore>) -> Self {
        Self { store }
    }
}

impl ProviderCredentialSource for ApplicationCredentialSource {
    fn acquire(&self, provider: &str, auth: &str) -> Result<ProviderCredential, ModelError> {
        let profile = provider::resolve_profile(provider, auth)
            .map_err(|error| ModelError::InvalidResponse(error.to_string()))?;
        let descriptor = profile.provider;
        let selected = profile.auth;
        let runtime = provider_runtime(descriptor.name)
            .ok_or_else(|| ModelError::UnsupportedProvider(provider.to_string()))?;
        match runtime {
            ProviderRuntime::OpenAi { auth_mode: mode } => {
                let openai_auth = match mode {
                    OpenAiRuntimeAuth::ApiKey => load_openai_api_key_auth(self.store.as_ref())?,
                    OpenAiRuntimeAuth::Codex => load_codex_auth(self.store.clone())?,
                };
                Ok(ProviderCredential::OpenAi { auth: openai_auth })
            }
            ProviderRuntime::Anthropic => Ok(ProviderCredential::Anthropic {
                auth: load_anthropic_auth(selected, self.store.clone())?,
            }),
            ProviderRuntime::Google => Ok(ProviderCredential::GoogleApiKey(SecretString::new(
                load_provider_api_key_auth("google", self.store.as_ref())?,
            ))),
            ProviderRuntime::GithubCopilot => Ok(ProviderCredential::GitHubCopilot(
                GitHubCopilotAuthManager::new(self.store.clone())?,
            )),
            ProviderRuntime::OpenAiCompatible { .. } => {
                let auth = match selected.auth_kind {
                    ProviderAuthKind::None => CompatibleAuth::None,
                    ProviderAuthKind::ApiKey { .. } => CompatibleAuth::ApiKey(
                        load_api_key_for_mode(selected.auth_kind, self.store.as_ref())?,
                    ),
                    ProviderAuthKind::BearerCredential {
                        env_var,
                        account,
                        missing_message,
                        ..
                    } => CompatibleAuth::ApiKey(load_stored_bearer_key(
                        env_var,
                        account,
                        missing_message,
                        self.store.as_ref(),
                    )?),
                    ProviderAuthKind::KimiOAuth { .. } => {
                        let env_var = selected
                            .auth_kind
                            .env_var()
                            .expect("Kimi OAuth must declare an environment variable");
                        let missing = missing_credentials_error("kimi-code");
                        let (source, tokens) = env_or_stored(
                            env_var,
                            |access_token| KimiTokens {
                                access_token,
                                refresh_token: None,
                                expires_at_unix: None,
                                scope: String::new(),
                                token_type: "Bearer".into(),
                                expires_in: None,
                            },
                            || Ok(load_kimi_tokens(self.store.as_ref())?),
                            missing,
                            KimiAuthSource::Env,
                            KimiAuthSource::Store,
                        )?;
                        CompatibleAuth::KimiOAuth(KimiAuthManager::from_tokens(
                            self.store.clone(),
                            source,
                            tokens,
                        ))
                    }
                    ProviderAuthKind::MetaOAuth { .. } => {
                        let env_var = selected
                            .auth_kind
                            .env_var()
                            .expect("Muse subscription auth must declare an environment variable");
                        if let Some(api_key) = crate::auth::meta_token::env_api_key(env_var) {
                            CompatibleAuth::ApiKey(api_key)
                        } else {
                            let tokens = load_meta_tokens(self.store.as_ref())?
                                .ok_or_else(|| missing_credentials_error("meta-muse"))?;
                            CompatibleAuth::MetaOAuth(MetaAuthManager::from_tokens(
                                self.store.clone(),
                                tokens,
                            ))
                        }
                    }
                    ProviderAuthKind::OllamaDeviceKey { missing_message } => {
                        CompatibleAuth::OllamaDevice(load_ollama_device_key(
                            self.store.as_ref(),
                            missing_message,
                        )?)
                    }
                    ProviderAuthKind::AnthropicOAuth { .. }
                    | ProviderAuthKind::CodexOAuth { .. }
                    | ProviderAuthKind::GithubCopilotDevice { .. }
                    | ProviderAuthKind::XaiOAuth { .. } => {
                        return Err(ModelError::UnsupportedProvider(provider.into()));
                    }
                };
                Ok(ProviderCredential::OpenAiCompatible(auth))
            }
            // Decision-model hosts serve no chat models; model selection
            // refuses them before this point.
            ProviderRuntime::SystemOne { .. } => {
                Err(ModelError::UnsupportedProvider(provider.to_string()))
            }
            ProviderRuntime::Xai => {
                let (source, tokens) = match selected.auth_kind {
                    ProviderAuthKind::ApiKey { .. } => (
                        XaiAuthSource::ApiKey,
                        XaiTokens {
                            access_token: load_api_key_for_mode(
                                selected.auth_kind,
                                self.store.as_ref(),
                            )?,
                            refresh_token: None,
                            expires_at_unix: None,
                            id_token: None,
                        },
                    ),
                    ProviderAuthKind::XaiOAuth {
                        env_var,
                        missing_message,
                        ..
                    } => env_or_stored(
                        env_var,
                        |access_token| XaiTokens {
                            access_token,
                            refresh_token: None,
                            expires_at_unix: None,
                            id_token: None,
                        },
                        || Ok(load_xai_tokens(self.store.as_ref())?),
                        missing_credential_error(missing_message),
                        XaiAuthSource::Env,
                        XaiAuthSource::Store,
                    )?,
                    _ => return Err(ModelError::UnsupportedProvider(provider.into())),
                };
                Ok(ProviderCredential::Xai(XaiAuthManager::from_tokens(
                    self.store.clone(),
                    source,
                    tokens,
                )))
            }
        }
    }
}

pub(crate) fn load_stored_bearer_key(
    env_var: &str,
    account: &str,
    missing_message: &'static str,
    store: &dyn CredentialStore,
) -> Result<String, ModelError> {
    if let Ok(key) = std::env::var(env_var) {
        if !key.trim().is_empty() {
            return Ok(key);
        }
    }
    store
        .get_secret(account)?
        .filter(|key| !key.trim().is_empty())
        .ok_or_else(|| missing_credential_error(missing_message))
}

fn env_or_stored<T, S>(
    env_var: &str,
    from_env: impl FnOnce(String) -> T,
    load: impl FnOnce() -> Result<Option<T>, ModelError>,
    missing: ModelError,
    env_source: S,
    store_source: S,
) -> Result<(S, T), ModelError> {
    match std::env::var(env_var) {
        Ok(value) if !value.trim().is_empty() => Ok((env_source, from_env(value))),
        _ => Ok((store_source, load()?.ok_or(missing)?)),
    }
}

pub(crate) fn load_api_key_for_mode(
    auth_kind: ProviderAuthKind,
    store: &dyn CredentialStore,
) -> Result<String, ModelError> {
    let ProviderAuthKind::ApiKey {
        env_var,
        account,
        missing_message,
        ..
    } = auth_kind
    else {
        return Err(ModelError::InvalidResponse(
            "expected API key auth kind".into(),
        ));
    };
    if let Ok(key) = std::env::var(env_var) {
        if !key.trim().is_empty() {
            return Ok(key);
        }
    }
    store
        .get_secret(account)?
        .filter(|key| !key.trim().is_empty())
        .ok_or_else(|| missing_credential_error(missing_message))
}

fn load_provider_api_key_auth(
    provider_name: &str,
    store: &dyn CredentialStore,
) -> Result<String, ModelError> {
    let descriptor = provider::provider_descriptor(provider_name)
        .ok_or_else(|| ModelError::UnsupportedProvider(provider_name.into()))?;
    let ProviderAuthKind::ApiKey {
        env_var,
        missing_message,
        ..
    } = descriptor.default_auth().auth_kind
    else {
        return Err(ModelError::UnsupportedProvider(provider_name.into()));
    };
    if let Ok(key) = std::env::var(env_var) {
        if !key.trim().is_empty() {
            return Ok(key);
        }
    }
    load_provider_api_key(store, descriptor.name)?
        .ok_or_else(|| missing_credential_error(missing_message))
}

fn load_openai_api_key_auth(store: &dyn CredentialStore) -> Result<Auth, ModelError> {
    let descriptor = provider::provider_descriptor("openai")
        .ok_or_else(|| ModelError::UnsupportedProvider("openai".into()))?;
    let ProviderAuthKind::ApiKey {
        env_var,
        missing_message,
        ..
    } = descriptor.default_auth().auth_kind
    else {
        return Err(ModelError::UnsupportedProvider("openai".into()));
    };
    if let Ok(key) = std::env::var(env_var) {
        if !key.trim().is_empty() {
            return Ok(Auth::ApiKey(key));
        }
    }
    let key = load_provider_api_key(store, descriptor.name)?
        .ok_or_else(|| missing_credential_error(missing_message))?;
    Ok(Auth::ApiKey(key))
}

fn load_codex_auth(store: Arc<dyn CredentialStore>) -> Result<Auth, ModelError> {
    let env_var = provider::provider_descriptor_by_id(provider::ProviderId::OpenAiCodex)
        .default_auth()
        .auth_kind
        .env_var()
        .expect("Codex OAuth must declare an environment variable");
    if let Ok(access_token) = std::env::var(env_var) {
        return Ok(Auth::codex(
            CodexTokens {
                access_token,
                refresh_token: None,
                id_token: None,
                account_id: std::env::var("CODEX_ACCOUNT_ID").ok(),
            },
            CodexAuthSource::Env,
            store,
        ));
    }
    let tokens = load_codex_tokens(store.as_ref())?
        .ok_or_else(|| missing_credentials_error("openai-codex"))?;
    Ok(Auth::codex(tokens, CodexAuthSource::Store, store))
}

/// Anthropic credential resolution for the selected auth mode.
///
/// Anthropic is OAuth-only: `ANTHROPIC_TOKEN` / `CLAUDE_CODE_OAUTH_TOKEN`
/// first, then Rho-owned OAuth tokens in the credential store.
fn load_anthropic_auth(
    selected: provider::AuthMode,
    store: Arc<dyn CredentialStore>,
) -> Result<crate::providers::anthropic::AnthropicAuth, ModelError> {
    use crate::providers::anthropic::AnthropicAuth;
    match selected.auth_kind {
        ProviderAuthKind::AnthropicOAuth {
            env_var,
            missing_message,
            ..
        } => {
            let env_token = [env_var, "CLAUDE_CODE_OAUTH_TOKEN"]
                .into_iter()
                .find_map(|name| {
                    std::env::var(name)
                        .ok()
                        .map(|value| value.trim().to_string())
                        .filter(|value| !value.is_empty())
                });
            if let Some(token) = env_token {
                return Ok(AnthropicAuth::oauth(AnthropicAuthManager::from_static_token(
                    token,
                )));
            }
            if let Some(tokens) = load_anthropic_tokens(store.as_ref())? {
                return Ok(AnthropicAuth::oauth(AnthropicAuthManager::from_store(
                    store, tokens,
                )));
            }
            Err(missing_credential_error(missing_message))
        }
        _ => Err(ModelError::UnsupportedProvider("anthropic".into())),
    }
}

/// Anthropic OAuth access token for model discovery, following the same
/// source order as [`load_anthropic_auth`]. A stored grant refreshes inline
/// and commits through `save_refreshed_tokens`.
pub(crate) async fn resolve_anthropic_request_auth(
    store: &dyn CredentialStore,
) -> Result<String, ModelError> {
    let env_token = ["ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"]
        .into_iter()
        .find_map(|name| {
            std::env::var(name)
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        });
    if let Some(token) = env_token {
        anthropic_oauth::ensure_claude_code_version(&crate::reqwest_client()).await?;
        return Ok(token);
    }
    if let Some(tokens) = load_anthropic_tokens(store)? {
        let tokens = refresh_stored_anthropic_tokens(store, tokens).await?;
        anthropic_oauth::ensure_claude_code_version(&crate::reqwest_client()).await?;
        return Ok(tokens.access_token);
    }
    Err(missing_credentials_error("anthropic"))
}

async fn refresh_stored_anthropic_tokens(
    store: &dyn CredentialStore,
    tokens: crate::credentials::AnthropicTokens,
) -> Result<crate::credentials::AnthropicTokens, ModelError> {
    let Some(expires_at_ms) = tokens.expires_at_ms else {
        return Ok(tokens);
    };
    if expires_at_ms > anthropic_oauth::now_unix_ms() + 60_000 {
        return Ok(tokens);
    }
    let Some(refresh_token) = tokens.refresh_token.clone() else {
        return Ok(tokens);
    };
    let refreshed =
        anthropic_oauth::refresh_anthropic_tokens(&crate::reqwest_client(), &refresh_token)
            .await
            .map_err(|error| {
                if error.relogin_required() {
                    missing_credentials_error("anthropic-oauth")
                } else {
                    ModelError::InvalidResponse(error.to_string())
                }
            })?;
    crate::credentials::save_refreshed_tokens(
        store,
        provider::ANTHROPIC_TOKENS_ACCOUNT,
        &tokens,
        &refreshed,
    )?;
    Ok(refreshed)
}

/// Loads the `~/.ollama` device key once the user signed in to it through Rho.
pub(crate) fn load_ollama_device_key(
    store: &dyn CredentialStore,
    missing_message: &'static str,
) -> Result<OllamaDeviceKey, ModelError> {
    if !crate::credentials::ollama_device_login_recorded(store)? {
        return Err(missing_credential_error(missing_message));
    }
    load_ollama_device_key_from(OllamaDeviceKey::load_default, missing_message)
}

pub(crate) fn load_ollama_device_key_from(
    load: impl FnOnce() -> Result<OllamaDeviceKey, crate::auth::ollama_device::OllamaDeviceError>,
    missing_message: &'static str,
) -> Result<OllamaDeviceKey, ModelError> {
    load().map_err(|error| match error {
        crate::auth::ollama_device::OllamaDeviceError::MissingKey(_) => {
            missing_credential_error(missing_message)
        }
        error => ModelError::InvalidResponse(error.to_string()),
    })
}
