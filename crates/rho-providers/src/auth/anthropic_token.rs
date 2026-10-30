//! Anthropic OAuth token management: proactive refresh and 401 force-refresh.
//!
//! Mirrors [`super::xai_token`]'s shape. `Store` tokens are Rho-owned and
//! commit back to the credential store on refresh; `Static` covers env
//! tokens, which cannot refresh.

use std::sync::Arc;

use tokio::sync::Mutex;

use crate::{
    auth::anthropic_oauth,
    credentials::{save_refreshed_tokens, AnthropicTokens, CredentialStore},
    model::ModelError,
    provider::ANTHROPIC_TOKENS_ACCOUNT,
};

const REFRESH_SKEW_MS: i64 = 60_000;
static REFRESH_LOCK: Mutex<()> = Mutex::const_new(());

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnthropicAuthSource {
    /// A static token from the environment (`ANTHROPIC_TOKEN` /
    /// `CLAUDE_CODE_OAUTH_TOKEN`). Never refreshed.
    Static,
    /// Rho-owned tokens from the credential store (`/login anthropic-oauth`).
    Store,
}

#[derive(Clone)]
pub(crate) struct AnthropicAuthMaterial {
    pub access_token: String,
    /// True for OAuth tokens on the native route: send Bearer + Claude Code
    /// identity headers and request transforms.
    pub oauth: bool,
}

impl std::fmt::Debug for AnthropicAuthMaterial {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnthropicAuthMaterial")
            .field("access_token", &"[REDACTED]")
            .field("oauth", &self.oauth)
            .finish()
    }
}

pub struct AnthropicAuthManager {
    client: reqwest::Client,
    store: Option<Arc<dyn CredentialStore>>,
    source: AnthropicAuthSource,
    tokens: Mutex<AnthropicTokens>,
}

impl std::fmt::Debug for AnthropicAuthManager {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AnthropicAuthManager")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl AnthropicAuthManager {
    /// Static OAuth-shaped credential (env token or setup token). It
    /// authenticates as Bearer but cannot refresh.
    pub(crate) fn from_static_token(access_token: String) -> Self {
        Self {
            client: crate::reqwest_client(),
            store: None,
            source: AnthropicAuthSource::Static,
            tokens: Mutex::new(AnthropicTokens {
                access_token,
                refresh_token: None,
                expires_at_ms: None,
            }),
        }
    }

    pub(crate) fn from_store(store: Arc<dyn CredentialStore>, tokens: AnthropicTokens) -> Self {
        Self {
            client: crate::reqwest_client(),
            store: Some(store),
            source: AnthropicAuthSource::Store,
            tokens: Mutex::new(tokens),
        }
    }

    /// Access token for one request, refreshing when the stored pair is
    /// inside the expiry skew.
    pub(crate) async fn auth_material(&self) -> Result<AnthropicAuthMaterial, ModelError> {
        let tokens = self.tokens.lock().await.clone();
        match self.source {
            AnthropicAuthSource::Static => Ok(AnthropicAuthMaterial {
                access_token: tokens.access_token,
                oauth: true,
            }),
            AnthropicAuthSource::Store if token_is_expiring(&tokens) => {
                self.refresh_if_current(&tokens.access_token).await
            }
            AnthropicAuthSource::Store => Ok(AnthropicAuthMaterial {
                access_token: tokens.access_token,
                oauth: true,
            }),
        }
    }

    /// Force a refresh after a 401, unless the token already changed.
    /// Returns `None` when no refresh is possible (static token, no refresh
    /// grant) so the caller surfaces the original error.
    pub(crate) async fn force_refresh(
        &self,
        failed_access_token: &str,
    ) -> Result<Option<AnthropicAuthMaterial>, ModelError> {
        match self.source {
            AnthropicAuthSource::Static => Ok(None),
            AnthropicAuthSource::Store => {
                if self.tokens.lock().await.refresh_token.is_none() {
                    return Ok(None);
                }
                self.refresh_if_current(failed_access_token).await.map(Some)
            }
        }
    }

    async fn refresh_if_current(
        &self,
        failed_access_token: &str,
    ) -> Result<AnthropicAuthMaterial, ModelError> {
        let _guard = REFRESH_LOCK.lock().await;
        let mut current = self.tokens.lock().await;
        if current.access_token != failed_access_token {
            return Ok(AnthropicAuthMaterial {
                access_token: current.access_token.clone(),
                oauth: true,
            });
        }
        let refresh_token = current
            .refresh_token
            .clone()
            .ok_or_else(|| crate::model::registry::missing_credentials_error("anthropic-oauth"))?;
        let refreshed = self.refresh(&refresh_token).await?;
        if let Some(store) = &self.store {
            save_refreshed_tokens(
                store.as_ref(),
                ANTHROPIC_TOKENS_ACCOUNT,
                &current.clone(),
                &refreshed,
            )?;
        }
        let access_token = refreshed.access_token.clone();
        *current = refreshed;
        Ok(AnthropicAuthMaterial {
            access_token,
            oauth: true,
        })
    }

    async fn refresh(&self, refresh_token: &str) -> Result<AnthropicTokens, ModelError> {
        anthropic_oauth::refresh_anthropic_tokens(&self.client, refresh_token)
            .await
            .map_err(|error| {
                if error.relogin_required() {
                    crate::model::registry::missing_credentials_error("anthropic-oauth")
                } else {
                    ModelError::InvalidResponse(error.to_string())
                }
            })
    }
}

fn token_is_expiring(tokens: &AnthropicTokens) -> bool {
    tokens.expires_at_ms.is_some_and(|expires_at_ms| {
        expires_at_ms <= anthropic_oauth::now_unix_ms().saturating_add(REFRESH_SKEW_MS)
    })
}
