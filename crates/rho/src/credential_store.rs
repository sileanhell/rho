use std::{
    path::Path,
    sync::{Arc, Mutex, OnceLock},
};

use rho_providers::credentials::{
    CredentialError, CredentialResult, CredentialStore, FileCredentialStore,
};

use crate::config::Config;

/// Application credential adapter over the file-backed store.
///
/// Secrets live in private files under the Rho home directory
/// (`RHO_HOME` or `~/.rho/credentials`); the store rereads the file on every
/// access, so logins saved by another process are visible immediately.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AppCredentialStore;

impl CredentialStore for AppCredentialStore {
    fn get_secret(&self, account: &str) -> CredentialResult<Option<String>> {
        selected_store()?.get_secret(account)
    }

    fn set_secret(&self, account: &str, secret: &str) -> CredentialResult<()> {
        selected_store()?.set_secret(account, secret)
    }

    fn delete_secret(&self, account: &str) -> CredentialResult<bool> {
        selected_store()?.delete_secret(account)
    }
}

/// Open the file store once for this process.
///
/// Also migrates legacy policy/web-search secrets when needed. Call at startup,
/// not on every settings read.
pub(crate) fn initialize_from_config(
    config: &mut Config,
    config_path: &Path,
) -> anyhow::Result<()> {
    activate()?;
    let mut dirty = false;
    if matches!(
        config.migrate_legacy_web_search_credentials(&AppCredentialStore),
        Ok(true)
    ) {
        dirty = true;
    }
    if config.promote_stored_custom_auth(&AppCredentialStore) {
        dirty = true;
    }
    if dirty {
        config.write_settings(config_path.to_path_buf())?;
    }
    Ok(())
}

/// Opens the store for a login inside a running process, such as ACP
/// `authenticate`. Unlike [`initialize_from_config`] it runs no migrations
/// and writes nothing, so it is safe beside startup or a live session.
pub(crate) fn activate_for_login(_config: &Config, _config_path: &Path) -> anyhow::Result<()> {
    activate()?;
    Ok(())
}

pub(crate) fn build_provider_from_config(
    config: &Config,
    credential_store: Arc<dyn CredentialStore>,
) -> Result<Arc<dyn rho_sdk::provider::ModelProvider>, rho_providers::model::ModelError> {
    let _scope = config
        .providers
        .thread_scope()
        .map_err(|error| rho_providers::model::ModelError::InvalidResponse(error.to_string()))?;
    let options = crate::app::sdk_config::provider_options_from_config(config)?;
    let credentials = rho_providers::auth::provider_credentials::ApplicationCredentialSource::new(
        credential_store,
    );
    rho_providers::providers::build_sdk_provider_with_source(options, &credentials)
}

/// Live rebuilds await catalog hydrate so npm-following providers pick the
/// advertised adapter before the first request, not after a background fetch.
///
/// Options are resolved synchronously so the thread-local custom-provider
/// overlay is never held across the catalog await.
pub(crate) async fn build_provider_from_config_ensuring_catalog(
    config: &Config,
    credential_store: Arc<dyn CredentialStore>,
) -> Result<Arc<dyn rho_sdk::provider::ModelProvider>, rho_providers::model::ModelError> {
    let options = crate::app::sdk_config::provider_options_from_config(config)?;
    options.ensure_catalog_for_construction().await;
    build_provider_from_config(config, credential_store)
}

pub(crate) async fn build_provider(
    provider: &str,
    model: &str,
    reasoning: rho_providers::reasoning::ReasoningLevel,
    auth: &str,
) -> Result<Arc<dyn rho_sdk::provider::ModelProvider>, rho_providers::model::ModelError> {
    build_provider_on(&Config::default(), provider, model, reasoning, auth).await
}

/// [`build_provider`] on `base`'s `[providers]`, so a configured endpoint or
/// custom provider reaches the model. Nothing else carries over from `base`.
pub(crate) async fn build_provider_on(
    base: &Config,
    provider: &str,
    model: &str,
    reasoning: rho_providers::reasoning::ReasoningLevel,
    auth: &str,
) -> Result<Arc<dyn rho_sdk::provider::ModelProvider>, rho_providers::model::ModelError> {
    let config = Config {
        provider: provider.into(),
        model: model.into(),
        reasoning,
        auth: auth.into(),
        providers: base.providers.clone(),
        ..Config::default()
    };
    build_provider_from_config_ensuring_catalog(&config, Arc::new(AppCredentialStore)).await
}

fn activate() -> CredentialResult<()> {
    let store: Arc<dyn CredentialStore> = Arc::new(FileCredentialStore::open()?);
    let mut guard = process_state()
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    *guard = Some(store);
    Ok(())
}

fn selected_store() -> CredentialResult<Arc<dyn CredentialStore>> {
    {
        let guard = process_state()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(store) = guard.as_ref() {
            return Ok(Arc::clone(store));
        }
    }
    activate()?;
    process_state()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone()
        .ok_or_else(|| {
            CredentialError::StoreUnavailable("credential store failed to initialize".into())
        })
}

fn process_state() -> &'static Mutex<Option<Arc<dyn CredentialStore>>> {
    static STATE: OnceLock<Mutex<Option<Arc<dyn CredentialStore>>>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(None))
}
