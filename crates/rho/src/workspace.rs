use std::path::{Path, PathBuf};

/// Environment variable that grants a workspace's project hooks.
pub const TRUST_PROJECT_HOOKS_ENV: &str = "RHO_TRUST_PROJECT_HOOKS";
/// Environment variable that grants a workspace's project agent definitions.
pub const TRUST_PROJECT_AGENTS_ENV: &str = "RHO_TRUST_PROJECT_AGENTS";
/// Environment variable that grants a workspace's project Agent Plugins.
pub const TRUST_PROJECT_PLUGINS_ENV: &str = "RHO_TRUST_PROJECT_PLUGINS";
/// Environment variable that grants a workspace's project OpenAPI sources.
pub const TRUST_PROJECT_OPENAPI_ENV: &str = "RHO_TRUST_PROJECT_OPENAPI";

/// Whether a workspace's project-supplied files may activate.
///
/// Shared by project hooks, agents, Agent Plugins, and OpenAPI sources.
/// Each feature has its own env var; only the exact value `1` grants trust.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectTrust {
    Trusted,
    Untrusted,
}

impl ProjectTrust {
    fn from_env(value: Option<&str>) -> Self {
        if value == Some("1") {
            Self::Trusted
        } else {
            Self::Untrusted
        }
    }

    fn from_env_var(name: &str) -> Self {
        Self::from_env(std::env::var(name).ok().as_deref())
    }

    pub fn from_hooks_env() -> Self {
        Self::from_env_var(TRUST_PROJECT_HOOKS_ENV)
    }

    pub fn from_agents_env() -> Self {
        Self::from_env_var(TRUST_PROJECT_AGENTS_ENV)
    }

    pub fn from_plugins_env() -> Self {
        Self::from_env_var(TRUST_PROJECT_PLUGINS_ENV)
    }

    pub fn from_openapi_env() -> Self {
        Self::from_env_var(TRUST_PROJECT_OPENAPI_ENV)
    }

    pub const fn is_trusted(self) -> bool {
        matches!(self, Self::Trusted)
    }
}

pub fn project_ancestor_dirs(cwd: &Path) -> Vec<PathBuf> {
    let ancestors: Vec<_> = cwd.ancestors().map(Path::to_path_buf).collect();
    let Some(root_index) = ancestors.iter().position(|path| path.join(".git").exists()) else {
        return vec![cwd.to_path_buf()];
    };

    ancestors[..=root_index].iter().rev().cloned().collect()
}
