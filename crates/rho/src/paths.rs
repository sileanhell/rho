use std::{
    ffi::OsString,
    fs,
    path::{Component, Path, PathBuf},
};

/// Renders paths consistently in user-facing text and structured output.
pub(crate) fn display(path: &Path) -> String {
    let rendered = path.to_string_lossy();
    #[cfg(windows)]
    {
        rendered.replace('\\', "/")
    }
    #[cfg(not(windows))]
    {
        rendered.into_owned()
    }
}

/// Encodes a path for injection into prompt assembly as path data.
///
/// Always a JSON string (quoted, control-escaped) so the value occupies one
/// structural token and cannot introduce additional prompt lines. Built on
/// [`display`] so Windows separators stay normalized before escaping.
pub(crate) fn prompt_data(path: &Path) -> String {
    serde_json::to_string(&display(path)).expect("path display is a string")
}

/// Encodes a path as a double-quoted XML-like attribute value for prompt tags.
///
/// Markup specials and control characters are entity-escaped so the path stays
/// inert attribute data and cannot change surrounding tag structure. Built on
/// [`display`] for the same separator normalization as other path rendering.
pub(crate) fn prompt_attr(path: &Path) -> String {
    let mut out = String::from('"');
    for ch in display(path).chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            ch if ch.is_control() => {
                use std::fmt::Write as _;
                let _ = write!(out, "&#x{:X};", u32::from(ch));
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
    out
}

/// Returns the user's home directory using platform-appropriate environment variables.
pub(crate) fn home_dir() -> Option<PathBuf> {
    home_dir_from_env(|name| std::env::var_os(name))
}

fn home_dir_from_env(mut var: impl FnMut(&str) -> Option<OsString>) -> Option<PathBuf> {
    if let Some(home) = var("HOME").filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(home));
    }

    #[cfg(windows)]
    {
        if let Some(profile) = var("USERPROFILE").filter(|value| !value.is_empty()) {
            return Some(PathBuf::from(profile));
        }

        if let (Some(drive), Some(path)) = (
            var("HOMEDRIVE").filter(|value| !value.is_empty()),
            var("HOMEPATH").filter(|value| !value.is_empty()),
        ) {
            let mut home = PathBuf::from(drive);
            home.push(path);
            return Some(home);
        }
    }

    None
}

pub(crate) fn rho_dir() -> anyhow::Result<PathBuf> {
    rho_dir_from_env(|name| std::env::var_os(name))
}

/// Global instructions loaded for every session: `~/.rho/AGENTS.md`.
pub(crate) fn user_agents_md(home: &Path) -> PathBuf {
    home.join(".rho").join("AGENTS.md")
}

/// Loose user skill trees: `~/.rho/skills`, then `~/.agents/skills`.
pub(crate) fn user_skill_dirs(home: &Path) -> [PathBuf; 2] {
    [
        home.join(".rho").join("skills"),
        home.join(".agents").join("skills"),
    ]
}

/// User agent definition trees: `~/.agents/agents`, then `~/.rho/agents`.
pub(crate) fn user_agent_dirs(home: &Path) -> [PathBuf; 2] {
    [
        home.join(".agents").join("agents"),
        home.join(".rho").join("agents"),
    ]
}

/// Durable workflow plans and runs: `$RHO_HOME/workflows` or `~/.rho/workflows`.
pub(crate) fn user_workflows_dir(rho_home: &Path) -> PathBuf {
    rho_home.join("workflows")
}

/// Rho-managed external runtimes (e.g. the Antigravity ACP server):
/// `$RHO_HOME/runtimes` or `~/.rho/runtimes`.
pub(crate) fn user_runtimes_dir(rho_home: &Path) -> PathBuf {
    rho_home.join("runtimes")
}

/// Host config the workflow tool reads: `$RHO_HOME/config.toml` or `~/.rho/config.toml`.
pub(crate) fn user_config_toml(rho_home: &Path) -> PathBuf {
    rho_home.join("config.toml")
}

/// User-owned instruction surfaces Rho already loads from `$HOME`.
///
/// Built from [`user_agents_md`], [`user_skill_dirs`], and [`user_agent_dirs`].
/// Credentials, config, hooks, sessions, and plugins stay out of this set.
///
/// Membership is a construction-time snapshot: lexical paths always match, and
/// a resolved path is kept only when the entry exists, is not a symlink, and
/// stays under `$HOME`. Authorize-path checks do not touch the filesystem.
#[derive(Clone, Debug, Default)]
pub(crate) struct UserInstructionSurfaces {
    files: Vec<AnchoredPath>,
    directories: Vec<AnchoredPath>,
}

#[derive(Clone, Debug)]
struct AnchoredPath {
    lexical: PathBuf,
    resolved: Option<PathBuf>,
}

impl UserInstructionSurfaces {
    pub(crate) fn from_process() -> Self {
        Self::from_home(home_dir().as_deref())
    }

    pub(crate) fn from_home(home: Option<&Path>) -> Self {
        let Some(home) = home else {
            return Self::default();
        };
        let mut directories: Vec<_> = user_skill_dirs(home)
            .into_iter()
            .map(|path| snapshot_path(path, home))
            .collect();
        directories.extend(
            user_agent_dirs(home)
                .into_iter()
                .map(|path| snapshot_path(path, home)),
        );
        Self {
            files: vec![snapshot_path(user_agents_md(home), home)],
            directories,
        }
    }

    pub(crate) fn contains(&self, path: &Path) -> bool {
        let path = lexical_normalize(path);
        self.files
            .iter()
            .any(|allowed| path_matches_file(&path, allowed))
            || self
                .directories
                .iter()
                .any(|root| path_is_under_anchored(&path, root))
    }
}

fn snapshot_path(lexical: PathBuf, home: &Path) -> AnchoredPath {
    let resolved = resolved_if_anchored(&lexical, home);
    AnchoredPath { lexical, resolved }
}

/// Keep a resolved form only when the entry is a real file or directory under
/// `$HOME`. A symlink root (or a resolved path that escaped home) is dropped
/// so canonicalize cannot widen the allowlist to `/` or `$HOME`.
fn resolved_if_anchored(lexical: &Path, home: &Path) -> Option<PathBuf> {
    let metadata = fs::symlink_metadata(lexical).ok()?;
    if metadata.file_type().is_symlink() {
        return None;
    }
    let resolved = lexical.canonicalize().ok()?;
    let home_resolved = home.canonicalize().unwrap_or_else(|_| home.to_path_buf());
    (resolved.starts_with(home) || resolved.starts_with(&home_resolved)).then_some(resolved)
}

fn path_matches_file(path: &Path, allowed: &AnchoredPath) -> bool {
    path == allowed.lexical.as_path() || allowed.resolved.as_deref() == Some(path)
}

fn path_is_under_anchored(path: &Path, root: &AnchoredPath) -> bool {
    path.starts_with(&root.lexical)
        || root
            .resolved
            .as_ref()
            .is_some_and(|resolved| path.starts_with(resolved))
}

/// True when `path` is the live canonical target of `{root}/{file_name}`.
///
/// PATH binaries often live outside the PATH directory (`/usr/bin/git` →
/// `/usr/lib/git-core/git`). Construction only snapshots the directory, so this
/// check is the authorize-path follow-up for those resolved identities.
fn path_is_resolved_dir_child(path: &Path, root: &AnchoredPath) -> bool {
    let Some(name) = path.file_name() else {
        return false;
    };
    // Compare canonical forms so macOS `/var` vs `/private/var` still matches.
    let query = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let resolves_to_path =
        |dir: &Path| fs::canonicalize(dir.join(name)).is_ok_and(|resolved| resolved == query);
    resolves_to_path(&root.lexical)
        || root.resolved.as_deref().is_some_and(|resolved| {
            resolved != root.lexical.as_path() && resolves_to_path(resolved)
        })
}

/// Host-owned surfaces the built-in `workflow` tool may read without the
/// agent-facing outside-workspace gate: the Rho workflow tree, the default
/// host config, Rho-managed runtimes, and directories on `PATH` at
/// construction.
///
/// Graph-supplied absolute paths outside this set still follow the normal
/// gate. The model's `read_file` of the same paths is not exempt. A PATH
/// binary whose canonical target left its PATH directory still matches via
/// a live `{dir}/{file_name}` canonicalize.
#[derive(Clone, Debug, Default)]
pub(crate) struct HostOwnedSurfaces {
    files: Vec<AnchoredPath>,
    directories: Vec<AnchoredPath>,
}

impl HostOwnedSurfaces {
    pub(crate) fn from_process() -> Self {
        Self::from_env(rho_dir().ok().as_deref(), std::env::var_os("PATH"))
    }

    pub(crate) fn from_env(rho_home: Option<&Path>, path_var: Option<OsString>) -> Self {
        let mut files = Vec::new();
        let mut directories = Vec::new();
        if let Some(rho_home) = rho_home {
            files.push(snapshot_path(user_config_toml(rho_home), rho_home));
            directories.push(snapshot_path(user_workflows_dir(rho_home), rho_home));
            // Workflows freeze a managed runtime server as an absolute path.
            directories.push(snapshot_path(user_runtimes_dir(rho_home), rho_home));
        }
        if let Some(path_var) = path_var {
            for directory in std::env::split_paths(&path_var) {
                if directory.as_os_str().is_empty() {
                    continue;
                }
                directories.push(snapshot_path(directory.clone(), &directory));
            }
        }
        Self { files, directories }
    }

    pub(crate) fn contains(&self, path: &Path) -> bool {
        let path = lexical_normalize(path);
        self.files
            .iter()
            .any(|allowed| path_matches_file(&path, allowed))
            || self.directories.iter().any(|root| {
                path_is_under_anchored(&path, root) || path_is_resolved_dir_child(&path, root)
            })
    }
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            component => out.push(component),
        }
    }
    out
}

fn rho_dir_from_env(mut var: impl FnMut(&str) -> Option<OsString>) -> anyhow::Result<PathBuf> {
    if let Some(root) = var("RHO_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(root));
    }
    home_dir_from_env(var)
        .map(|home| home.join(".rho"))
        .ok_or_else(|| anyhow::anyhow!("could not determine home directory"))
}

pub(crate) fn usage_database_path() -> anyhow::Result<PathBuf> {
    Ok(rho_dir()?.join("usage.sqlite3"))
}

pub(crate) fn prompt_history_database_path() -> anyhow::Result<PathBuf> {
    Ok(rho_dir()?.join("prompt-history.sqlite3"))
}

/// Process-wide lock for tests that read or mutate `RHO_HOME` / related env.
///
/// Hold this for the entire critical section. Concurrent tests that only set the
/// variable briefly still race with readers that use `rho_dir()` afterward.
#[cfg(test)]
pub(crate) fn process_env_lock() -> std::sync::MutexGuard<'static, ()> {
    use std::sync::{Mutex, OnceLock};
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}
