//! Where the inline shell's executable comes from.
//!
//! A bare name keeps `Command`'s PATH lookup except for `bash` on Windows,
//! where a missing PATH entry falls through to the bash Git for Windows puts
//! beside `git.exe`, then to the pinned busybox-w64 Rho keeps at
//! `$RHO_HOME/bin/bash.exe`, downloading it first when absent. BusyBox answers
//! to the `bash` applet name, so the single file doubles as every coreutil a
//! `!` command may call.
//!
//! The pin mirrors `antigravity_runtime::install`: URL, SHA-256, and exact
//! size per platform, recorded 2026-10-30 from frippery.org.

use std::{
    io,
    path::{Component, Path, PathBuf},
};

/// No bytes for this long means a dead transfer, not a slow one.
#[cfg(windows)]
const STALL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
#[cfg(windows)]
const CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// One platform's pinned busybox binary.
#[cfg(windows)]
#[derive(Clone, Copy)]
struct PinnedShell {
    url: &'static str,
    sha256: &'static str,
    /// Exact binary size; a longer or shorter body fails the download.
    bytes: u64,
}

/// The busybox-w64 release Rho installs as `bash`, if one exists for this CPU.
#[cfg(windows)]
fn pinned_bash() -> Option<PinnedShell> {
    Some(match std::env::consts::ARCH {
        "x86_64" => PinnedShell {
            url: "https://frippery.org/files/busybox/busybox-w64-FRP-6354-g5f09fb446.exe",
            sha256: "95343c871869cc0ab57ecec26b2f9af0370eb9c954f7e5ddd017163e4e0d09a0",
            bytes: 707_072,
        },
        "aarch64" => PinnedShell {
            url: "https://frippery.org/files/busybox/busybox-w64a-FRP-6354-g5f09fb446.exe",
            sha256: "ef426a4967f0a4657343242e85c290a8fb3272c6145c230439c587a0a97f9969",
            bytes: 655_872,
        },
        _ => return None,
    })
}

/// Whether `shell` can run without a PATH entry.
///
/// Windows `bash` is offered unconditionally: Git's bash usually exists even
/// when it is not on PATH, and the managed copy downloads on first use.
pub(super) fn provisionable(shell: &str) -> bool {
    cfg!(windows) && is_bare_bash(shell)
}

/// The executable `shell` names, provisioning a managed `bash` when needed.
///
/// Names with separators pass through untouched so configured paths fail (or
/// run) exactly as before.
pub(super) async fn resolve_executable(shell: &str) -> io::Result<PathBuf> {
    if !is_bare_name(shell) {
        return Ok(PathBuf::from(shell));
    }
    if let Some(path) = crate::executable::find_on_path(shell) {
        return Ok(path);
    }
    #[cfg(windows)]
    if is_bare_bash(shell) {
        return windows_bash().await;
    }
    Ok(PathBuf::from(shell))
}

/// A single bare file name, matching [`crate::executable::find_on_path`].
fn is_bare_name(shell: &str) -> bool {
    let mut components = Path::new(shell).components();
    matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none()
}

fn is_bare_bash(shell: &str) -> bool {
    is_bare_name(shell)
        && (shell.eq_ignore_ascii_case("bash") || shell.eq_ignore_ascii_case("bash.exe"))
}

#[cfg(windows)]
async fn windows_bash() -> io::Result<PathBuf> {
    if let Some(bash) = git_bash() {
        return Ok(bash);
    }
    let bash = managed_bash_path()?;
    if bash.is_file() {
        return Ok(bash);
    }
    download_bash(&bash).await?;
    Ok(bash)
}

/// Git for Windows keeps `usr/bin/bash.exe` next to `<root>/cmd/git.exe`,
/// even when `bash` itself is not on PATH.
#[cfg(windows)]
fn git_bash() -> Option<PathBuf> {
    let git = crate::executable::find_on_path("git")?;
    let root = git.parent()?.parent()?;
    let bash = root.join("usr").join("bin").join("bash.exe");
    bash.is_file().then_some(bash)
}

/// `$RHO_HOME/bin/bash.exe`, the Rho-managed shell.
#[cfg(windows)]
fn managed_bash_path() -> io::Result<PathBuf> {
    Ok(crate::paths::rho_dir()
        .map_err(io::Error::other)?
        .join("bin")
        .join("bash.exe"))
}

/// Best-effort `bash` for the model-facing shell tools: PATH, Git's `usr/bin`
/// copy, then an already-installed managed bash.
///
/// Unlike [`resolve_executable`] this never downloads. The bare `bash`
/// fallback means a host with no bash at all still gets a spawn error naming
/// `bash` rather than a config-time failure.
#[cfg(windows)]
pub(crate) fn model_bash_executable() -> PathBuf {
    if let Some(bash) = crate::executable::find_on_path("bash") {
        return bash;
    }
    if let Some(bash) = git_bash() {
        return bash;
    }
    managed_bash_path()
        .ok()
        .filter(|path| path.is_file())
        .unwrap_or_else(|| PathBuf::from("bash"))
}

/// Download the pinned busybox binary and publish it as `dest` in one rename.
/// A losing race with another inline shell leaves the winner's file, which is
/// exactly what both wanted.
#[cfg(windows)]
async fn download_bash(dest: &Path) -> io::Result<()> {
    use sha2::Digest as _;
    use std::io::Write as _;

    let pinned = pinned_bash().ok_or_else(|| {
        io::Error::other(format!(
            "no bash download for {}-{}; install Git for Windows or put bash on PATH",
            std::env::consts::OS,
            std::env::consts::ARCH
        ))
    })?;
    let client = crate::reqwest_client_builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(STALL_TIMEOUT)
        .build()
        .map_err(io::Error::other)?;
    let body = client
        .get(pinned.url)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(io::Error::other)?
        .bytes()
        .await
        .map_err(|error| io::Error::other(format!("could not download bash: {error}")))?;
    if body.len() as u64 != pinned.bytes {
        return Err(io::Error::other(format!(
            "bash download is {} bytes; the pinned release is {} bytes",
            body.len(),
            pinned.bytes
        )));
    }
    let digest = hex::encode(sha2::Sha256::digest(&body));
    if digest != pinned.sha256 {
        return Err(io::Error::other(format!(
            "bash download has SHA-256 {digest}, expected {}; nothing was installed",
            pinned.sha256
        )));
    }

    let dir = dest.parent().ok_or_else(|| {
        io::Error::other(format!(
            "managed bash path {} has no parent",
            crate::paths::display(dest)
        ))
    })?;
    std::fs::create_dir_all(dir)?;
    let mut temp = tempfile::NamedTempFile::new_in(dir)?;
    temp.write_all(&body)?;
    temp.flush()?;
    match temp.persist(dest) {
        Ok(_) => Ok(()),
        // Another inline shell published first; its verified file is fine.
        Err(_) if dest.is_file() => Ok(()),
        Err(error) => Err(io::Error::other(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_bash_names_provision() {
        assert!(is_bare_bash("bash"));
        assert!(is_bare_bash("bash.exe"));
        assert!(is_bare_bash("BASH"));
    }

    #[test]
    fn paths_and_other_shells_do_not_provision() {
        assert!(!is_bare_bash("sh"));
        assert!(!is_bare_bash("powershell"));
        assert!(!is_bare_bash("bin/bash"));
        assert!(!is_bare_bash("/usr/bin/bash"));
        assert!(!is_bare_bash("C:\\tools\\bash.exe"));
    }
}
