//! Windows console isolation for side processes.
//!
//! Internal cross-crate support for Rho, not a supported downstream API.

/// Spawn `command` without attaching to the host console or showing a window.
///
/// A console child inherits its parent's console by default, so a spawned
/// `powershell.exe` or `cmd.exe` can reconfigure the host's own console (font,
/// modes, title) while it runs. `CREATE_NO_WINDOW` runs the child with no
/// console at all; piped stdio is unaffected and GUI children still show their
/// own windows.
///
/// Use for every child the host runs in the background. Interactive
/// foreground children (editors, sign-in flows that inherit stdio) keep the
/// console instead.
pub fn hide_console_window(command: &mut std::process::Command) {
    use std::os::windows::process::CommandExt;
    command.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
}
