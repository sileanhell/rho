use std::{io, os::windows::ffi::OsStrExt, path::Path, ptr};

use windows_sys::Win32::{
    Foundation::{CloseHandle, LocalFree, ERROR_SUCCESS, HANDLE},
    Security::{
        Authorization::{
            BuildTrusteeWithSidW, SetEntriesInAclW, SetNamedSecurityInfoW, EXPLICIT_ACCESS_W,
            SET_ACCESS, SE_FILE_OBJECT,
        },
        DACL_SECURITY_INFORMATION, GetLengthSid, GetTokenInformation, NO_INHERITANCE, TokenUser,
        PROTECTED_DACL_SECURITY_INFORMATION, SUB_CONTAINERS_AND_OBJECTS_INHERIT, TOKEN_QUERY,
        TOKEN_USER,
    },
    Storage::FileSystem::{
        MoveFileExW, FILE_ALL_ACCESS, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    },
    System::Threading::{GetCurrentProcess, OpenProcessToken},
};

use super::{CredentialError, CredentialResult};

/// Atomically replace `destination` with `source`, overwriting when needed.
pub(super) fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    let result = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if result == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Returns the current user's SID from the process token. The trustee is
/// granted by SID, not by name lookup: an unqualified user name is ambiguous
/// when it matches the machine or another authority and can resolve to a
/// trustee that is not the user.
fn current_user_sid() -> CredentialResult<Vec<u8>> {
    unsafe {
        let mut token: HANDLE = ptr::null_mut();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
            return Err(CredentialError::StoreUnavailable(format!(
                "could not open process token: {}",
                io::Error::last_os_error()
            )));
        }
        let sid = read_token_user_sid(token);
        CloseHandle(token);
        sid
    }
}

unsafe fn read_token_user_sid(token: HANDLE) -> CredentialResult<Vec<u8>> {
    let mut len = 0u32;
    unsafe { GetTokenInformation(token, TokenUser, ptr::null_mut(), 0, &mut len) };
    if len == 0 {
        return Err(CredentialError::StoreUnavailable(format!(
            "could not query process token user: {}",
            io::Error::last_os_error()
        )));
    }
    let mut buffer = vec![0u8; len as usize];
    if unsafe {
        GetTokenInformation(
            token,
            TokenUser,
            buffer.as_mut_ptr().cast(),
            len,
            &mut len,
        )
    } == 0
    {
        return Err(CredentialError::StoreUnavailable(format!(
            "could not query process token user: {}",
            io::Error::last_os_error()
        )));
    }
    let sid = unsafe { (*buffer.as_ptr().cast::<TOKEN_USER>()).User.Sid };
    let sid_len = unsafe { GetLengthSid(sid) } as usize;
    if sid_len == 0 || sid_len > buffer.len() {
        return Err(CredentialError::StoreUnavailable(
            "could not read process token user SID".into(),
        ));
    }
    Ok(unsafe { std::slice::from_raw_parts(sid.cast::<u8>(), sid_len) }.to_vec())
}

pub(super) fn set_private_windows_acl(path: &Path, directory: bool) -> CredentialResult<()> {
    let sid = current_user_sid()?;
    let mut access = EXPLICIT_ACCESS_W {
        grfAccessPermissions: FILE_ALL_ACCESS,
        grfAccessMode: SET_ACCESS,
        grfInheritance: if directory {
            SUB_CONTAINERS_AND_OBJECTS_INHERIT
        } else {
            NO_INHERITANCE
        },
        ..Default::default()
    };
    unsafe { BuildTrusteeWithSidW(&mut access.Trustee, sid.as_ptr() as *mut _) };
    let mut acl = ptr::null_mut();
    let result = unsafe { SetEntriesInAclW(1, &access, ptr::null(), &mut acl) };
    if result != ERROR_SUCCESS {
        return Err(CredentialError::StoreUnavailable(format!(
            "could not build private ACL for {}: Windows error {result}",
            path.display()
        )));
    }

    let wide_path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let result = unsafe {
        SetNamedSecurityInfoW(
            wide_path.as_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            ptr::null_mut(),
            ptr::null_mut(),
            acl,
            ptr::null(),
        )
    };
    unsafe { LocalFree(acl.cast()) };
    if result != ERROR_SUCCESS {
        return Err(CredentialError::StoreUnavailable(format!(
            "could not protect credential path {}: Windows error {result}",
            path.display()
        )));
    }
    Ok(())
}
