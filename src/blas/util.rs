use std::path::{Path, PathBuf};

use crate::error::LaError;

#[cfg(target_os = "windows")]
pub const LIB_SUFFIXES: &[&str] = &[".dll"];
#[cfg(target_os = "linux")]
pub const LIB_SUFFIXES: &[&str] = &[".so"];
#[cfg(target_os = "macos")]
pub const LIB_SUFFIXES: &[&str] = &[".dylib"];

#[cfg(target_os = "windows")]
pub const SEARCH_PATHS: &[&str] = &[];
#[cfg(target_os = "linux")]
pub const SEARCH_PATHS: &[&str] = &["/lib", "/usr/lib"];
#[cfg(target_os = "macos")]
pub const SEARCH_PATHS: &[&str] = &[];

#[cfg(target_os = "windows")]
pub const SEARCH_PATH_VARS: &[&str] = &["MKLROOT", "LIB", "INCLUDE", "PATH"];
#[cfg(target_os = "linux")]
pub const SEARCH_PATH_VARS: &[&str] = &[
    "MKLROOT",
    "LD_LIBRARY_PATH",
    "CPATH",
    "LIBRARY_PATH",
    "PATH",
];
#[cfg(target_os = "macos")]
pub const SEARCH_PATH_VARS: &[&str] = &[
    "MKLROOT",
    "DYLD_LIBRARY_PATH",
    "CPATH",
    "LIBRARY_PATH",
    "PATH",
];

// I could rework this to not require allocation.
pub fn get_default_search_paths() -> Vec<PathBuf> {
    let mut search_paths = Vec::new();

    for path in SEARCH_PATHS.iter() {
        search_paths.push(path.into())
    }

    for var in SEARCH_PATH_VARS.iter() {
        if let Some(var) = std::env::var_os(var) {
            for path in std::env::split_paths(&var) {
                search_paths.push(path);
            }
        }
    }

    search_paths
}

pub fn find_lib_path_with_dir(
    lib_name: impl AsRef<str>,
    dir: impl AsRef<Path>,
) -> Result<PathBuf, LaError> {
    let lib_name = lib_name.as_ref();
    for entry in dir.as_ref().read_dir()?.flatten() {
        let path = entry.path();
        if let Some(file_name) = path.file_name().and_then(|file_name| file_name.to_str()) {
            if !file_name.trim_start_matches("lib").starts_with(lib_name) {
                continue;
            }
            if LIB_SUFFIXES
                .iter()
                .find(|suffix| file_name.ends_with(*suffix))
                .is_none()
            {
                continue;
            }
            return Ok(path);
        }
    }

    Err(LaError::NoLaLibrary)
}

pub fn find_lib_path(lib_name: impl AsRef<str>) -> Result<PathBuf, LaError> {
    find_lib_path_with_search_paths(lib_name, get_default_search_paths().iter().as_ref())
}

pub fn find_lib_path_with_additional_search_paths<P>(
    lib_name: impl AsRef<str>,
    search_paths: impl IntoIterator<Item = P>,
) -> Result<PathBuf, LaError>
where
    P: AsRef<Path>,
{
    if let Ok(path) = find_lib_path_with_search_paths(lib_name.as_ref(), search_paths) {
        return Ok(path);
    }
    if let Ok(path) = find_lib_path(lib_name) {
        return Ok(path);
    }

    Err(LaError::NoLaLibrary)
}

pub fn find_lib_path_with_search_paths<P>(
    lib_name: impl AsRef<str>,
    search_paths: impl IntoIterator<Item = P>,
) -> Result<PathBuf, LaError>
where
    P: AsRef<Path>,
{
    let lib_name = lib_name.as_ref();

    for dir in search_paths {
        if let Ok(path) = find_lib_path_with_dir(lib_name, dir) {
            return Ok(path);
        }
    }
    Err(LaError::NoLaLibrary)
}
