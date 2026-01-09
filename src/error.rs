#[derive(thiserror::Error, Debug)]
pub enum LaError {
    #[error("No linear algebra library was found")]
    NoLaLibrary,
    #[cfg(feature = "dynamic")]
    #[error(transparent)]
    LibLoading(#[from] libloading::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
