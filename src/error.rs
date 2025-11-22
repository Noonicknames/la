#[derive(thiserror::Error, Debug)]
pub enum LaError {
    #[error("No linear algebra library was found")]
    NoLaLibrary,
    #[error(transparent)]
    LibLoading(#[from] libloading::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("{} errors", self)]
    Multi(Vec<LaError>),
}

impl FromIterator<LaError> for LaError {
    fn from_iter<T: IntoIterator<Item = LaError>>(iter: T) -> Self {
        Self::Multi(Vec::from_iter(iter))
    }
}

impl LaError {

}
