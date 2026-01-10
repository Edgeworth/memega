use std::fmt;

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

/// Simple string-based error for use with `Error::msg`.
#[derive(Debug)]
struct StringError(String);

impl fmt::Display for StringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StringError {}

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    EvolverError(#[from] Box<dyn std::error::Error + Send + Sync>),

    #[error("length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    #[error("invalid LGP: {0}")]
    InvalidLgp(String),

    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    #[error(transparent)]
    ParseFloat(#[from] std::num::ParseFloatError),
}

impl Error {
    /// Create an evolver error from any error type.
    pub fn evolver<E: std::error::Error + Send + Sync + 'static>(err: E) -> Self {
        Self::EvolverError(Box::new(err))
    }

    /// Create an evolver error from a string message.
    pub fn evolver_msg(msg: impl Into<String>) -> Self {
        Self::EvolverError(Box::new(StringError(msg.into())))
    }
}
