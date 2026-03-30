#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Float Parse Error: {0}")]
    ParseFloatError(#[from] std::num::ParseFloatError),

    #[error("Int Parse Error: {0}")]
    ParseIntError(#[from] std::num::ParseIntError),

    #[error("Enum Parse Error: {0}")]
    ParseError(#[from] strum::ParseError),

    #[error("Format Error: {0}")]
    Format(String),

    #[error("Size mismatch: expected {expected}, got {got}")]
    SizeMismatch{expected: usize, got: usize},

    #[error("Generic: {0}")]
    Generic(String)
}
