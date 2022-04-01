use std::fmt;

pub struct FmtAny<T>(pub T);

trait FmtAnyTrait {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl<T> FmtAnyTrait for FmtAny<T> {
    default fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<unknown>")
    }
}

impl<T> FmtAnyTrait for FmtAny<T>
where
    T: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T> FmtAnyTrait for &FmtAny<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl<T> fmt::Debug for FmtAny<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        FmtAnyTrait::fmt(self, f)
    }
}

impl<T> fmt::Display for FmtAny<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        FmtAnyTrait::fmt(self, f)
    }
}
