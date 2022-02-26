pub fn rand_vec<T>(k: usize, mut f: impl FnMut() -> T) -> Vec<T> {
    (0..k).map(|_| f()).collect()
}

#[must_use]
pub fn vec_to_str(input: &[char]) -> String {
    input.iter().collect()
}

#[must_use]
pub fn str_to_vec(input: &str) -> Vec<char> {
    input.chars().collect()
}

pub fn clamp_vec(v: &mut [f64], lo: Option<f64>, hi: Option<f64>) {
    for k in v.iter_mut() {
        if let Some(lo) = lo {
            *k = k.max(lo);
        }
        if let Some(hi) = hi {
            *k = k.min(hi);
        }
    }
}
