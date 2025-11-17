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

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_rand_vec() {
        let mut counter = 0;
        let result = rand_vec(5, || {
            counter += 1;
            counter
        });
        assert_eq!(result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_rand_vec_empty() {
        let result = rand_vec(0, || 42);
        assert_eq!(result, Vec::<i32>::new());
    }

    #[test]
    fn test_vec_to_str() {
        assert_eq!(vec_to_str(&['h', 'e', 'l', 'l', 'o']), "hello");
        assert_eq!(vec_to_str(&[]), "");
    }

    #[test]
    fn test_str_to_vec() {
        assert_eq!(str_to_vec("hello"), vec!['h', 'e', 'l', 'l', 'o']);
        assert_eq!(str_to_vec(""), Vec::<char>::new());
    }

    #[test]
    fn test_clamp_vec_both_bounds() {
        let mut v = vec![1.0, 5.0, 10.0, 15.0, 20.0];
        clamp_vec(&mut v, Some(5.0), Some(15.0));
        assert_eq!(v, vec![5.0, 5.0, 10.0, 15.0, 15.0]);
    }

    #[test]
    fn test_clamp_vec_lower_only() {
        let mut v = vec![1.0, 5.0, 10.0];
        clamp_vec(&mut v, Some(5.0), None);
        assert_eq!(v, vec![5.0, 5.0, 10.0]);
    }

    #[test]
    fn test_clamp_vec_upper_only() {
        let mut v = vec![1.0, 5.0, 10.0];
        clamp_vec(&mut v, None, Some(5.0));
        assert_eq!(v, vec![1.0, 5.0, 5.0]);
    }

    #[test]
    fn test_clamp_vec_no_bounds() {
        let mut v = vec![1.0, 5.0, 10.0];
        let original = v.clone();
        clamp_vec(&mut v, None, None);
        assert_eq!(v, original);
    }

    #[test]
    fn test_clamp_vec_empty() {
        let mut v: Vec<f64> = vec![];
        clamp_vec(&mut v, Some(0.0), Some(10.0));
        assert_eq!(v, Vec::<f64>::new());
    }
}
