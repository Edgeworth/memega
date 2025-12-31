use rand::Rng;

fn uniform_with_replacement<R: Rng + ?Sized>(len: usize, k: usize, r: &mut R) -> Vec<usize> {
    if len == 0 {
        return vec![];
    }
    (0..k).map(|_| r.random_range(0..len)).collect()
}

// Roulette wheel selection:
#[must_use]
pub fn rws(w: &[f64]) -> Option<usize> {
    multi_rws(w, 1).first().copied()
}

pub fn rws_rng<R: Rng + ?Sized>(w: &[f64], r: &mut R) -> Option<usize> {
    multi_rws_rng(w, 1, r).first().copied()
}

#[must_use]
pub fn multi_rws(w: &[f64], k: usize) -> Vec<usize> {
    let mut r = rand::rng();
    multi_rws_rng(w, k, &mut r)
}

pub fn multi_rws_rng<R: Rng + ?Sized>(w: &[f64], k: usize, r: &mut R) -> Vec<usize> {
    if k == 0 || w.is_empty() {
        return vec![];
    }

    assert!(w.iter().all(|&v| v.is_finite() && v >= 0.0), "must be non-negative and finite");
    let sum = w.iter().sum::<f64>();
    if sum == 0.0 {
        return uniform_with_replacement(w.len(), k, r);
    }

    let mut idxs = Vec::with_capacity(k);
    for _ in 0..k {
        let cursor = r.random_range(0.0..=sum);
        let mut cursum = 0.0;
        let mut picked = w.len() - 1;
        for (i, v) in w.iter().enumerate() {
            cursum += v;
            if cursum >= cursor {
                picked = i;
                break;
            }
        }
        idxs.push(picked);
    }
    idxs
}

// Stochastic universal sampling:
#[must_use]
pub fn sus(w: &[f64], k: usize) -> Vec<usize> {
    let mut r = rand::rng();
    sus_rng(w, k, &mut r)
}

pub fn sus_rng<R: Rng + ?Sized>(w: &[f64], k: usize, r: &mut R) -> Vec<usize> {
    if k == 0 || w.is_empty() {
        return vec![];
    }

    assert!(w.iter().all(|&v| v.is_finite() && v >= 0.0), "must be non-negative and finite");
    let sum = w.iter().sum::<f64>();
    if sum == 0.0 {
        return uniform_with_replacement(w.len(), k, r);
    }

    let step = sum / k as f64;
    let mut idxs = Vec::with_capacity(k);
    let mut idx = 0;
    let mut cursum = 0.0;
    let mut cursor = r.random_range(0.0..=step);
    for _ in 0..k {
        while idx + 1 < w.len() && cursum + w[idx] < cursor {
            cursum += w[idx];
            idx += 1;
        }
        idxs.push(idx);
        cursor += step;
    }
    idxs
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    #[test]
    fn rws_basic() {
        let mut r = StdRng::seed_from_u64(0);
        assert_eq!(rws_rng(&[], &mut r), None);
        assert_eq!(rws_rng(&[1.0], &mut r), Some(0));
        assert_eq!(rws_rng(&[0.0, 1.0], &mut r), Some(1));
    }

    #[test]
    fn multi_rws_basic() {
        let mut r = StdRng::seed_from_u64(0);
        assert_eq!(multi_rws_rng(&[], 0, &mut r), []);
        assert_eq!(multi_rws_rng(&[], 1, &mut r), []);
        assert_eq!(multi_rws_rng(&[1.0], 0, &mut r), []);
        assert_eq!(multi_rws_rng(&[1.0], 1, &mut r), [0]);
        assert_eq!(multi_rws_rng(&[0.0], 2, &mut r), [0, 0]);
        assert_eq!(multi_rws_rng(&[0.0, 1.0], 1, &mut r), [1]);
    }

    #[test]
    fn sus_basic() {
        let mut r = StdRng::seed_from_u64(0);
        assert_eq!(sus_rng(&[], 0, &mut r), []);
        assert_eq!(sus_rng(&[], 1, &mut r), []);
        assert_eq!(sus_rng(&[1.0], 0, &mut r), []);
        assert_eq!(sus_rng(&[1.0], 1, &mut r), [0]);
        assert_eq!(sus_rng(&[0.0], 2, &mut r), [0, 0]);
        assert!(sus_rng(&[1.0, 1.0], 1, &mut r).into_iter().all(|idx| idx < 2));
        assert!(sus_rng(&[0.0, 1.0], 1, &mut r).into_iter().all(|idx| idx < 2));
        assert!(sus_rng(&[1.0, 1.0], 2, &mut r).into_iter().all(|idx| idx < 2));
        assert!(sus_rng(&[1.0, 2.0], 3, &mut r).into_iter().all(|idx| idx < 2));
    }
}
