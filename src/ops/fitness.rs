// Calculating fitnesses:
pub fn count_different<T: PartialEq>(s1: &[T], s2: &[T]) -> usize {
    let min = s1.len().min(s2.len());
    let max = s1.len().max(s2.len());
    let mut count = 0;
    for i in 0..min {
        if s1[i] != s2[i] {
            count += 1;
        }
    }
    count + max - min
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_different() {
        assert_eq!(count_different(&[1], &[1]), 0);
        assert_eq!(count_different(&[1], &[2]), 1);
        assert_eq!(count_different(&[1], &[1, 2]), 1);
        assert_eq!(count_different(&[1, 2], &[1]), 1);
    }
}
