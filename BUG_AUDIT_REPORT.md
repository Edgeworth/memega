# Bug Audit Report for memega
**Date:** 2025-11-16
**Auditor:** Claude (Automated Code Analysis)
**Codebase Version:** Latest (commit 01c27ec)

## Executive Summary

This report documents a comprehensive security and correctness audit of the memega genetic algorithms library. The audit identified **14 bugs** ranging from critical runtime issues to minor edge cases.

### Critical Findings
- **3 Critical bugs** that can cause runtime panics or incorrect algorithm behavior
- **4 High severity bugs** affecting core functionality
- **4 Medium severity bugs** with edge case failures
- **3 Low severity issues** (theoretical or unlikely)

---

## Codebase Overview

**memega** is a Rust-based genetic algorithms library featuring:
- Linear genetic programming (LGP) with custom VM
- Hyperparameter evolution (meta-GA)
- Multiple crossover/mutation operators
- Speciation and niching support
- Python bindings via PyO3

**Statistics:**
- ~2,144 lines of core library code
- 47 Rust source files
- Rust 2024 edition (nightly features)
- Well-tested with comprehensive unit tests

---

## Critical Bugs (Priority 1)

### Bug #1: Inverted Mutation Rate Logic
**File:** `src/evaluators/lgp/eval.rs:110`
**Severity:** 🔴 **CRITICAL**

**Issue:**
```rust
fn mutate(&self, s: &mut LgpState, rate: f64, idx: usize) {
    let mut r = rand::rng();
    if r.random::<f64>() > rate {  // BUG: Should be <, not >
        return;
    }
```

**Impact:** Mutation rates work backwards - a rate of 0.9 gives 10% mutation chance instead of 90%. This fundamentally breaks the genetic algorithm's evolution mechanism for LGP.

**Fix:**
```rust
if r.random::<f64>() < rate {
    return;
}
```

---

### Bug #2: Array Bounds Panic in SUS
**File:** `src/ops/sampling.rs:62-64`
**Severity:** 🔴 **CRITICAL**

**Issue:**
```rust
for _ in 0..k {
    while cursum + w[idx] < cursor {  // No bounds check on idx
        cursum += w[idx];
        idx += 1;  // Can exceed w.len()
    }
```

**Impact:** Runtime panic during selection phase due to out-of-bounds access. Can occur due to floating-point precision errors.

**Fix:**
```rust
while idx < w.len() && cursum + w[idx] < cursor {
    cursum += w[idx];
    idx += 1;
}
```

---

### Bug #3: Division by Zero in multi_fitness
**File:** `src/eval.rs:53-54`
**Severity:** 🔴 **CRITICAL**

**Issue:**
```rust
let fitness = match reduction {
    FitnessReduction::ArithmeticMean => cumulative / inputs.len() as f64,
    FitnessReduction::GeometricMean => cumulative.powf(1.0 / inputs.len() as f64),
};
```

**Impact:** If `inputs` is empty, division by zero occurs (ArithmeticMean) or infinity is computed (GeometricMean), resulting in NaN/infinite fitness values.

**Fix:**
```rust
if inputs.is_empty() {
    return Err(eyre!("inputs cannot be empty"));
}
```

---

## High Severity Bugs (Priority 2)

### Bug #4: Unreachable Code in Random Distribution
**Files:**
- `src/evolve/cfg.rs:155-157` (Duplicates)
- `src/evolve/cfg.rs:173-175` (FitnessReduction)

**Severity:** 🟠 **HIGH**

**Issue:**
```rust
impl Distribution<Duplicates> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Duplicates {
        match r.random_range(0..1) {  // Only includes 0, never 1
            0 => Duplicates::DisallowDuplicates,
            _ => Duplicates::AllowDuplicates,  // UNREACHABLE
        }
    }
}
```

**Impact:**
- `Duplicates::AllowDuplicates` can never be randomly selected
- `FitnessReduction::GeometricMean` can never be randomly selected
- Hyperparameter evolution cannot explore these options

**Fix:** Change to `r.random_range(0..2)`

---

### Bug #5: Division by Zero in round_sf
**File:** `src/evaluators/lgp/cfg.rs:67-71`
**Severity:** 🟠 **HIGH**

**Issue:**
```rust
fn round_sf(v: f64, sf: usize) -> f64 {
    let digits = v.abs().log10().ceil() as i32;  // log10(0) = -infinity
    let power = 10f64.powi(digits - sf as i32);
    (v / power).round() * power
}
```

**Impact:** When `v` is 0.0, `log10(0)` returns `-∞`, causing invalid computation.

**Fix:**
```rust
if v == 0.0 {
    return 0.0;
}
```

---

### Bug #6: Tournament Selection Population Growth
**File:** `src/genr/evaluated.rs:78-88`
**Severity:** 🟠 **HIGH**

**Issue:**
```rust
Survival::Tournament(q) => {
    // ... tournament logic ...
    survivors.into_iter().map(|(_, mem)| mem.clone()).collect()
    // Missing: truncation to cfg.pop_size
}
```

**Impact:** Unlike other survival strategies, Tournament doesn't limit population size, causing unbounded growth.

**Fix:**
```rust
survivors.into_iter().take(cfg.pop_size).map(|(_, mem)| mem.clone()).collect()
```

---

### Bug #7: Missing Tournament Size Validation
**File:** `src/genr/evaluated.rs:82`
**Severity:** 🟠 **HIGH**

**Issue:**
```rust
let opponents = self.mems.choose_multiple(&mut rng, q);
```

**Impact:** If `q > self.mems.len()`, undefined behavior or panic.

**Fix:**
```rust
let opponents = self.mems.choose_multiple(&mut rng, q.min(self.mems.len()));
```

---

## Medium Severity Bugs (Priority 3)

### Bug #8: Incomplete RWS Sampling
**File:** `src/ops/sampling.rs:28-37`
**Severity:** 🟡 **MEDIUM**

**Issue:**
```rust
for _ in 0..k {
    let cursor = r.random_range(0.0..=sum);
    let mut cursum = 0.0;
    for (i, v) in w.iter().enumerate() {
        cursum += v;
        if cursum >= cursor {
            idxs.push(i);
            break;  // If never satisfied, nothing pushed
        }
    }
}
```

**Impact:** Floating-point errors can cause fewer samples than requested to be returned.

**Fix:** Push last index if loop completes without breaking.

---

### Bug #9: PMX Crossover Cycle Detection
**File:** `src/ops/crossover.rs:66-74`
**Severity:** 🟡 **MEDIUM**

**Issue:**
```rust
if count > s1.len() {
    ins = s2[i];  // Might already be in c1, creating duplicates
    break;
}
```

**Impact:** Invalid permutations with duplicate elements.

**Fix:** Implement proper cycle detection or mark visited elements.

---

### Bug #10: Unchecked LGP VM Memory Access
**File:** `src/evaluators/lgp/vm/lgpvm.rs:38-42`
**Severity:** 🟡 **MEDIUM**

**Issue:**
```rust
pub fn mem(&self, idx: u8) -> f64 {
    self.mem[idx as usize]  // No bounds check
}
```

**Impact:** Panic when executing malformed LGP programs.

**Fix:** Add bounds checking or use `.get()` with error handling.

---

### Bug #11: Crossover Order Uninitialized Elements
**File:** `src/ops/crossover.rs:136-144`
**Severity:** 🟡 **MEDIUM**

**Issue:** Fallback logic may leave `Default::default()` values in output.

**Impact:** Invalid permutations when input has duplicates.

**Fix:** Ensure all positions are filled or validate inputs.

---

## Low Severity Issues (Priority 4)

### Bug #12: Float Negation in Assertion
**File:** `src/genr/species.rs:92`
**Severity:** 🟢 **LOW**

**Issue:**
```rust
assert!(s.is_sorted_by_key(|v| -v.fitness), "Must be sorted by fitness (bug)");
```

**Impact:** Assertion may fail incorrectly for edge case float values.

**Fix:** Use `is_sorted_by` with proper comparator.

---

### Bug #13: Theoretical Species Overflow
**File:** `src/genr/species.rs:113`
**Severity:** 🟢 **LOW**

**Issue:** Species ID counter could theoretically overflow (u64).

**Impact:** None in practice (would require 2^64 species).

**Fix:** None needed.

---

### Bug #14: Missing Parameter Validation
**File:** Various configuration builders
**Severity:** 🟢 **LOW**

**Issue:** No validation that configuration parameters are sensible.

**Impact:** Unexpected behavior with extreme values.

**Fix:** Add validation in builder methods.

---

## Testing Recommendations

1. **Add edge case tests:**
   - Empty input arrays
   - Zero values in numeric operations
   - Extreme configuration values
   - Floating-point precision edge cases

2. **Add property-based testing:**
   - Crossover operations preserve permutation validity
   - Population size constraints are maintained
   - Mutation rates behave correctly

3. **Add fuzzing:**
   - Random operator sequences
   - Random configuration combinations
   - Invalid LGP program generation

4. **Integration tests:**
   - End-to-end GA runs with various configurations
   - Tournament selection with different sizes
   - Speciation with different radii

---

## Conclusion

The memega codebase is well-structured and demonstrates good software engineering practices. However, the identified bugs (particularly #1-3) require immediate attention as they can cause:

1. **Incorrect algorithm behavior** (inverted mutation rates)
2. **Runtime panics** (array bounds, division by zero)
3. **Population size issues** (tournament selection)
4. **Missing functionality** (unreachable enum variants)

### Recommended Action Plan

**Week 1 (Critical):**
- Fix bugs #1-3 (mutation rate, SUS bounds, division by zero)
- Add tests to prevent regression

**Week 2 (High Priority):**
- Fix bugs #4-7 (random ranges, round_sf, tournament selection)
- Add validation for configuration parameters

**Week 3 (Medium Priority):**
- Fix bugs #8-11 (RWS, PMX, VM bounds, crossover)
- Improve error handling

**Ongoing:**
- Expand test coverage
- Add property-based testing
- Consider fuzzing for operator correctness

---

**Report End**
