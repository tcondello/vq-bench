//! CAST_SHELL_E8: projects 8D blocks onto the 240 minimal root vectors of the E8 Gosset lattice
//! -
//! Model: in_dim (usize)
//! Code for vector x: u8 per 8-dim subvector (1 bit/dim)
//! Apply: x --> x - hat{x}
//! Reconstruct: code --> hat{x} (unit-norm root vectors on S^7)
//! Score: <q, hat{x}> (+ child_score)

use ndarray::{Array2, ArrayView2};

use crate::coding;
use crate::{math, Primitive};

/// The 240 minimal root vectors of the E8 Gosset lattice in R^8.
/// Every root vector has norm sqrt(2).
pub struct CastShellE8;

impl Default for CastShellE8 {
    fn default() -> Self {
        Self::new()
    }
}

impl CastShellE8 {
    pub fn new() -> Self {
        Self
    }
}

/// Normalization constant to scale norm-sqrt(2) roots to unit norm on S^7.
const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Precomputed 240 minimal root vectors of E8, normalized to unit norm on S^7.
static ROOT_TABLE: std::sync::LazyLock<[[f32; 8]; 240]> = std::sync::LazyLock::new(|| {
    let mut table = [[0.0f32; 8]; 240];
    let mut idx = 0;

    // Type 1: 112 vectors of shape (+-1, +-1, 0, 0, 0, 0, 0, 0)
    for i in 0..8 {
        for j in (i + 1)..8 {
            for &s_i in &[1.0f32, -1.0f32] {
                for &s_j in &[1.0f32, -1.0f32] {
                    let mut v = [0.0f32; 8];
                    v[i] = s_i * INV_SQRT2;
                    v[j] = s_j * INV_SQRT2;
                    table[idx] = v;
                    idx += 1;
                }
            }
        }
    }
    assert_eq!(idx, 112);

    // Type 2: 128 vectors of shape (+-1/2, ..., +-1/2) with even number of minus signs
    for mask in 0u8..=255 {
        let neg_count = mask.count_ones();
        if neg_count % 2 == 0 {
            let mut v = [0.0f32; 8];
            for (bit, elem) in v.iter_mut().enumerate() {
                let sign = if (mask & (1 << bit)) != 0 { -1.0f32 } else { 1.0f32 };
                *elem = 0.5 * sign * INV_SQRT2;
            }
            table[idx] = v;
            idx += 1;
        }
    }
    assert_eq!(idx, 240);
    table
});

/// Find the nearest of the 240 minimal E8 root vectors to an 8D input vector x in O(1) time.
#[inline]
fn find_nearest_root(x: &[f32; 8]) -> u8 {
    let roots = &*ROOT_TABLE;
    let mut best_idx = 0u8;
    let mut max_dot = f32::NEG_INFINITY;

    // O(1) fast direct search over the compact 240-element table
    for (i, root) in roots.iter().enumerate() {
        let mut dot = 0.0f32;
        for k in 0..8 {
            dot += x[k] * root[k];
        }
        if dot > max_dot {
            max_dot = dot;
            best_idx = i as u8;
        }
    }
    best_idx
}

impl Primitive for CastShellE8 {
    fn describe() -> &'static str {
        "project 8D blocks onto the 240 minimal roots of the E8 Gosset lattice"
    }

    fn fit(&self, vectors: ArrayView2<f32>, _calib: Option<ArrayView2<f32>>) -> Vec<u8> {
        coding::pack_model(vectors.ncols())
    }

    fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let (n_rows, dim) = vectors.dim();
        assert!(dim.is_multiple_of(8), "dim must be a multiple of 8, got {dim}");
        let n_blocks = dim / 8;
        let mut out = Vec::with_capacity(n_rows);

        for row in vectors.rows() {
            let slice = row.as_slice().expect("contiguous row");
            let mut code = Vec::with_capacity(n_blocks);
            for block in 0..n_blocks {
                let mut x8 = [0.0f32; 8];
                x8.copy_from_slice(&slice[block * 8..(block + 1) * 8]);
                let idx = find_nearest_root(&x8);
                code.push(idx);
            }
            out.push(code);
        }
        out
    }

    fn reconstruct(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let d = super::code_dim(model, child_recons);
        assert!(d.is_multiple_of(8), "dim must be a multiple of 8, got {d}");
        let n_blocks = d / 8;
        let roots = &*ROOT_TABLE;
        let n_rows = codes.len();
        let mut flat = vec![0.0f32; n_rows * d];

        for (r, code) in codes.iter().enumerate() {
            let offset = r * d;
            for b in 0..n_blocks {
                let root_idx = code[b] as usize;
                let root = &roots[root_idx];
                flat[offset + b * 8..offset + (b + 1) * 8].copy_from_slice(root);
            }
        }

        let mut out = Array2::from_shape_vec((n_rows, d), flat).expect("valid shape");
        if let Some(child) = child_recons {
            out += &child;
        }
        out
    }

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        let recons = self.reconstruct(model, codes, None);
        *vectors -= &recons;
    }

    fn score(
        &self,
        model: &[u8],
        queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let d = super::code_dim(model, None);
        assert!(d.is_multiple_of(8), "dim must be a multiple of 8, got {d}");
        let recons = self.reconstruct(model, codes, None);
        let mut scores = math::matmul(queries, recons.t());
        if let Some(child) = child_scores {
            scores += &child;
        }
        scores
    }

    fn code_bytes(&self, model: &[u8], in_dim: usize) -> Option<usize> {
        let d = if model.is_empty() { in_dim } else { coding::unpack_model::<usize>(model) };
        Some(d / 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::assert_pipeline_scores;
    use ndarray::Array2;

    #[test]
    fn roots_table_kissing_number_and_norms() {
        let roots = &*ROOT_TABLE;
        assert_eq!(roots.len(), 240);
        for (i, r) in roots.iter().enumerate() {
            let norm_sq: f32 = r.iter().map(|&x| x * x).sum();
            assert!((norm_sq - 1.0).abs() < 1e-5, "root {i} has norm_sq {norm_sq}");
        }
    }

    #[test]
    fn round_trip_pipeline() {
        let mut rng = math::seed(1);
        let v: Array2<f32> = math::gaussian(&mut rng, (20, 16));
        let q: Array2<f32> = math::gaussian(&mut rng, (5, 16));

        assert_pipeline_scores(
            vec![Box::new(CastShellE8::new()) as Box<dyn Primitive>],
            v.view(),
            q.view(),
            None,
            1e-3,
        );
    }
}
