//! CAST_E8: 8-dimensional Gosset lattice sphere-packing vector quantization.
//! -
//! Model: input dim d
//! Code for vector x: d b-bit lattice coordinate levels (shifted 2p_i), plus trailing f32 scale
//! Apply: x --> x - hat(x)          (residual for downstream stages)
//! Reconstruct: y --> scale * p(x) + y
//! Score: s --> scale * <q, p(x)> + s

use ndarray::{Array1, Array2, ArrayView2};

use crate::coding::CodeLayout;
use crate::{coding, Primitive};

/// Number of dimensions per E8 lattice block.
pub const E8_DIM: usize = 8;

/// E8 Gosset lattice rounder.
///
/// Projects 8-dimensional subvectors onto the E8 root lattice using the Conway-Sloane
/// (1982) O(1) constant-time decoding algorithm.
pub struct CastE8 {
    bits: u8,
}

impl CastE8 {
    /// Create a new CastE8 rounder with `bits` per coordinate.
    pub fn new(bits: u8) -> Self {
        assert!((2..=8).contains(&bits), "bits must be in 2..=8, got {bits}");
        Self { bits }
    }

    fn max_coord(&self) -> f32 {
        ((1 << (self.bits - 1)) - 1) as f32
    }

    fn offset(&self) -> u32 {
        1 << (self.bits - 1)
    }

    fn layout(&self, d: usize) -> CodeLayout {
        CodeLayout::new().bits(d, self.bits).scalars(1)
    }
}

/// Conway-Sloane (1982) fast nearest-point projection algorithm for the E8 lattice.
///
/// Finds the closest point in E_8 = D_8 U (D_8 + 1/2 * 1) for an 8-dimensional vector.
pub fn nearest_e8_point(y: &[f32; 8], max_val: f32) -> [f32; 8] {
    // 1. Candidate in D_8: nearest integer vector with even coordinate sum
    let mut k = [0i32; 8];
    let mut sum_k = 0i32;
    let mut max_err_d8 = -1.0f32;
    let mut worst_idx_d8 = 0usize;
    let mut diff_d8 = [0.0f32; 8];

    for i in 0..8 {
        let rounded = y[i].round() as i32;
        k[i] = rounded;
        sum_k += rounded;
        let err = (y[i] - rounded as f32).abs();
        diff_d8[i] = y[i] - rounded as f32;
        if err > max_err_d8 {
            max_err_d8 = err;
            worst_idx_d8 = i;
        }
    }

    if sum_k % 2 != 0 {
        // Adjust the coordinate with the largest rounding error
        if diff_d8[worst_idx_d8] >= 0.0 {
            k[worst_idx_d8] += 1;
        } else {
            k[worst_idx_d8] -= 1;
        }
    }

    // 2. Candidate in D_8 + 1/2: nearest half-integer vector with even sum
    let mut m = [0i32; 8];
    let mut sum_m = 0i32;
    let mut max_err_coset = -1.0f32;
    let mut worst_idx_coset = 0usize;
    let mut diff_coset = [0.0f32; 8];

    for i in 0..8 {
        let y_shift = y[i] - 0.5;
        let rounded = y_shift.round() as i32;
        m[i] = rounded;
        sum_m += rounded;
        let err = (y_shift - rounded as f32).abs();
        diff_coset[i] = y_shift - rounded as f32;
        if err > max_err_coset {
            max_err_coset = err;
            worst_idx_coset = i;
        }
    }

    if sum_m % 2 != 0 {
        if diff_coset[worst_idx_coset] >= 0.0 {
            m[worst_idx_coset] += 1;
        } else {
            m[worst_idx_coset] -= 1;
        }
    }

    // Compare squared distances to y
    let mut dist_d8 = 0.0f32;
    let mut dist_coset = 0.0f32;
    for i in 0..8 {
        let d1 = y[i] - k[i] as f32;
        let d2 = y[i] - (m[i] as f32 + 0.5);
        dist_d8 += d1 * d1;
        dist_coset += d2 * d2;
    }

    let mut out = [0.0f32; 8];
    if dist_d8 <= dist_coset {
        for i in 0..8 {
            out[i] = (k[i] as f32).clamp(-max_val, max_val);
        }
    } else {
        for i in 0..8 {
            out[i] = (m[i] as f32 + 0.5).clamp(-max_val, max_val);
        }
    }
    out
}

impl Primitive for CastE8 {
    fn describe() -> &'static str {
        "round 8-dimensional blocks to the E8 Gosset root lattice (Conway-Sloane)"
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        coding::pack_model(vectors.ncols())
    }

    fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let (n, d) = (vectors.nrows(), vectors.ncols());
        let max_val = self.max_coord();
        let offset = self.offset();
        let mut levels = Array2::<u32>::zeros((n, d));
        let mut scales = Array1::<f32>::zeros(n);

        for (i, row) in vectors.rows().into_iter().enumerate() {
            // Per-vector scale: RMS-based or AbsMax
            let mut max_abs = 0.0f32;
            for &x in row.iter() {
                max_abs = max_abs.max(x.abs());
            }
            let scale = if max_abs > 1e-12 { max_abs / max_val } else { 1.0 };
            let inv_scale = 1.0 / scale;
            scales[i] = scale;

            for block in 0..(d / E8_DIM) {
                let start = block * E8_DIM;
                let mut chunk = [0.0f32; 8];
                for j in 0..8 {
                    chunk[j] = row[start + j] * inv_scale;
                }
                let pt = nearest_e8_point(&chunk, max_val);
                for j in 0..8 {
                    // Double coordinate (2 * pt[j]) to convert half-integers to odd integers
                    let doubled = (pt[j] * 2.0).round() as i32;
                    let code_val = (doubled + offset as i32).clamp(0, (1 << self.bits) - 1) as u32;
                    levels[[i, start + j]] = code_val;
                }
            }
        }

        self.layout(d).pack(levels.view(), &[scales.view()])
    }

    fn apply(&self, _model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        let (_n, d) = vectors.dim();
        let (levels, [scales]) = self.layout(d).unpack::<1>(codes);
        let offset = self.offset() as f32;

        for (i, mut row) in vectors.rows_mut().into_iter().enumerate() {
            let s = scales[i];
            for j in 0..d {
                let p_j = (levels[[i, j]] as f32 - offset) * 0.5;
                row[j] -= s * p_j;
            }
        }
    }

    fn reconstruct(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let d = super::code_dim(model, child_recons);
        let (levels, [scales]) = self.layout(d).unpack::<1>(codes);
        let offset = self.offset() as f32;
        let mut out = Array2::zeros((codes.len(), d));

        for (i, mut row) in out.rows_mut().into_iter().enumerate() {
            let s = scales[i];
            for j in 0..d {
                let p_j = (levels[[i, j]] as f32 - offset) * 0.5;
                row[j] = s * p_j;
            }
        }

        if let Some(child) = child_recons {
            out += &child;
        }
        out
    }

    fn score(
        &self,
        _model: &[u8],
        queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let d = queries.ncols();
        let (levels, [scales]) = self.layout(d).unpack::<1>(codes);
        let offset = self.offset() as f32;
        let n_q = queries.nrows();
        let n_v = codes.len();

        let mut out = Array2::zeros((n_q, n_v));
        for i in 0..n_v {
            let s = scales[i] * 0.5;
            for q_idx in 0..n_q {
                let q_row = queries.row(q_idx);
                let mut dot = 0.0f32;
                for j in 0..d {
                    dot += q_row[j] * (levels[[i, j]] as f32 - offset);
                }
                out[[q_idx, i]] = s * dot;
            }
        }

        if let Some(child) = child_scores {
            out += &child;
        }
        out
    }

    fn code_bytes(&self, _model: &[u8], in_dim: usize) -> Option<usize> {
        Some(self.layout(in_dim).byte_len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math;
    use crate::util::testing::{assert_close, refs};
    use ndarray::Array2;

    #[test]
    fn e8_nearest_point_satisfies_parity() {
        let y = [0.8, 1.1, -0.2, 0.4, -1.3, 0.9, 0.1, -0.7];
        let p = nearest_e8_point(&y, 8.0);
        let doubled: Vec<i32> = p.iter().map(|&x| (x * 2.0).round() as i32).collect();
        let is_all_even = doubled.iter().all(|&x| x % 2 == 0);
        let is_all_odd = doubled.iter().all(|&x| x % 2 != 0);
        assert!(is_all_even || is_all_odd, "must be D8 or coset");
        let sum: i32 = p.iter().map(|&x| x.round() as i32).sum();
        assert_eq!(sum % 2, 0, "sum of rounded coords must be even");
    }

    #[test]
    fn round_trip_and_score() {
        let mut rng = math::seed(42);
        let v: Array2<f32> = math::gaussian(&mut rng, (20, 16));
        let q: Array2<f32> = math::gaussian(&mut rng, (5, 16));

        let cast = CastE8::new(4);
        let model = cast.fit(v.view(), None);
        let codes = cast.encode(&model, v.view());
        let r = refs(&codes);

        let recon = cast.reconstruct(&model, &r, None);
        assert_eq!(recon.dim(), (20, 16));

        let scores = cast.score(&model, q.view(), &r, None);
        let exact = q.dot(&recon.t());
        assert_close(&scores, &exact, 1e-4);
    }
}
