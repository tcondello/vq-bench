//! CAST_MULTI_SHELL_E8: concentric multi-shell E8 Gosset lattice rounder
//! -
//! Model: scale (f32), in_dim (usize)
//! Code for vector x: b bits per dimension (packed 8b-bit lattice codes)
//! Apply: x --> x - hat{x}
//! Reconstruct: code --> hat{x} (scaled lattice vectors on concentric E8 shells)
//! Score: <q, hat{x}> (+ child_score)

use ndarray::{Array2, ArrayView2};

use crate::coding::{self, CodeLayout};
use crate::{math, Primitive};

/// Concentric multi-shell E8 Gosset lattice rounder.
pub struct CastMultiShellE8 {
    bits: u8,
}

impl CastMultiShellE8 {
    pub fn new(bits: u8) -> Self {
        assert!(
            (1..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 1..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
        Self { bits }
    }
}

/// The code layout: `bits` per dimension packed into standard bit planes.
fn layout(bits: u8, dim: usize) -> CodeLayout {
    CodeLayout::new().bits(dim, bits)
}

/// Nearest integer with tie-breaking towards even.
#[inline]
fn round_half_to_even(x: f32) -> f32 {
    let round = x.round();
    if (x - round).abs() == 0.5 && (round as i32) % 2 != 0 {
        if x > round { round + 1.0 } else { round - 1.0 }
    } else {
        round
    }
}

/// Project an 8-dimensional point onto the closest point of the E8 root lattice.
#[inline]
fn project_e8_8d(x: &[f32; 8]) -> [f32; 8] {
    // Coset 0: Z^8 with even coordinate sum
    let mut f0 = [0.0f32; 8];
    let mut sum0 = 0i32;
    let mut max_diff0 = 0.0f32;
    let mut max_idx0 = 0;

    for i in 0..8 {
        let r = round_half_to_even(x[i]);
        f0[i] = r;
        sum0 += r as i32;
        let diff = (x[i] - r).abs();
        if diff > max_diff0 {
            max_diff0 = diff;
            max_idx0 = i;
        }
    }

    if sum0 % 2 != 0 {
        if x[max_idx0] > f0[max_idx0] {
            f0[max_idx0] += 1.0;
        } else {
            f0[max_idx0] -= 1.0;
        }
    }

    // Coset 1: (Z + 1/2)^8 with even coordinate sum
    let mut f1 = [0.0f32; 8];
    let mut sum1 = 0i32;
    let mut max_diff1 = 0.0f32;
    let mut max_idx1 = 0;

    for i in 0..8 {
        let x_shifted = x[i] - 0.5;
        let r = round_half_to_even(x_shifted);
        f1[i] = r + 0.5;
        sum1 += r as i32;
        let diff = (x_shifted - r).abs();
        if diff > max_diff1 {
            max_diff1 = diff;
            max_idx1 = i;
        }
    }

    if sum1 % 2 != 0 {
        if (x[max_idx1] - 0.5) > (f1[max_idx1] - 0.5) {
            f1[max_idx1] += 1.0;
        } else {
            f1[max_idx1] -= 1.0;
        }
    }

    // Distance to Coset 0 vs Coset 1
    let mut dist0 = 0.0f32;
    let mut dist1 = 0.0f32;
    for i in 0..8 {
        let d0 = x[i] - f0[i];
        let d1 = x[i] - f1[i];
        dist0 += d0 * d0;
        dist1 += d1 * d1;
    }

    if dist0 <= dist1 {
        f0
    } else {
        f1
    }
}

impl Primitive for CastMultiShellE8 {
    fn describe() -> &'static str {
        "quantize 8D subvectors onto concentric E8 Gosset lattice shells"
    }

    fn fit(&self, vectors: ArrayView2<f32>, _calib: Option<ArrayView2<f32>>) -> Vec<u8> {
        let dim = vectors.ncols();
        let total_sq: f32 = vectors.iter().map(|&x| x * x).sum();
        let rms = (total_sq / (vectors.len() as f32)).sqrt().max(1e-6);
        let max_lvl = ((1u32 << (self.bits - 1)) as f32).max(1.0);
        let step = (rms * 2.5) / max_lvl;
        coding::pack_model((step, dim))
    }

    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let (step, dim): (f32, usize) = coding::unpack_model(model);
        assert!(dim.is_multiple_of(8), "dim must be a multiple of 8, got {dim}");
        let n_rows = vectors.nrows();
        let n_blocks = dim / 8;
        let inv_step = 1.0 / step;

        let max_int = ((1i32 << self.bits) - 1) as f32;
        let mid = (1i32 << (self.bits - 1)) as f32;

        let mut out_levels = Array2::<u32>::zeros((n_rows, dim));

        for (r, row) in vectors.rows().into_iter().enumerate() {
            let slice = row.as_slice().expect("contiguous row");
            for b in 0..n_blocks {
                let mut x8 = [0.0f32; 8];
                for k in 0..8 {
                    x8[k] = slice[b * 8 + k] * inv_step;
                }
                let p8 = project_e8_8d(&x8);
                for k in 0..8 {
                    let lvl = (p8[k] + mid).clamp(0.0, max_int).round() as u32;
                    out_levels[[r, b * 8 + k]] = lvl;
                }
            }
        }

        layout(self.bits, dim).pack(out_levels.view(), &[])
    }

    fn reconstruct(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let (step, dim): (f32, usize) = coding::unpack_model(model);
        let (lvls, _) = layout(self.bits, dim).unpack::<0>(codes);
        let mid = (1i32 << (self.bits - 1)) as f32;

        let mut out = lvls.mapv(|u| (u as f32 - mid) * step);
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
        let (step, dim): (f32, usize) = coding::unpack_model(model);
        let (lvls, _) = layout(self.bits, dim).unpack::<0>(codes);
        let mid = (1i32 << (self.bits - 1)) as f32;

        let q_sum = queries.sum_axis(ndarray::Axis(1));
        let dots = math::matmul(queries, lvls.mapv(|u| u as f32).t());

        let mut scores = Array2::<f32>::zeros(dots.dim());
        for (q, row) in dots.rows().into_iter().enumerate() {
            let offset = q_sum[q] * mid;
            for (c, &dot) in row.into_iter().enumerate() {
                scores[[q, c]] = (dot - offset) * step;
            }
        }

        if let Some(child) = child_scores {
            scores += &child;
        }
        scores
    }

    fn code_bytes(&self, model: &[u8], in_dim: usize) -> Option<usize> {
        let dim = if model.is_empty() { in_dim } else { coding::unpack_model::<(f32, usize)>(model).1 };
        Some(layout(self.bits, dim).byte_len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::assert_pipeline_scores;
    use ndarray::Array2;

    #[test]
    fn round_trip_pipeline() {
        let mut rng = math::seed(1);
        let v: Array2<f32> = math::gaussian(&mut rng, (30, 16));
        let q: Array2<f32> = math::gaussian(&mut rng, (6, 16));

        assert_pipeline_scores(
            vec![Box::new(CastMultiShellE8::new(4)) as Box<dyn Primitive>],
            v.view(),
            q.view(),
            None,
            1e-3,
        );
    }
}
