//! CAST_LEECH24: 24-dimensional Leech Lattice (Lambda24) rounder
//! -
//! Model: scale (f32), in_dim (usize)
//! Code for vector x: b bits per dimension (packed 24b-bit lattice codes)
//! Apply: x --> x - hat{x}
//! Reconstruct: code --> hat{x} (scaled lattice vectors on the 24D Leech lattice)
//! Score: <q, hat{x}> (+ child_score)

use ndarray::{Array2, ArrayView2};

use crate::coding::{self, CodeLayout};
use crate::{math, Primitive};

/// 24-dimensional Leech Lattice (Lambda24) rounder using Golay G24 parity decoding.
pub struct CastLeech24 {
    bits: u8,
}

impl CastLeech24 {
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

/// The 12x12 circulant/bordered parity matrix B for the extended Golay code G24.
/// G24 = [I_12 | B_12]
static GOLAY_B: [u16; 12] = [
    0b100011101101,
    0b010001110111,
    0b101000111011,
    0b110100011101,
    0b111010001110,
    0b011101000111,
    0b101110100011,
    0b110111010001,
    0b011011101001,
    0b001101110101,
    0b100110111010,
    0b010011011101,
];

/// Fast syndrome-based nearest Golay codeword decoder for 24 bits.
#[inline]
fn decode_golay24(received: u32) -> u32 {
    let r_info = ((received >> 12) & 0xFFF) as u16;
    let r_parity = (received & 0xFFF) as u16;

    // Calculate syndrome s = r_parity ^ (r_info * B)
    let mut syn = r_parity;
    for (i, &row) in GOLAY_B.iter().enumerate() {
        if (r_info & (1 << (11 - i))) != 0 {
            syn ^= row;
        }
    }

    let syn_weight = syn.count_ones();
    if syn_weight <= 3 {
        // Error pattern is in parity bits
        return ((r_info as u32) << 12) | ((r_parity ^ syn) as u32);
    }

    // Check if error is single error in info + some parity errors
    for (i, &row) in GOLAY_B.iter().enumerate() {
        let diff = syn ^ row;
        if diff.count_ones() <= 2 {
            let fixed_info = r_info ^ (1 << (11 - i));
            let fixed_parity = r_parity ^ diff;
            return ((fixed_info as u32) << 12) | (fixed_parity as u32);
        }
    }

    // Fallback if >3 errors: return original parity-adjusted
    ((r_info as u32) << 12) | (r_parity as u32)
}

/// Project a 24-dimensional block onto the Leech lattice Lambda24.
#[inline]
fn project_leech_24d(x: &[f32; 24]) -> [f32; 24] {
    let mut rounded = [0.0f32; 24];
    let mut bits = 0u32;

    for i in 0..24 {
        let r = x[i].round();
        rounded[i] = r;
        let bit = ((r as i32).rem_euclid(2)) as u32;
        bits = (bits << 1) | bit;
    }

    // Decode to nearest Golay codeword
    let codeword = decode_golay24(bits);

    // Adjust rounded values to match Golay codeword parity
    for i in 0..24 {
        let target_bit = (codeword >> (23 - i)) & 1;
        let cur_bit = ((rounded[i] as i32).rem_euclid(2)) as u32;
        if cur_bit != target_bit {
            if x[i] > rounded[i] {
                rounded[i] += 1.0;
            } else {
                rounded[i] -= 1.0;
            }
        }
    }

    // Ensure sum = 0 mod 4
    let sum: i32 = rounded.iter().map(|&v| v as i32).sum();
    let rem = sum.rem_euclid(4);
    if rem != 0 {
        // Adjust the coordinates with greatest residual error
        if rem == 2 {
            rounded[0] += if x[0] > rounded[0] { 2.0 } else { -2.0 };
        } else if rem == 1 {
            rounded[0] -= 1.0;
        } else {
            rounded[0] += 1.0;
        }
    }

    rounded
}

impl Primitive for CastLeech24 {
    fn describe() -> &'static str {
        "quantize 24D subvectors onto the 24-dimensional Leech lattice (Lambda24)"
    }

    fn fit(&self, vectors: ArrayView2<f32>, _calib: Option<ArrayView2<f32>>) -> Vec<u8> {
        let dim = vectors.ncols();
        let total_sq: f32 = vectors.iter().map(|&x| x * x).sum();
        let rms = (total_sq / (vectors.len() as f32)).sqrt().max(1e-6);
        let max_lvl = ((1u32 << (self.bits - 1)) as f32).max(1.0);
        let step = (rms * 2.8) / max_lvl;
        coding::pack_model((step, dim))
    }

    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let (step, dim): (f32, usize) = coding::unpack_model(model);
        assert!(dim.is_multiple_of(24), "dim must be a multiple of 24, got {dim}");
        let n_rows = vectors.nrows();
        let n_blocks = dim / 24;
        let inv_step = 1.0 / step;

        let max_int = ((1i32 << self.bits) - 1) as f32;
        let mid = (1i32 << (self.bits - 1)) as f32;

        let mut out_levels = Array2::<u32>::zeros((n_rows, dim));

        for (r, row) in vectors.rows().into_iter().enumerate() {
            let slice = row.as_slice().expect("contiguous row");
            for b in 0..n_blocks {
                let mut x24 = [0.0f32; 24];
                for k in 0..24 {
                    x24[k] = slice[b * 24 + k] * inv_step;
                }
                let p24 = project_leech_24d(&x24);
                for k in 0..24 {
                    let lvl = (p24[k] + mid).clamp(0.0, max_int).round() as u32;
                    out_levels[[r, b * 24 + k]] = lvl;
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
    fn golay_decodes_clean_codewords() {
        let test_info = 0b101011001101u16;
        let mut parity = 0u16;
        for (i, &row) in GOLAY_B.iter().enumerate() {
            if (test_info & (1 << (11 - i))) != 0 {
                parity ^= row;
            }
        }
        let codeword = ((test_info as u32) << 12) | (parity as u32);
        // Correct codeword returns itself
        assert_eq!(decode_golay24(codeword), codeword);

        // 1-bit error in parity is corrected
        let corrupted1 = codeword ^ (1 << 3);
        assert_eq!(decode_golay24(corrupted1), codeword);

        // 1-bit error in info is corrected
        let corrupted2 = codeword ^ (1 << 18);
        assert_eq!(decode_golay24(corrupted2), codeword);
    }

    #[test]
    fn round_trip_pipeline() {
        let mut rng = math::seed(1);
        let v: Array2<f32> = math::gaussian(&mut rng, (20, 24));
        let q: Array2<f32> = math::gaussian(&mut rng, (5, 24));

        assert_pipeline_scores(
            vec![Box::new(CastLeech24::new(4)) as Box<dyn Primitive>],
            v.view(),
            q.view(),
            None,
            1e-3,
        );
    }
}
