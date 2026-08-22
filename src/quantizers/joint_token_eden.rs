//! `joint_token_eden`: Joint Intra-Document Multi-Vector Token Quantizer for ColBERT.
//! -
//! Jointly codes blocks of L document token vectors by quantizing a shared document centroid
//! anchor and compressing intra-document token residuals.

use anyhow::{ensure, Result};
use ndarray::{Array1, Array2, ArrayView2, Axis};

use super::catalog::{get, get_or};
use super::rotation::Rotation;
use crate::coding::{self, CodeLayout};
use crate::{codebooks, Center, Normalize, Params, Primitive, Quantizer};

pub struct JointTokenEden {
    bits: u8,
    block_len: usize,
    rotation: Rotation,
    seed: u64,
    dim: usize,
    anchor_codebook: Vec<f32>,
    res_codebook: Vec<f32>,
}

impl JointTokenEden {
    pub fn new(bits: u8, block_len: usize, rotation: Rotation, seed: u64, dim: usize) -> Result<Self> {
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 1..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
        ensure!(block_len >= 1, "block_len must be >= 1, got {block_len}");

        Ok(Self {
            bits,
            block_len,
            rotation,
            seed,
            dim,
            anchor_codebook: codebooks::lloyd_max_normal(16), // 4-bit anchor for centroid
            res_codebook: codebooks::lloyd_max_normal(1usize << bits),
        })
    }

    fn anchor_layout(&self) -> CodeLayout {
        CodeLayout::new().bits(self.dim, 4).scalars(1)
    }

    fn res_layout(&self) -> CodeLayout {
        CodeLayout::new().bits(self.dim, self.bits).scalars(1)
    }
}

impl Quantizer for JointTokenEden {
    fn name() -> &'static str {
        "joint_token_eden"
    }

    fn display_name() -> &'static str {
        "JointTokenEDEN"
    }

    fn params() -> &'static [&'static str] {
        &["b", "block_len", "rotation"]
    }

    fn describe() -> &'static str {
        "Joint intra-document multi-vector token quantizer (shared anchor + token residuals)"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let bits = get(p, "b")?;
        let block_len = get_or(p, "block_len", 32usize)?;
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;
        Self::new(bits, block_len, rotation, seed, dim)
    }

    fn fit(&self, vectors: ArrayView2<f32>, queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let center = Center;
        let norm = Normalize;
        let rot = self.rotation.stage(self.seed);

        let m_center = center.fit(vectors, queries);
        let mut v = vectors.to_owned();
        center.apply(&m_center, &mut v, &[]);

        let m_norm = norm.fit(v.view(), None);
        let norms = v.mapv(|x| x * x).sum_axis(Axis(1)).mapv(f32::sqrt);
        let norm_codes = CodeLayout::new().scalars(1).pack_scalars(&[norms.view()]);
        let norm_refs: Vec<&[u8]> = norm_codes.iter().map(Vec::as_slice).collect();
        norm.apply(&m_norm, &mut v, &norm_refs);

        let m_rot = rot.fit(v.view(), None);
        rot.apply(&m_rot, &mut v, &[]);

        coding::pack_model((m_center, m_norm, m_rot, self.dim))
    }

    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let (m_center, m_norm, m_rot, _d): (Vec<u8>, Vec<u8>, Vec<u8>, usize) =
            coding::unpack_model(model);

        let center = Center;
        let norm = Normalize;
        let rot = self.rotation.stage(self.seed);

        let mut v = vectors.to_owned();
        let norms = vectors.mapv(|x| x * x).sum_axis(Axis(1)).mapv(f32::sqrt);
        let norm_codes = CodeLayout::new().scalars(1).pack_scalars(&[norms.view()]);
        let norm_refs: Vec<&[u8]> = norm_codes.iter().map(Vec::as_slice).collect();

        center.apply(&m_center, &mut v, &[]);
        norm.apply(&m_norm, &mut v, &norm_refs);
        rot.apply(&m_rot, &mut v, &[]);

        let (n_v, d) = (v.nrows(), v.ncols());
        let sqrt_d = (d as f32).sqrt();
        let inv_sqrt_d = 1.0 / sqrt_d;

        let anchor_len = self.anchor_layout().byte_len();
        let res_len = self.res_layout().byte_len();
        let norm_len = CodeLayout::new().scalars(1).byte_len();

        let mut encoded = Vec::with_capacity(n_v);

        // Process in document blocks of length `self.block_len`
        let n_blocks = n_v.div_ceil(self.block_len);
        for b_idx in 0..n_blocks {
            let start = b_idx * self.block_len;
            let end = (start + self.block_len).min(n_v);
            let block_size = end - start;

            let block = v.slice(ndarray::s![start..end, ..]);

            // 1. Compute block centroid
            let mut centroid = Array1::<f32>::zeros(d);
            for row in block.rows() {
                centroid += &row;
            }
            centroid.mapv_inplace(|x| x / (block_size as f32));

            // Quantize centroid anchor (4-bit Gaussian)
            let mut anchor_levels = Array2::<u32>::zeros((1, d));
            let mut anchor_dequant = vec![0f32; d];
            for (j, &c) in centroid.iter().enumerate() {
                let lvl = codebooks::nearest(&self.anchor_codebook, c * sqrt_d);
                anchor_levels[[0, j]] = lvl as u32;
                anchor_dequant[j] = self.anchor_codebook[lvl] * inv_sqrt_d;
            }
            let dot_ca: f32 = centroid.iter().zip(&anchor_dequant).map(|(&x, &a)| x * a).sum();
            let norm_a_sq: f32 = anchor_dequant.iter().map(|&a| a * a).sum();
            let s_anchor = if dot_ca > 1e-9 && norm_a_sq > 1e-9 { dot_ca / norm_a_sq } else { 1.0 };
            let anchor_scale = Array1::from_vec(vec![s_anchor]);

            let packed_anchor = self.anchor_layout().pack(anchor_levels.view(), &[anchor_scale.view()]);
            let anchor_bytes = &packed_anchor[0];

            let anchor_recon: Vec<f32> = anchor_dequant.iter().map(|&x| x * s_anchor).collect();

            // 2. Quantize each token's residual relative to the anchor
            for (i, row) in block.rows().into_iter().enumerate() {
                let glob_idx = start + i;
                let mut res_levels = Array2::<u32>::zeros((1, d));
                let mut res_codeword = vec![0f32; d];

                for (j, &x) in row.iter().enumerate() {
                    let r = x - anchor_recon[j];
                    let lvl = codebooks::nearest(&self.res_codebook, r * sqrt_d);
                    res_levels[[0, j]] = lvl as u32;
                    res_codeword[j] = self.res_codebook[lvl] * inv_sqrt_d;
                }

                let dot_rc: f32 = row.iter().zip(&anchor_recon).zip(&res_codeword)
                    .map(|((&x, &a), &c)| (x - a) * c).sum();
                let norm_rc_sq: f32 = res_codeword.iter().map(|&c| c * c).sum();
                let s_res = if dot_rc > 1e-9 && norm_rc_sq > 1e-9 { dot_rc / norm_rc_sq } else { 1.0 };
                let res_scale = Array1::from_vec(vec![s_res]);

                let packed_res = self.res_layout().pack(res_levels.view(), &[res_scale.view()]);

                // Combined code layout per vector: [Norm (4B), Anchor (anchor_len B), Residual (res_len B)]
                let n_c = &norm_codes[glob_idx];
                let mut row_code = Vec::with_capacity(norm_len + anchor_len + res_len);
                row_code.extend_from_slice(n_c);
                row_code.extend_from_slice(anchor_bytes);
                row_code.extend_from_slice(&packed_res[0]);
                encoded.push(row_code);
            }
        }

        encoded
    }

    fn reconstruct(&self, model: &[u8], codes: &[&[u8]]) -> Array2<f32> {
        let (m_center, m_norm, m_rot, _d): (Vec<u8>, Vec<u8>, Vec<u8>, usize) =
            coding::unpack_model(model);

        let norm_len = CodeLayout::new().scalars(1).byte_len();
        let anchor_len = self.anchor_layout().byte_len();

        let norm_slices: Vec<&[u8]> = codes.iter().map(|c| &c[..norm_len]).collect();
        let anchor_slices: Vec<&[u8]> = codes.iter().map(|c| &c[norm_len..norm_len + anchor_len]).collect();
        let res_slices: Vec<&[u8]> = codes.iter().map(|c| &c[norm_len + anchor_len..]).collect();

        let (a_levels, [a_scales]) = self.anchor_layout().unpack::<1>(&anchor_slices);
        let (r_levels, [r_scales]) = self.res_layout().unpack::<1>(&res_slices);

        let inv_sqrt_d = 1.0 / (self.dim as f32).sqrt();
        let n = codes.len();
        let mut recon = Array2::<f32>::zeros((n, self.dim));

        for i in 0..n {
            let a_s = a_scales[i];
            let r_s = r_scales[i];
            for j in 0..self.dim {
                let a_val = self.anchor_codebook[a_levels[[i, j]] as usize] * inv_sqrt_d * a_s;
                let r_val = self.res_codebook[r_levels[[i, j]] as usize] * inv_sqrt_d * r_s;
                recon[[i, j]] = a_val + r_val;
            }
        }

        let rot = self.rotation.stage(self.seed);
        let recon_rot = rot.reconstruct(&m_rot, &[], Some(recon.view()));

        let norm = Normalize;
        let recon_norm = norm.reconstruct(&m_norm, &norm_slices, Some(recon_rot.view()));

        let center = Center;
        center.reconstruct(&m_center, &[], Some(recon_norm.view()))
    }

    fn score(&self, model: &[u8], queries: ArrayView2<f32>, codes: &[&[u8]]) -> Array2<f32> {
        let (m_center, m_norm, m_rot, _d): (Vec<u8>, Vec<u8>, Vec<u8>, usize) =
            coding::unpack_model(model);

        let center = Center;
        let norm = Normalize;
        let rot = self.rotation.stage(self.seed);

        let mut q = queries.to_owned();
        center.apply_queries(&m_center, &mut q);
        norm.apply_queries(&m_norm, &mut q);
        rot.apply_queries(&m_rot, &mut q);

        let norm_len = CodeLayout::new().scalars(1).byte_len();
        let anchor_len = self.anchor_layout().byte_len();

        let norm_slices: Vec<&[u8]> = codes.iter().map(|c| &c[..norm_len]).collect();
        let anchor_slices: Vec<&[u8]> = codes.iter().map(|c| &c[norm_len..norm_len + anchor_len]).collect();
        let res_slices: Vec<&[u8]> = codes.iter().map(|c| &c[norm_len + anchor_len..]).collect();

        let (a_levels, [a_scales]) = self.anchor_layout().unpack::<1>(&anchor_slices);
        let (r_levels, [r_scales]) = self.res_layout().unpack::<1>(&res_slices);

        let inv_sqrt_d = 1.0 / (self.dim as f32).sqrt();
        let n = codes.len();
        let mut dequant = Array2::<f32>::zeros((n, self.dim));

        for i in 0..n {
            let a_s = a_scales[i];
            let r_s = r_scales[i];
            for j in 0..self.dim {
                let a_val = self.anchor_codebook[a_levels[[i, j]] as usize] * inv_sqrt_d * a_s;
                let r_val = self.res_codebook[r_levels[[i, j]] as usize] * inv_sqrt_d * r_s;
                dequant[[i, j]] = a_val + r_val;
            }
        }

        let scores = q.dot(&dequant.t());

        let rot_score = rot.score(&m_rot, queries, &[], Some(scores.view()));
        let norm_score = norm.score(&m_norm, queries, &norm_slices, Some(rot_score.view()));
        center.score(&m_center, queries, &[], Some(norm_score.view()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math;
    use crate::util::testing::{params, refs};
    use ndarray::Array2;
    use serde_json::json;

    #[test]
    fn test_joint_token_eden_build_and_score() {
        let mut rng = math::seed(42);
        let d = 64;
        let v: Array2<f32> = math::gaussian(&mut rng, (64, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (5, d));

        for bits in [1, 2] {
            let p = params(&[("b", json!(bits)), ("block_len", json!(16))]);
            let codec = JointTokenEden::build(&p, 1, d).unwrap();
            let model = codec.fit(v.view(), None);
            let codes = codec.encode(&model, v.view());
            let est = codec.score(&model, q.view(), &refs(&codes));
            let exact = q.dot(&v.t());
            let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
            let st: f32 = exact.iter().map(|t| t * t).sum();
            assert!(((se / st) - 1.0).abs() < 0.45, "slope {}", se / st);
        }
    }
}
