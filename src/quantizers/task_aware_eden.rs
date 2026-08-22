//! `task_aware_eden`: Task-Aware Anisotropic Score-Preserving Quantization on EDEN.
//! -
//! Minimizes anisotropic ranking loss:
//! L_aniso(x, xhat) = ||x_perp - xhat_perp||^2 + (1 + omega) * ||x_par - xhat_par||^2
//! where x_par is the projection along the vector direction x.

use anyhow::{ensure, Result};
use ndarray::{Array1, Array2, ArrayView2, Axis};

use super::catalog::{get, get_or};
use super::rotation::Rotation;
use crate::coding::{self, CodeLayout};
use crate::{codebooks, math, Center, Normalize, Params, Primitive, Quantizer};

pub struct TaskAwareEden {
    bits: u8,
    omega: f32,
    rotation: Rotation,
    seed: u64,
    dim: usize,
    codebook: Vec<f32>,
}

impl TaskAwareEden {
    pub fn new(bits: u8, omega: f32, rotation: Rotation, seed: u64, dim: usize) -> Result<Self> {
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 1..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
        ensure!(omega >= 0.0, "omega must be >= 0.0, got {omega}");

        Ok(Self {
            bits,
            omega,
            rotation,
            seed,
            dim,
            codebook: codebooks::lloyd_max_normal(1usize << bits),
        })
    }

    fn layout(&self) -> CodeLayout {
        CodeLayout::new().bits(self.dim, self.bits).scalars(1)
    }

    fn split(&self, codes: &[&[u8]]) -> (Array2<u32>, Array1<f32>) {
        let (levels, [scales]) = self.layout().unpack::<1>(codes);
        (levels, scales)
    }

    fn dequant(&self, codes: &[&[u8]]) -> Array2<f32> {
        let (levels, scales) = self.split(codes);
        let inv_sqrt_d = 1.0 / (self.dim as f32).sqrt();
        let mut out = levels.mapv(|level| self.codebook[level as usize] * inv_sqrt_d);
        math::scale_rows(&mut out, scales.view());
        out
    }
}

impl Quantizer for TaskAwareEden {
    fn name() -> &'static str {
        "task_aware_eden"
    }

    fn display_name() -> &'static str {
        "TaskAwareEDEN"
    }

    fn params() -> &'static [&'static str] {
        &["b", "omega", "rotation"]
    }

    fn describe() -> &'static str {
        "Task-aware anisotropic score-preserving EDEN (ScaNN-style directional loss)"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let bits = get(p, "b")?;
        let omega = get_or(p, "omega", 1.0f32)?;
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;
        Self::new(bits, omega, rotation, seed, dim)
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

        let mut levels = Array2::<u32>::zeros((n_v, d));
        let mut scales = Array1::<f32>::zeros(n_v);

        // Anisotropic coordinate selection & optimal scale
        for (i, row) in v.rows().into_iter().enumerate() {
            let mut codeword = vec![0f32; d];
            for (j, &x) in row.iter().enumerate() {
                let level = codebooks::nearest(&self.codebook, x * sqrt_d);
                levels[[i, j]] = level as u32;
                codeword[j] = self.codebook[level] * inv_sqrt_d;
            }

            // Anisotropic optimal scale minimizing:
            // ||x - S*c||^2 + omega * (<x, x - S*c>)^2 / ||x||^2
            // For unit norm vector x (||x|| = 1):
            // dL/dS = -2 <x, c> + 2 S ||c||^2 - 2 omega (1 - S <x, c>) <x, c> = 0
            // => S * (||c||^2 + omega <x, c>^2) = (1 + omega) <x, c>
            // => S_aniso = (1 + omega) <x, c> / (||c||^2 + omega <x, c>^2)
            let dot_xc: f32 = row.iter().zip(&codeword).map(|(&x, &c)| x * c).sum();
            let norm_c_sq: f32 = codeword.iter().map(|&c| c * c).sum();
            
            let denom = norm_c_sq + self.omega * dot_xc * dot_xc;
            let s_aniso = if denom > 1e-9 && dot_xc > 1e-9 {
                (1.0 + self.omega) * dot_xc / denom
            } else if dot_xc > 1e-9 {
                1.0 / dot_xc
            } else {
                1.0
            };

            scales[i] = s_aniso;
        }

        let packed_levels = self.layout().pack(levels.view(), &[scales.view()]);
        
        // Combine norm codes and quantized levels
        let mut combined = Vec::with_capacity(n_v);
        for (n_c, p_c) in norm_codes.into_iter().zip(packed_levels) {
            let mut row = Vec::with_capacity(n_c.len() + p_c.len());
            row.extend_from_slice(&n_c);
            row.extend_from_slice(&p_c);
            combined.push(row);
        }
        combined
    }

    fn reconstruct(&self, model: &[u8], codes: &[&[u8]]) -> Array2<f32> {
        let (m_center, m_norm, m_rot, _d): (Vec<u8>, Vec<u8>, Vec<u8>, usize) =
            coding::unpack_model(model);

        let norm_len = CodeLayout::new().scalars(1).byte_len();
        let norm_slices: Vec<&[u8]> = codes.iter().map(|c| &c[..norm_len]).collect();
        let quant_slices: Vec<&[u8]> = codes.iter().map(|c| &c[norm_len..]).collect();

        let recon = self.dequant(&quant_slices);

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
        let norm_slices: Vec<&[u8]> = codes.iter().map(|c| &c[..norm_len]).collect();
        let quant_slices: Vec<&[u8]> = codes.iter().map(|c| &c[norm_len..]).collect();

        let dequant_vecs = self.dequant(&quant_slices);
        let scores = q.dot(&dequant_vecs.t());

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
    fn test_task_aware_eden_build_and_score() {
        let mut rng = math::seed(42);
        let d = 64;
        let v: Array2<f32> = math::gaussian(&mut rng, (40, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (5, d));

        for bits in [2, 4] {
            for omega in [0.0, 0.5, 2.0] {
                let p = params(&[("b", json!(bits)), ("omega", json!(omega))]);
                let codec = TaskAwareEden::build(&p, 1, d).unwrap();
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
}
