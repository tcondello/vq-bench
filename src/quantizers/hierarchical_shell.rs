//! `hierarchical_shell`: Progressive Nested Spherical Shell & Spatial Bit-Plane Quantizer (H3-style).
//! -
//! Hierarchically partitions the high-dimensional sphere into nested residual shells.
//! Codes are structured in progressive bit-planes: Level 1 (1 b/d coarse hemisphere),
//! Level 2 (2 b/d residual shell), Level 3 (3 b/d), Level 4 (4 b/d).
//!
//! Enables two key properties:
//! 1. Fast candidate generation at 1.0–1.5 b/d via binary spherical spatial partition.
//! 2. Prefix decodability: a 4 b/d index can be truncated to 2 b/d and 1 b/d for multi-rate search without retraining.

use anyhow::{ensure, Result};
use ndarray::{Array2, ArrayView2};

use super::catalog::{get_or};
use super::rotation::Rotation;
use crate::util::coding::{self, ModelField};
use crate::{Params, Quantizer};

/// The `hierarchical_shell` family.
pub struct HierarchicalShell {
    bits: u8,
    truncate_to: Option<u8>,
    rotation: Rotation,
    seed: u64,
}

impl HierarchicalShell {
    pub fn new(
        bits: u8,
        truncate_to: Option<u8>,
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Self> {
        ensure!((1..=8).contains(&bits), "b must be in 1..=8, got {bits}");
        if let Some(t) = truncate_to {
            ensure!(
                t > 0 && t <= bits,
                "truncate_to must be in 1..={bits}, got {t}"
            );
        }
        ensure!(dim > 0, "dim must be > 0");
        Ok(Self {
            bits,
            truncate_to,
            rotation,
            seed,
        })
    }
}

/// Serialized model fields for HierarchicalShell.
struct HierarchicalShellModel {
    dim: usize,
    bits: u8,
    mean: Vec<f32>,
    scales: Vec<f32>,
    rotation_model: Vec<u8>,
}

impl ModelField for HierarchicalShellModel {
    fn write(&self, buf: &mut Vec<u8>) {
        self.dim.write(buf);
        (self.bits as u32).write(buf);
        self.mean.write(buf);
        self.scales.write(buf);
        self.rotation_model.write(buf);
    }

    fn read(cur: &mut &[u8]) -> Self {
        let dim = usize::read(cur);
        let bits = u32::read(cur) as u8;
        let mean = Vec::<f32>::read(cur);
        let scales = Vec::<f32>::read(cur);
        let rotation_model = Vec::<u8>::read(cur);
        Self {
            dim,
            bits,
            mean,
            scales,
            rotation_model,
        }
    }
}

/// Pack 1-bit signs (true for >= 0, false for < 0) into bytes.
#[inline]
fn pack_signs(signs: &[bool]) -> Vec<u8> {
    let num_bytes = signs.len().div_ceil(8);
    let mut out = vec![0u8; num_bytes];
    for (i, &s) in signs.iter().enumerate() {
        if s {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

impl Quantizer for HierarchicalShell {
    fn name() -> &'static str {
        "hierarchical_shell"
    }

    fn display_name() -> &'static str {
        "HierarchicalShell"
    }

    fn params() -> &'static [&'static str] {
        &["b", "truncate_to", "rotation"]
    }

    fn describe() -> &'static str {
        "Progressive Nested Spherical Shell & Spatial Bit-Plane Quantizer (H3-style prefix decodable)"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let bits: u8 = get_or(p, "b", 1)?;
        let trunc_param: u8 = get_or(p, "truncate_to", 0)?;
        let truncate_to = if trunc_param == 0 { None } else { Some(trunc_param) };
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;
        Self::new(bits, truncate_to, rotation, seed, dim)
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let n = vectors.nrows();
        let dim = vectors.ncols();
        let bits = self.bits as usize;

        // 1. Mean center
        let mut mean = vec![0.0f32; dim];
        for row in vectors.rows() {
            for (j, &val) in row.iter().enumerate() {
                mean[j] += val;
            }
        }
        for m in &mut mean {
            *m /= n.max(1) as f32;
        }

        let mut centered = vectors.to_owned();
        for mut row in centered.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                *val -= mean[j];
            }
        }

        // 2. Unit normalize and rotate
        let mut unit_v = centered;
        for mut row in unit_v.rows_mut() {
            let norm = row.dot(&row).sqrt();
            if norm > 1e-12 {
                row.mapv_inplace(|x| x / norm);
            }
        }

        let rot_primitive = self.rotation.stage(self.seed ^ 0x4353);
        let rotation_model = rot_primitive.fit(unit_v.view(), None);
        let mut rotated = unit_v;
        rot_primitive.apply(&rotation_model, &mut rotated, &[]);

        // 3. Fit progressive nested spherical scales
        let mut residuals = rotated;
        let mut scales = Vec::with_capacity(bits);

        for _ in 0..bits {
            let mut sum_alpha = 0.0f64;
            let mut signs_batch = Vec::with_capacity(n);

            for row in residuals.rows() {
                let mut l1_norm = 0.0f32;
                let mut signs = Vec::with_capacity(dim);
                for &val in row.iter() {
                    let s = val >= 0.0;
                    signs.push(s);
                    l1_norm += val.abs();
                }
                let alpha = l1_norm / dim as f32;
                sum_alpha += alpha as f64;
                signs_batch.push((signs, alpha));
            }

            let avg_scale = (sum_alpha / n.max(1) as f64) as f32;
            scales.push(avg_scale);

            // Update residuals for next nested layer
            for (mut row, (signs, _)) in residuals.rows_mut().into_iter().zip(signs_batch) {
                for (j, &s) in signs.iter().enumerate() {
                    let s_val = if s { 1.0f32 } else { -1.0f32 };
                    row[j] -= avg_scale * s_val;
                }
            }
        }

        let model = HierarchicalShellModel {
            dim,
            bits: self.bits,
            mean,
            scales,
            rotation_model,
        };
        coding::pack_model(model)
    }

    fn encode(&self, model_bytes: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let model: HierarchicalShellModel = coding::unpack_model(model_bytes);
        let n = vectors.nrows();
        let dim = model.dim;
        let total_layers = model.bits as usize;
        let active_layers = self.truncate_to.map_or(total_layers, |t| (t as usize).min(total_layers));
        let bytes_per_layer = dim.div_ceil(8);

        // Center
        let mut centered = vectors.to_owned();
        for mut row in centered.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                *val -= model.mean[j];
            }
        }

        // Normalize
        for mut row in centered.rows_mut() {
            let norm = row.dot(&row).sqrt();
            if norm > 1e-12 {
                row.mapv_inplace(|x| x / norm);
            }
        }

        // Rotate
        let rot_primitive = self.rotation.stage(self.seed ^ 0x4353);
        let mut rotated = centered;
        rot_primitive.apply(&model.rotation_model, &mut rotated, &[]);

        let mut out = Vec::with_capacity(n);
        let mut residuals = rotated;

        // Encode layer by layer in progressive bit-planes
        let mut all_layer_codes = vec![Vec::with_capacity(bytes_per_layer); n * active_layers];

        for (l, &scale) in model.scales.iter().take(active_layers).enumerate() {
            for (i, mut row) in residuals.rows_mut().into_iter().enumerate() {
                let mut signs = Vec::with_capacity(dim);
                for val in row.iter_mut() {
                    let s = *val >= 0.0;
                    signs.push(s);
                    let s_val = if s { 1.0f32 } else { -1.0f32 };
                    *val -= scale * s_val;
                }
                let packed = pack_signs(&signs);
                all_layer_codes[i * active_layers + l] = packed;
            }
        }

        for i in 0..n {
            let mut vec_code = Vec::with_capacity(active_layers * bytes_per_layer);
            for l in 0..active_layers {
                vec_code.extend_from_slice(&all_layer_codes[i * active_layers + l]);
            }
            out.push(vec_code);
        }

        out
    }

    fn score(&self, model_bytes: &[u8], query: ArrayView2<f32>, cand_codes: &[&[u8]]) -> Array2<f32> {
        let model: HierarchicalShellModel = coding::unpack_model(model_bytes);
        let dim = model.dim;
        let bytes_per_layer = dim.div_ceil(8);

        // Center query
        let mut centered = query.to_owned();
        for mut row in centered.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                *val -= model.mean[j];
            }
        }

        // Normalize
        for mut row in centered.rows_mut() {
            let norm = row.dot(&row).sqrt();
            if norm > 1e-12 {
                row.mapv_inplace(|x| x / norm);
            }
        }

        // Rotate query
        let rot_primitive = self.rotation.stage(self.seed ^ 0x4353);
        let mut q_rot = centered;
        rot_primitive.apply(&model.rotation_model, &mut q_rot, &[]);
        let q_vec = q_rot.row(0);
        let q_sum: f32 = q_vec.sum();

        let mut scores = Vec::with_capacity(cand_codes.len());

        for &code in cand_codes {
            if code.is_empty() || bytes_per_layer == 0 {
                scores.push(0.0f32);
                continue;
            }

            let num_layers = code.len() / bytes_per_layer;
            let mut total_dot = 0.0f32;

            for (l, layer_chunk) in code.chunks_exact(bytes_per_layer).enumerate() {
                if l >= model.scales.len() || l >= num_layers {
                    break;
                }
                let scale = model.scales[l];

                // Fast unpacked dot: sum_{j: bit=1} q[j] * 2 - q_sum
                let mut set_sum = 0.0f32;
                for (byte_idx, &b) in layer_chunk.iter().enumerate() {
                    for bit in 0..8 {
                        let bit_idx = byte_idx * 8 + bit;
                        if bit_idx < dim && (b & (1 << bit)) != 0 {
                            set_sum += q_vec[bit_idx];
                        }
                    }
                }
                let layer_dot = 2.0 * set_sum - q_sum;
                total_dot += scale * layer_dot;
            }

            scores.push(total_dot);
        }

        Array2::from_shape_vec((1, cand_codes.len()), scores).expect("score shape")
    }

    fn reconstruct(&self, model_bytes: &[u8], cand_codes: &[&[u8]]) -> Array2<f32> {
        let model: HierarchicalShellModel = coding::unpack_model(model_bytes);
        let dim = model.dim;
        let bytes_per_layer = dim.div_ceil(8);
        let rot_primitive = self.rotation.stage(self.seed ^ 0x4353);

        let mut rotated_recons = Array2::<f32>::zeros((cand_codes.len(), dim));

        for (i, &code) in cand_codes.iter().enumerate() {
            if code.is_empty() || bytes_per_layer == 0 {
                continue;
            }
            let num_layers = code.len() / bytes_per_layer;
            for (l, layer_chunk) in code.chunks_exact(bytes_per_layer).enumerate() {
                if l >= model.scales.len() || l >= num_layers {
                    break;
                }
                let scale = model.scales[l];
                for (byte_idx, &b) in layer_chunk.iter().enumerate() {
                    for bit in 0..8 {
                        let bit_idx = byte_idx * 8 + bit;
                        if bit_idx < dim {
                            let sign = if (b & (1 << bit)) != 0 { 1.0f32 } else { -1.0f32 };
                            rotated_recons[[i, bit_idx]] += scale * sign;
                        }
                    }
                }
            }
        }

        // Invert rotation and add mean
        let reconstructed = rot_primitive.reconstruct(&model.rotation_model, &[], Some(rotated_recons.view()));
        let mut out = reconstructed;
        for mut row in out.rows_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                *val += model.mean[j];
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::params;
    use serde_json::json;

    #[test]
    fn prefix_truncation_is_consistent() {
        let p4 = params(&[("b", json!(4))]);
        let q4 = HierarchicalShell::build(&p4, 1, 64).unwrap();

        let p_trunc = params(&[("b", json!(4)), ("truncate_to", json!(1))]);
        let q_trunc = HierarchicalShell::build(&p_trunc, 1, 64).unwrap();

        let v = Array2::<f32>::ones((10, 64));
        let m4 = q4.fit(v.view(), None);
        let c4 = q4.encode(&m4, v.view());

        let c_trunc = q_trunc.encode(&m4, v.view());
        assert_eq!(c_trunc[0].len(), 8); // 64 / 8 = 8 bytes for 1 b/d
        assert_eq!(c4[0].len(), 32); // 4 * 8 = 32 bytes for 4 b/d
        assert_eq!(&c4[0][..8], &c_trunc[0][..]);
    }
}
