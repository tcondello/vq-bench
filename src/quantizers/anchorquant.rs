//! `anchorquant`: Anchor-Residual KV Cache Compression (AnchorKV).
//! -
//! Represents vectors using a codebook of k unit-norm anchor directions, encoding
//! each vector by its closest anchor index and scalar projection gain. Vectors
//! with the largest residual errors selectively receive a 2-bit Lloyd-Max residual.

use anyhow::{ensure, Result};
use ndarray::{Array2, ArrayView2};

use super::catalog::get_or;
use super::rotation::Rotation;
use crate::util::codebooks::{lloyd_max_normal, nearest};
use crate::util::coding::{self, ModelField};
use crate::{math, Params, Quantizer};

/// The `anchorquant` family.
pub struct AnchorQuant {
    k: usize,
    b_res: u8,
    res_ratio: f32,
    rotation: Rotation,
    iters: usize,
    seed: u64,
}

impl AnchorQuant {
    pub fn new(
        k: usize,
        b_res: u8,
        res_ratio: f32,
        rotation: Rotation,
        iters: usize,
        seed: u64,
        dim: usize,
    ) -> Result<Self> {
        ensure!((1..=256).contains(&k), "k must be in 1..=256, got {k}");
        ensure!(b_res == 0 || b_res == 2, "b_res must be 0 or 2, got {b_res}");
        ensure!(
            (0.0..=1.0).contains(&res_ratio),
            "res_ratio must be in 0.0..=1.0, got {res_ratio}"
        );
        ensure!(dim > 0, "dim must be > 0");
        Ok(Self {
            k,
            b_res,
            res_ratio,
            rotation,
            iters,
            seed,
        })
    }
}

/// Serialized model fields for AnchorQuant.
struct AnchorModel {
    dim: usize,
    k: usize,
    b_res: u8,
    res_ratio: f32,
    anchors: Array2<f32>,
    rotation_model: Vec<u8>,
}

impl ModelField for AnchorModel {
    fn write(&self, buf: &mut Vec<u8>) {
        self.dim.write(buf);
        self.k.write(buf);
        (self.b_res as u32).write(buf);
        self.res_ratio.write(buf);
        self.anchors.write(buf);
        self.rotation_model.write(buf);
    }

    fn read(cur: &mut &[u8]) -> Self {
        let dim = usize::read(cur);
        let k = usize::read(cur);
        let b_res = u32::read(cur) as u8;
        let res_ratio = f32::read(cur);
        let anchors = Array2::<f32>::read(cur);
        let rotation_model = Vec::<u8>::read(cur);
        Self {
            dim,
            k,
            b_res,
            res_ratio,
            anchors,
            rotation_model,
        }
    }
}

/// Bitpack 2-bit values into bytes.
fn pack_2bit(values: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len().div_ceil(4));
    for chunk in values.chunks(4) {
        let mut b = 0u8;
        for (i, &v) in chunk.iter().enumerate() {
            b |= (v & 0x03) << (i * 2);
        }
        out.push(b);
    }
    out
}

/// Unpack 2-bit values from bytes.
fn unpack_2bit(bytes: &[u8], count: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(count);
    for &b in bytes {
        for i in 0..4 {
            if out.len() < count {
                out.push((b >> (i * 2)) & 0x03);
            }
        }
    }
    out
}

impl Quantizer for AnchorQuant {
    fn name() -> &'static str {
        "anchorquant"
    }

    fn display_name() -> &'static str {
        "AnchorQuant"
    }

    fn params() -> &'static [&'static str] {
        &["k", "b_res", "res_ratio", "rotation", "iters"]
    }

    fn describe() -> &'static str {
        "k unit-norm anchor directions with scalar gain + selective 2-bit Lloyd-Max residuals"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let k: usize = get_or(p, "k", 64)?;
        let b_res: u8 = get_or(p, "b_res", 2)?;
        let res_ratio: f32 = get_or(p, "res_ratio", 0.2)?;
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;
        let iters: usize = get_or(p, "iters", 15)?;
        Self::new(k, b_res, res_ratio, rotation, iters, seed, dim)
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let dim = vectors.ncols();
        let k = self.k.min(vectors.nrows()).max(1);

        // Normalize fit vectors to unit sphere for spherical k-means
        let mut unit_v = vectors.to_owned();
        for mut row in unit_v.rows_mut() {
            let norm = row.dot(&row).sqrt();
            if norm > 1e-12 {
                row.mapv_inplace(|x| x / norm);
            }
        }

        let centroids = math::lloyd_kmeans(unit_v.view(), k, self.iters, self.seed);
        let mut anchors = centroids;
        for mut row in anchors.rows_mut() {
            let norm = row.dot(&row).sqrt();
            if norm > 1e-12 {
                row.mapv_inplace(|x| x / norm);
            }
        }

        let rot_primitive = self.rotation.stage(self.seed ^ 0x59A4);
        let rotation_model = rot_primitive.fit(vectors, None);

        let model = AnchorModel {
            dim,
            k,
            b_res: self.b_res,
            res_ratio: self.res_ratio,
            anchors,
            rotation_model,
        };
        coding::pack_model(model)
    }

    fn encode(&self, model_bytes: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let model: AnchorModel = coding::unpack_model(model_bytes);
        let n = vectors.nrows();
        let dim = model.dim;
        let rot_primitive = self.rotation.stage(self.seed ^ 0x59A4);
        let codebook = lloyd_max_normal(1 << model.b_res);

        // Compute correlations with all anchors: (n x k)
        let dots = math::matmul(vectors, model.anchors.t());

        // For each vector, find best anchor and compute projection residual error
        let mut best_anchors = Vec::with_capacity(n);
        let mut gains = Vec::with_capacity(n);
        let mut errors = Vec::with_capacity(n);

        for i in 0..n {
            let dot_row = dots.row(i);
            let mut best_a = 0usize;
            let mut max_dot = f32::NEG_INFINITY;
            for (j, &d) in dot_row.iter().enumerate() {
                if d > max_dot {
                    max_dot = d;
                    best_a = j;
                }
            }
            let gain = max_dot;
            let v_row = vectors.row(i);
            let v_norm_sq = v_row.dot(&v_row);
            let err = (v_norm_sq - gain * gain).max(0.0);

            best_anchors.push(best_a as u8);
            gains.push(gain);
            errors.push(err);
        }

        // Determine error threshold for residual allocation
        let threshold = if model.res_ratio > 0.0 && model.b_res > 0 && n > 0 {
            let mut sorted_errs = errors.clone();
            sorted_errs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let keep_count = ((n as f32 * model.res_ratio).round() as usize).clamp(1, n);
            sorted_errs[keep_count - 1]
        } else {
            f32::INFINITY
        };

        let mut codes = Vec::with_capacity(n);
        for i in 0..n {
            let best_a = best_anchors[i];
            let gain = gains[i];
            let err = errors[i];
            let mut code = Vec::new();
            code.push(best_a);
            code.extend_from_slice(&gain.to_le_bytes());

            if model.b_res > 0 && model.res_ratio > 0.0 && err >= threshold {
                code.push(1u8); // has_residual flag = 1
                let v_row = vectors.row(i);
                let anchor_row = model.anchors.row(best_a as usize);
                let mut residual = Array2::zeros((1, dim));
                for d in 0..dim {
                    residual[[0, d]] = v_row[d] - gain * anchor_row[d];
                }
                // Rotate residual
                rot_primitive.apply(&model.rotation_model, &mut residual, &[]);
                let rot_res = residual.row(0);

                // Absmax scaling
                let mut max_abs = 0.0f32;
                for &val in rot_res.iter() {
                    max_abs = max_abs.max(val.abs());
                }
                let scale = if max_abs > 1e-12 { max_abs } else { 1.0 };
                code.extend_from_slice(&scale.to_le_bytes());

                // Quantize to 2-bit Lloyd-Max levels
                let levels: Vec<u8> = rot_res
                    .iter()
                    .map(|&val| nearest(&codebook, val / scale) as u8)
                    .collect();
                let packed = pack_2bit(&levels);
                code.extend_from_slice(&packed);
            } else if model.b_res > 0 && model.res_ratio > 0.0 {
                code.push(0u8); // has_residual flag = 0
            }
            codes.push(code);
        }
        codes
    }

    fn reconstruct(&self, model_bytes: &[u8], codes: &[&[u8]]) -> Array2<f32> {
        let model: AnchorModel = coding::unpack_model(model_bytes);
        let n = codes.len();
        let dim = model.dim;
        let rot_primitive = self.rotation.stage(self.seed ^ 0x59A4);
        let padded_dim = rot_primitive.out_dim(dim);
        let codebook = lloyd_max_normal(1 << model.b_res);
        let mut recons = Array2::zeros((n, dim));

        for (i, &code) in codes.iter().enumerate() {
            let anchor_idx = code[0] as usize;
            let gain = f32::from_le_bytes(code[1..5].try_into().unwrap());
            let anchor = model.anchors.row(anchor_idx);

            for d in 0..dim {
                recons[[i, d]] = gain * anchor[d];
            }

            if code.len() > 5 && code[5] == 1 {
                let scale = f32::from_le_bytes(code[6..10].try_into().unwrap());
                let levels = unpack_2bit(&code[10..], padded_dim);
                let mut rot_res = Array2::zeros((1, padded_dim));
                for d in 0..padded_dim {
                    rot_res[[0, d]] = scale * codebook[levels[d] as usize];
                }
                let unrotated = rot_primitive.reconstruct(
                    &model.rotation_model,
                    &[],
                    Some(rot_res.view()),
                );
                for d in 0..dim {
                    recons[[i, d]] += unrotated[[0, d]];
                }
            }
        }
        recons
    }

    fn score(
        &self,
        model_bytes: &[u8],
        queries: ArrayView2<f32>,
        candidate_codes: &[&[u8]],
    ) -> Array2<f32> {
        let model: AnchorModel = coding::unpack_model(model_bytes);
        let n_q = queries.nrows();
        let n_c = candidate_codes.len();
        let dim = model.dim;
        let rot_primitive = self.rotation.stage(self.seed ^ 0x59A4);
        let padded_dim = rot_primitive.out_dim(dim);
        let codebook = lloyd_max_normal(1 << model.b_res);

        // Precompute query dots with anchors: (n_q x k)
        let q_anchors = math::matmul(queries, model.anchors.t());

        // Precompute rotated queries for fast residual scoring
        let mut rot_queries = queries.to_owned();
        rot_primitive.apply_queries(&model.rotation_model, &mut rot_queries);

        let mut scores = Array2::zeros((n_q, n_c));

        for (j, &code) in candidate_codes.iter().enumerate() {
            let anchor_idx = code[0] as usize;
            let gain = f32::from_le_bytes(code[1..5].try_into().unwrap());
            let has_res = code.len() > 5 && code[5] == 1;

            if has_res {
                let scale = f32::from_le_bytes(code[6..10].try_into().unwrap());
                let levels = unpack_2bit(&code[10..], padded_dim);
                for q_i in 0..n_q {
                    let coarse_score = gain * q_anchors[[q_i, anchor_idx]];
                    let mut res_score = 0.0f32;
                    let q_row = rot_queries.row(q_i);
                    for d in 0..padded_dim {
                        res_score += q_row[d] * (scale * codebook[levels[d] as usize]);
                    }
                    scores[[q_i, j]] = coarse_score + res_score;
                }
            } else {
                for q_i in 0..n_q {
                    scores[[q_i, j]] = gain * q_anchors[[q_i, anchor_idx]];
                }
            }
        }
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{params, refs};
    use serde_json::json;

    fn make_test_anchorquant(
        k: usize,
        b_res: u8,
        res_ratio: f32,
        seed: u64,
        dim: usize,
    ) -> Result<AnchorQuant> {
        let p = params(&[
            ("k", json!(k)),
            ("b_res", json!(b_res)),
            ("res_ratio", json!(res_ratio)),
        ]);
        AnchorQuant::build(&p, seed, dim)
    }

    #[test]
    fn rejects_out_of_range_params() {
        assert!(make_test_anchorquant(0, 2, 0.2, 1, 32).is_err());
        assert!(make_test_anchorquant(300, 2, 0.2, 1, 32).is_err());
        assert!(make_test_anchorquant(16, 3, 0.2, 1, 32).is_err());
        assert!(make_test_anchorquant(16, 2, 1.5, 1, 32).is_err());
        assert!(make_test_anchorquant(16, 2, 0.2, 1, 32).is_ok());
    }

    #[test]
    fn round_trip_and_score() {
        let mut rng = math::seed(42);
        let d = 32;
        let v = math::gaussian(&mut rng, (60, d));
        let q = math::gaussian(&mut rng, (6, d));

        let quant = make_test_anchorquant(8, 2, 0.5, 1, d).unwrap();
        let model = quant.fit(v.view(), None);
        let codes = quant.encode(&model, v.view());
        assert_eq!(codes.len(), 60);

        let recons = quant.reconstruct(&model, &refs(&codes));
        assert_eq!(recons.dim(), (60, d));

        let scores = quant.score(&model, q.view(), &refs(&codes));
        assert_eq!(scores.dim(), (6, 60));

        let exact = q.dot(&v.t());
        let se: f32 = scores.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
