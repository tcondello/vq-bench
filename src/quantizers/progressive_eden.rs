//! `progressive_eden`: Hierarchical Progressive Bit-Plane Residuals on EDEN-prod.
//! -
//! Encodes vectors into progressive bit-plane stages (1b + 1b + 2b = 4 b/d total)
//! allowing a single 4-bit index to be dynamically truncated to 2 b/d and 1 b/d
//! at query time.

use anyhow::{ensure, Result};
use ndarray::{Array2, ArrayView2};

use super::catalog::get_or;
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{
    CastNormal, Center, Normalize, NormalScale, Params, Pipeline, Primitive, Quantizer,
};

/// The `progressive_eden` family.
pub struct ProgressiveEden {
    eval_bits: u8,
    dim: usize,
    full_pipe: Pipeline,
    pipe_2b: Pipeline,
    pipe_1b: Pipeline,
}

impl ProgressiveEden {
    pub fn new(eval_bits: u8, rotation: Rotation, seed: u64, dim: usize) -> Result<Self> {
        ensure!(
            eval_bits == 1 || eval_bits == 2 || eval_bits == 4,
            "eval_bits must be 1, 2, or 4; got {eval_bits}"
        );

        let s1 = Box::new(CastNormal::new(1, NormalScale::Unbiased));
        let s2 = Box::new(CastNormal::new(1, NormalScale::Unbiased));
        let s3 = Box::new(CastNormal::new(2, NormalScale::Unbiased));

        let full_pipe = Pipeline::new(
            dim,
            vec![
                Box::new(Center),
                Box::new(Normalize),
                rotation.stage(seed),
                s1,
                s2,
                s3,
            ],
        )?;

        let s1_b = Box::new(CastNormal::new(1, NormalScale::Unbiased));
        let s2_b = Box::new(CastNormal::new(1, NormalScale::Unbiased));
        let pipe_2b = Pipeline::new(
            dim,
            vec![
                Box::new(Center),
                Box::new(Normalize),
                rotation.stage(seed),
                s1_b,
                s2_b,
            ],
        )?;

        let s1_c = Box::new(CastNormal::new(1, NormalScale::Unbiased));
        let pipe_1b = Pipeline::new(
            dim,
            vec![
                Box::new(Center),
                Box::new(Normalize),
                rotation.stage(seed),
                s1_c,
            ],
        )?;

        Ok(Self {
            eval_bits,
            dim,
            full_pipe,
            pipe_2b,
            pipe_1b,
        })
    }

    fn norm_bytes(&self) -> usize {
        CodeLayout::new().scalars(1).byte_len()
    }

    fn s1_bytes(&self) -> usize {
        CodeLayout::new().bits(self.dim, 1).scalars(1).byte_len()
    }

    fn s2_bytes(&self) -> usize {
        CodeLayout::new().bits(self.dim, 1).scalars(1).byte_len()
    }

    fn prefix_bytes(&self, bits: u8) -> usize {
        match bits {
            1 => self.norm_bytes() + self.s1_bytes(),
            2 => self.norm_bytes() + self.s1_bytes() + self.s2_bytes(),
            _ => usize::MAX,
        }
    }
}

impl Quantizer for ProgressiveEden {
    fn name() -> &'static str {
        "progressive_eden"
    }

    fn display_name() -> &'static str {
        "ProgressiveEDEN"
    }

    fn params() -> &'static [&'static str] {
        &["eval_bits", "rotation"]
    }

    fn describe() -> &'static str {
        "Progressive bit-plane residual EDEN (1b + 1b + 2b = 4 b/d, truncatable to 1b, 2b, 4b)"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let eval_bits = get_or(p, "eval_bits", 4)?;
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;
        Self::new(eval_bits, rotation, seed, dim)
    }

    fn fit(&self, vectors: ArrayView2<f32>, queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        self.full_pipe.fit(vectors, queries)
    }

    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        self.full_pipe.encode(model, vectors)
    }

    fn reconstruct(&self, model: &[u8], codes: &[&[u8]]) -> Array2<f32> {
        if self.eval_bits == 4 {
            self.full_pipe.reconstruct(model, codes, None)
        } else if self.eval_bits == 2 {
            let pfx = self.prefix_bytes(2);
            let truncated: Vec<&[u8]> = codes.iter().map(|c| &c[..pfx.min(c.len())]).collect();
            self.pipe_2b.reconstruct(model, &truncated, None)
        } else {
            let pfx = self.prefix_bytes(1);
            let truncated: Vec<&[u8]> = codes.iter().map(|c| &c[..pfx.min(c.len())]).collect();
            self.pipe_1b.reconstruct(model, &truncated, None)
        }
    }

    fn score(&self, model: &[u8], queries: ArrayView2<f32>, codes: &[&[u8]]) -> Array2<f32> {
        if self.eval_bits == 4 {
            self.full_pipe.score(model, queries, codes, None)
        } else if self.eval_bits == 2 {
            let pfx = self.prefix_bytes(2);
            let truncated: Vec<&[u8]> = codes.iter().map(|c| &c[..pfx.min(c.len())]).collect();
            self.pipe_2b.score(model, queries, &truncated, None)
        } else {
            let pfx = self.prefix_bytes(1);
            let truncated: Vec<&[u8]> = codes.iter().map(|c| &c[..pfx.min(c.len())]).collect();
            self.pipe_1b.score(model, queries, &truncated, None)
        }
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
    fn test_progressive_eden_eval_rates() {
        let mut rng = math::seed(42);
        let d = 64;
        let v: Array2<f32> = math::gaussian(&mut rng, (40, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (5, d));

        for eval_b in [1, 2, 4] {
            let p = params(&[("eval_bits", json!(eval_b))]);
            let codec = ProgressiveEden::build(&p, 1, d).unwrap();
            let model = codec.fit(v.view(), None);
            let codes = codec.encode(&model, v.view());
            let est = codec.score(&model, q.view(), &refs(&codes));
            let exact = q.dot(&v.t());
            let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
            let st: f32 = exact.iter().map(|t| t * t).sum();
            assert!(((se / st) - 1.0).abs() < 0.45, "eval_bits {eval_b} slope {}", se / st);
        }
    }
}
