//! `colbertv2_quant`: ColBERTv2 Corpus-Level Centroid + Residual Quantizer.
//! -
//! Center -> Kmeans(centroids) -> CastNormal(residual_bits, Unbiased)
//!
//! Bit Accounting per vector (d=128, K=1024):
//! - Kmeans Centroid Index: ceil(log2 1024) = 10 bits
//! - Residual Levels: b * 128 bits
//! - Residual Scale: 32 bits (4 bytes f32)
//!
//! Total bits = 10 + 128*b + 32 (1.33 b/d at b=1, 2.33 b/d at b=2).

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use crate::coding::CodeLayout;
use crate::{CastNormal, Center, Kmeans, NormalScale, Params, Pipeline, Quantizer};

pub struct Colbertv2Quant(pub Pipeline);

impl Colbertv2Quant {
    pub fn pipeline(centroids: usize, residual_bits: u8, seed: u64, dim: usize) -> Result<Pipeline> {
        ensure!(
            (2..=256).contains(&centroids),
            "centroids must be in 2..=256, got {centroids}"
        );
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&residual_bits),
            "residual_bits must be in 1..={}, got {residual_bits}",
            CodeLayout::MAX_BITS
        );

        Pipeline::new(
            dim,
            vec![
                Box::new(Center),
                Box::new(Kmeans::new(centroids, seed)),
                Box::new(CastNormal::new(residual_bits, NormalScale::Unbiased)),
            ],
        )
    }
}

impl Quantizer for Colbertv2Quant {
    fn name() -> &'static str {
        "colbertv2_quant"
    }

    fn display_name() -> &'static str {
        "ColBERTv2"
    }

    fn params() -> &'static [&'static str] {
        &["centroids", "residual_bits"]
    }

    fn describe() -> &'static str {
        "Center -> Kmeans(centroids) -> CastNormal(residual_bits, unbiased)"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let centroids = get_or(p, "centroids", 256usize)?;
        let residual_bits = get(p, "residual_bits")?;
        Ok(Self(Self::pipeline(centroids, residual_bits, seed, dim)?))
    }

    crate::pipeline_quantizer!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math;
    use crate::util::testing::{params, refs};
    use ndarray::Array2;
    use serde_json::json;

    #[test]
    fn test_colbertv2_quant_pipeline() {
        let mut rng = math::seed(42);
        let d = 32;
        let v: Array2<f32> = math::gaussian(&mut rng, (60, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (5, d));

        for r_bits in [1, 2] {
            let p = params(&[("centroids", json!(16)), ("residual_bits", json!(r_bits))]);
            let codec = Colbertv2Quant::build(&p, 1, d).unwrap();
            let model = codec.fit(v.view(), None);
            let codes = codec.encode(&model, v.view());
            let est = codec.score(&model, q.view(), &refs(&codes));
            let exact = q.dot(&v.t());
            let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
            let st: f32 = exact.iter().map(|t| t * t).sum();
            assert!(((se / st) - 1.0).abs() < 0.45, "residual_bits {r_bits} slope {}", se / st);
        }
    }
}
