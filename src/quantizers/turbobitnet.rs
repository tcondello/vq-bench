//! `turbobitnet`: AbsMean Ternary with 1-Bit QJL Residual Refinement.
//! -
//! Quantizes vectors using fast per-vector AbsMean ternary {-1, 0, +1}, then
//! refines the residual with an orthogonal 1-bit QJL projection for an unbiased
//! inner product estimate.

use anyhow::Result;

use super::catalog::get_or;
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::{CastTernary, Center, Normalize, Params, Pipeline, Quantizer};

/// Independent seed offset for residual rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0x7B17;

/// The `turbobitnet` family.
pub struct TurboBitNet(pub Pipeline);

impl TurboBitNet {
    /// Pipeline for TurboBitNet:
    /// `Center -> Normalize -> Rotation -> CastTernary -> Qjl(1.0)`
    pub fn pipeline(rotation: Rotation, seed: u64, dim: usize) -> Result<Pipeline> {
        let stage = rotation.stage(seed);
        let mid_dim = stage.out_dim(dim);
        let res_seed = seed ^ RESIDUAL_ROTATION_SEED;
        let residual = Qjl::pipeline(1.0, rotation, res_seed, mid_dim)?;

        Pipeline::new(
            dim,
            vec![
                Box::new(Center),
                Box::new(Normalize),
                stage,
                Box::new(CastTernary),
                Box::new(residual),
            ],
        )
    }
}

impl Quantizer for TurboBitNet {
    fn name() -> &'static str {
        "turbobitnet"
    }

    fn display_name() -> &'static str {
        "TurboBitNet"
    }

    fn params() -> &'static [&'static str] {
        &["rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> Rotate -> CastTernary -> QJL(1.0)"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;
        Ok(Self(Self::pipeline(rotation, seed, dim)?))
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
    fn round_trip_and_score() {
        let mut rng = math::seed(42);
        let d = 32;
        let v: Array2<f32> = math::gaussian(&mut rng, (60, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (6, d));

        let p = params(&[("rotation", json!("hadamard"))]);
        let codec = TurboBitNet::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
