//! `bitnet_shell`: Native AbsMean Ternary with Rotated Spherical Shell E8 Residual.
//! -
//! Quantizes activations to ternary {-1, 0, +1} in native space with BitNet AbsMean scaling,
//! rotates the continuous residual, and quantizes across the 240 minimal roots of E8 on S^7.

use anyhow::{ensure, Result};

use super::catalog::get_or;
use super::rotation::Rotation;
use crate::{CastMultiShellE8, CastTernary, Center, Normalize, Params, Pipeline, Quantizer};

/// Independent seed offset for residual rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0xB175E;

/// The `bitnet_shell` family.
pub struct BitNetShell(pub Pipeline);

impl BitNetShell {
    /// Pipeline for BitNetShell:
    /// `Center -> Normalize -> Rotation -> CastTernary -> CastMultiShellE8(1)`
    pub fn pipeline(
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(dim.is_multiple_of(8), "dim must be a multiple of 8, got {dim}");

        let stage = rotation.stage(seed);
        let res_seed = seed ^ RESIDUAL_ROTATION_SEED;
        let res_stage = rotation.stage(res_seed);

        Pipeline::new(
            dim,
            vec![
                Box::new(Center),
                Box::new(Normalize),
                stage,
                Box::new(CastTernary),
                res_stage,
                Box::new(CastMultiShellE8::new(1)),
            ],
        )
    }
}

impl Quantizer for BitNetShell {
    fn name() -> &'static str {
        "bitnet_shell"
    }

    fn display_name() -> &'static str {
        "BitNetShell"
    }

    fn params() -> &'static [&'static str] {
        &["rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> CastTernary -> Rotate -> CastShellE8 [Native Ternary + S^7 Lattice]"
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
        let codec = BitNetShell::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
