//! `e8quant`: 8-dimensional Gosset lattice sphere-packing vector quantization.
//! -
//! Rotates vectors onto an isotropic sphere and projects 8-dimensional subvectors
//! onto the E8 root lattice using the Conway-Sloane constant-time nearest point algorithm.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{CastE8, Center, Normalize, Params, Pipeline, Quantizer};

/// Independent seed offset for residual rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0xE859A;

/// The `e8quant` family.
pub struct E8Quant(pub Pipeline);

impl E8Quant {
    /// Pipeline for E8 lattice quantization:
    /// `Center -> Normalize -> Rotation -> CastE8(b)` (with optional QJL residual at b >= 3).
    pub fn pipeline(
        bits: u8,
        residual: bool,
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(
            (2..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 2..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
        ensure!(dim.is_multiple_of(8), "dim must be a multiple of 8, got {dim}");

        let stage = rotation.stage(seed);
        let mid_dim = stage.out_dim(dim);

        if residual && bits >= 3 {
            let res_seed = seed ^ RESIDUAL_ROTATION_SEED;
            let qjl = Qjl::pipeline(1.0, rotation, res_seed, mid_dim)?;
            Pipeline::new(
                dim,
                vec![
                    Box::new(Center),
                    Box::new(Normalize),
                    stage,
                    Box::new(CastE8::new(bits - 1)),
                    Box::new(qjl),
                ],
            )
        } else {
            Pipeline::new(
                dim,
                vec![
                    Box::new(Center),
                    Box::new(Normalize),
                    stage,
                    Box::new(CastE8::new(bits)),
                ],
            )
        }
    }
}

impl Quantizer for E8Quant {
    fn name() -> &'static str {
        "e8quant"
    }

    fn display_name() -> &'static str {
        "E8Quant"
    }

    fn params() -> &'static [&'static str] {
        &["b", "residual", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> Rotate -> CastE8(b)"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let b = get(p, "b")?;
        let residual = get_or(p, "residual", false)?;
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;
        Ok(Self(Self::pipeline(b, residual, rotation, seed, dim)?))
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

    fn make_test_e8quant(
        bits: u8,
        residual: bool,
        rotation: &str,
        seed: u64,
        dim: usize,
    ) -> Result<E8Quant> {
        let p = params(&[
            ("b", json!(bits)),
            ("residual", json!(residual)),
            ("rotation", json!(rotation)),
        ]);
        E8Quant::build(&p, seed, dim)
    }

    #[test]
    fn rejects_invalid_params() {
        assert!(make_test_e8quant(1, false, "hadamard", 1, 16).is_err());
        assert!(make_test_e8quant(9, false, "hadamard", 1, 16).is_err());
        assert!(make_test_e8quant(4, false, "hadamard", 1, 15).is_err()); // not multiple of 8
        assert!(make_test_e8quant(4, false, "hadamard", 1, 16).is_ok());
    }

    #[test]
    fn round_trip_and_score() {
        let mut rng = math::seed(42);
        let d = 32;
        let v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (8, d));

        for residual in [false, true] {
            let codec = make_test_e8quant(4, residual, "hadamard", 1, d).unwrap();
            let model = codec.fit(v.view(), None);
            let codes = codec.encode(&model, v.view());
            let est = codec.score(&model, q.view(), &refs(&codes));
            let exact = q.dot(&v.t());
            let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
            let st: f32 = exact.iter().map(|t| t * t).sum();
            assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
        }
    }
}
