//! `multi_shell_e8`: Concentric Multi-Shell E8 Gosset Lattice Quantization.
//! -
//! Quantizes high-dimensional embeddings across concentric shells of the E8 Gosset
//! lattice with radial scale adaptation, supporting variable bit budgets (b=1..8).

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{CastMultiShellE8, Center, Normalize, Params, Pipeline, Quantizer};

/// Independent seed offset for shape residual rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0x88EE8;

/// The `multi_shell_e8` family.
pub struct MultiShellE8(pub Pipeline);

impl MultiShellE8 {
    /// Pipeline for MultiShellE8:
    /// `Center -> Normalize -> Rotation -> CastMultiShellE8(b)` (with optional QJL residual).
    pub fn pipeline(
        bits: u8,
        residual: bool,
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 1..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
        ensure!(dim.is_multiple_of(8), "dim must be a multiple of 8, got {dim}");

        let stage = rotation.stage(seed);
        let mid_dim = stage.out_dim(dim);

        if residual && bits >= 2 {
            let res_seed = seed ^ RESIDUAL_ROTATION_SEED;
            let qjl = Qjl::pipeline(1.0, rotation, res_seed, mid_dim)?;
            Pipeline::new(
                dim,
                vec![
                    Box::new(Center),
                    Box::new(Normalize),
                    stage,
                    Box::new(CastMultiShellE8::new(bits - 1)),
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
                    Box::new(CastMultiShellE8::new(bits)),
                ],
            )
        }
    }
}

impl Quantizer for MultiShellE8 {
    fn name() -> &'static str {
        "multi_shell_e8"
    }

    fn display_name() -> &'static str {
        "MultiShellE8"
    }

    fn params() -> &'static [&'static str] {
        &["b", "residual", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> Rotate -> CastMultiShellE8(b) [Concentric Gosset Lattice Shells]"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let bits = get(p, "b")?;
        let residual = get_or(p, "residual", true)?;
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;

        Ok(Self(Self::pipeline(bits, residual, rotation, seed, dim)?))
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

        let p = params(&[("b", json!(4)), ("residual", json!(true))]);
        let codec = MultiShellE8::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
