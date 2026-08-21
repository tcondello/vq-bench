//! `turbo_shell`: Concentric Multi-Shell E8 Lattice with Dual Orthogonal QJL Cascade.
//! -
//! Quantizes across concentric shells of the E8 Gosset lattice with radial adaptation,
//! followed by a dual-stage orthogonal QJL random projection cascade on the residual.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{CastMultiShellE8, Center, Normalize, Params, Pipeline, Quantizer};

/// Independent seed offsets for dual residual rotations.
const RESIDUAL_ROTATION_SEED1: u64 = 0x7890E8;
const RESIDUAL_ROTATION_SEED2: u64 = 0x7890E9;

/// The `turbo_shell` family.
pub struct TurboShell(pub Pipeline);

impl TurboShell {
    /// Pipeline for TurboShell:
    /// `Center -> Normalize -> Rotation -> CastMultiShellE8(b) -> Qjl(1.0) -> Qjl(1.0)`
    pub fn pipeline(
        bits: u8,
        cascade: bool,
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

        if cascade && bits >= 2 {
            let res_seed1 = seed ^ RESIDUAL_ROTATION_SEED1;
            let res_seed2 = seed ^ RESIDUAL_ROTATION_SEED2;
            let qjl1 = Qjl::pipeline(1.0, rotation, res_seed1, mid_dim)?;
            let qjl2 = Qjl::pipeline(1.0, rotation, res_seed2, mid_dim)?;

            Pipeline::new(
                dim,
                vec![
                    Box::new(Center),
                    Box::new(Normalize),
                    stage,
                    Box::new(CastMultiShellE8::new(bits.saturating_sub(2).max(1))),
                    Box::new(qjl1),
                    Box::new(qjl2),
                ],
            )
        } else {
            let res_seed1 = seed ^ RESIDUAL_ROTATION_SEED1;
            let qjl1 = Qjl::pipeline(1.0, rotation, res_seed1, mid_dim)?;
            Pipeline::new(
                dim,
                vec![
                    Box::new(Center),
                    Box::new(Normalize),
                    stage,
                    Box::new(CastMultiShellE8::new(bits.saturating_sub(1).max(1))),
                    Box::new(qjl1),
                ],
            )
        }
    }
}

impl Quantizer for TurboShell {
    fn name() -> &'static str {
        "turbo_shell"
    }

    fn display_name() -> &'static str {
        "TurboShell"
    }

    fn params() -> &'static [&'static str] {
        &["b", "cascade", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> Rotate -> CastMultiShellE8(b) -> [Dual QJL Cascade]"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let b = get(p, "b")?;
        let cascade = get_or(p, "cascade", true)?;
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;

        Ok(Self(Self::pipeline(b, cascade, rotation, seed, dim)?))
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

        let p = params(&[("b", json!(4)), ("cascade", json!(true))]);
        let codec = TurboShell::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
