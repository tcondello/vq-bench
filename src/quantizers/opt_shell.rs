//! `opt_shell`: ITQ Learned Optimal Rotation with Spherical Shell Lattice Quantization.
//! -
//! Learns an orthogonal Procrustes rotation minimizing angular quantization distortion,
//! projecting onto the 240 minimal root vectors of the E8 Gosset lattice on S^7 with QJL residual.

use anyhow::{ensure, Result};

use super::catalog::get_or;
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::{
    CastShellE8, Center, Normalize, OptimizeSigns, Params, Pipeline, Quantizer,
};

/// Independent seed offset for residual rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0x0975;

/// The `opt_shell` family.
pub struct OptShell(pub Pipeline);

impl OptShell {
    /// Pipeline for OptShell:
    /// `Center -> Normalize -> OptimizeSigns -> CastShellE8` (with optional QJL residual).
    pub fn pipeline(
        residual: bool,
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(dim.is_multiple_of(8), "dim must be a multiple of 8, got {dim}");

        if residual {
            let res_seed = seed ^ RESIDUAL_ROTATION_SEED;
            let qjl = Qjl::pipeline(1.0, rotation, res_seed, dim)?;
            Pipeline::new(
                dim,
                vec![
                    Box::new(Center),
                    Box::new(Normalize),
                    Box::new(OptimizeSigns::new(seed)),
                    Box::new(CastShellE8::new()),
                    Box::new(qjl),
                ],
            )
        } else {
            Pipeline::new(
                dim,
                vec![
                    Box::new(Center),
                    Box::new(Normalize),
                    Box::new(OptimizeSigns::new(seed)),
                    Box::new(CastShellE8::new()),
                ],
            )
        }
    }
}

impl Quantizer for OptShell {
    fn name() -> &'static str {
        "opt_shell"
    }

    fn display_name() -> &'static str {
        "OptShell"
    }

    fn params() -> &'static [&'static str] {
        &["residual", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> OptimizeSigns -> CastShellE8 [Learned Rotation + S^7 Roots]"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let residual = get_or(p, "residual", true)?;
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;

        Ok(Self(Self::pipeline(residual, rotation, seed, dim)?))
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
        let d = 16;
        let v: Array2<f32> = math::gaussian(&mut rng, (60, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (6, d));

        let p = params(&[("residual", json!(false))]);
        let codec = OptShell::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
