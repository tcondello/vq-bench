//! `shell_e8`: Spherical Root Shell Lattice Quantization.
//! -
//! Quantizes high-dimensional unit-norm spherical embeddings onto the 240 minimal
//! root vectors of the E8 Gosset lattice on S^7 (1.00 bit/dim base codebook in a single
//! byte per 8D subvector), with optional orthogonal QJL residual refinement.

use anyhow::{ensure, Result};

use super::catalog::get_or;
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::{CastShellE8, Center, Normalize, Params, Pipeline, Quantizer};

/// Independent seed offset for shape residual rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0x54E11;

/// The `shell_e8` family.
pub struct ShellE8(pub Pipeline);

impl ShellE8 {
    /// Pipeline for ShellE8:
    /// `Center -> Normalize -> Rotation -> CastShellE8` (with optional QJL residual).
    pub fn pipeline(
        residual: bool,
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(dim.is_multiple_of(8), "dim must be a multiple of 8, got {dim}");

        let stage = rotation.stage(seed);
        let mid_dim = stage.out_dim(dim);

        if residual {
            let res_seed = seed ^ RESIDUAL_ROTATION_SEED;
            let qjl = Qjl::pipeline(1.0, rotation, res_seed, mid_dim)?;
            Pipeline::new(
                dim,
                vec![
                    Box::new(Center),
                    Box::new(Normalize),
                    stage,
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
                    stage,
                    Box::new(CastShellE8::new()),
                ],
            )
        }
    }
}

impl Quantizer for ShellE8 {
    fn name() -> &'static str {
        "shell_e8"
    }

    fn display_name() -> &'static str {
        "ShellE8"
    }

    fn params() -> &'static [&'static str] {
        &["residual", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> Rotate -> CastShellE8 [240 Minimal E8 Roots on S^7]"
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
        let d = 32;
        let v: Array2<f32> = math::gaussian(&mut rng, (60, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (6, d));

        let p = params(&[("residual", json!(true))]);
        let codec = ShellE8::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.40, "slope {}", se / st);
    }
}
