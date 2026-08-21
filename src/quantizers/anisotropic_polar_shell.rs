//! `anisotropic_polar_shell`: Anisotropic Coordinate Whitened 8D Gosset Sphere Quantization.
//! -
//! Normalizes per-dimension dynamic ranges via MinMaxDim before unit-sphere normalization
//! and 8D Gosset E8 lattice shell projection with QJL residual refinement.

use anyhow::{ensure, Result};

use super::catalog::get_or;
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::{
    CastShellE8, CastUint, MinMax, MinMaxDim, Normalize, Params, Pipeline, Primitive, Quantizer,
    Resize, SpikeSplit, Split,
};

/// The `anisotropic_polar_shell` family.
pub struct AnisotropicPolarShell(pub Pipeline);

impl AnisotropicPolarShell {
    /// Pipeline for AnisotropicPolarShell:
    /// `SpikeSplit(ratio)`:
    /// - Branch 0 (Outliers): `MinMax -> CastUint(8)`
    /// - Branch 1 (Bulk): `MinMaxDim(-1, 1) -> Normalize -> Rotate(Hadamard) -> CastShellE8 -> Qjl(1.0)`
    pub fn pipeline(
        residual: bool,
        ratio: f32,
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(ratio > 0.0 && ratio < 0.5, "ratio must be in (0.0, 0.5), got {ratio}");

        let splitter = SpikeSplit::new(ratio);

        let split = Split::from_factory(splitter, move |branch, branch_dim| {
            if branch == 0 {
                Pipeline::new(
                    branch_dim,
                    vec![
                        Box::new(MinMax::default()) as Box<dyn Primitive>,
                        Box::new(CastUint::new(8)),
                    ],
                )
                .expect("valid outlier pipeline")
            } else {
                let wide = branch_dim.div_ceil(8) * 8;
                let rot = rotation.stage(seed.wrapping_add(branch as u64));
                let mut stages: Vec<Box<dyn Primitive>> = vec![
                    Box::new(MinMaxDim::new(-1.0, 1.0)),
                    Box::new(Normalize),
                    rot,
                ];
                if wide != branch_dim {
                    stages.push(Box::new(Resize::to(wide)));
                }
                stages.push(Box::new(CastShellE8::new()));
                if residual {
                    let qjl = Qjl::pipeline(1.0, rotation, seed ^ 0x901A8, wide)
                        .expect("valid qjl residual");
                    stages.push(Box::new(qjl));
                }
                Pipeline::new(branch_dim, stages).expect("valid bulk shell pipeline")
            }
        });

        Pipeline::new(dim, vec![Box::new(split)])
    }
}

impl Quantizer for AnisotropicPolarShell {
    fn name() -> &'static str {
        "anisotropic_polar_shell"
    }

    fn display_name() -> &'static str {
        "AnisotropicPolarShell"
    }

    fn params() -> &'static [&'static str] {
        &["residual", "ratio", "rotation"]
    }

    fn describe() -> &'static str {
        "SpikeSplit -> [MinMax(8), MinMaxDim -> Normalize -> Rotate -> CastShellE8 -> Qjl(1.0)]"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let residual = get_or(p, "residual", true)?;
        let ratio = get_or(p, "ratio", 0.05)?;
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;

        Ok(Self(Self::pipeline(residual, ratio, rotation, seed, dim)?))
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
        let mut v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        v.column_mut(0).mapv_inplace(|x| 20.0 * x);
        v.column_mut(1).mapv_inplace(|x| 15.0 * x);
        let q: Array2<f32> = math::gaussian(&mut rng, (8, d));

        let p = params(&[("residual", json!(true)), ("ratio", json!(0.0625))]);
        let codec = AnisotropicPolarShell::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
