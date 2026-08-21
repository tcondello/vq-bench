//! `cascade_shell_eden`: Coarse-to-Fine Gosset Shell & Directional Gaussian Cascade.
//! -
//! Stages coarse 8D Gosset spherical lattice quantization followed by directional Gaussian
//! Lloyd-Max residual refinement and orthogonal 1-bit QJL unbiasing.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{
    CastMultiShellE8, CastNormal, Center, Normalize, NormalScale, Params, Pipeline, Primitive,
    Quantizer, Resize,
};

/// Independent seed offset for residual rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0xCE711;

/// The `cascade_shell_eden` family.
pub struct CascadeShellEden(pub Pipeline);

impl CascadeShellEden {
    /// Pipeline for CascadeShellEden:
    /// `Center -> Normalize -> Rotate -> CastMultiShellE8(b-1) -> CastNormal(1, Plain) -> Qjl(1.0)`
    pub fn pipeline(bits: u8, rotation: Rotation, seed: u64, dim: usize) -> Result<Pipeline> {
        ensure!(
            (2..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 2..={}, got {bits}",
            CodeLayout::MAX_BITS
        );

        let wide = dim.div_ceil(8) * 8;
        let rot_stage = rotation.stage(seed);
        let rot_dim = rot_stage.out_dim(wide);

        let res_seed = seed ^ RESIDUAL_ROTATION_SEED;
        let qjl = Qjl::pipeline(1.0, rotation, res_seed, dim)?;

        let mut stages: Vec<Box<dyn Primitive>> = vec![
            Box::new(Center),
            Box::new(Normalize),
        ];

        if wide != dim {
            stages.push(Box::new(Resize::to(wide)));
        }
        stages.push(rot_stage);
        if rot_dim != wide {
            stages.push(Box::new(Resize::to(wide)));
        }

        let coarse_bits = bits.saturating_sub(1).max(1);
        if wide.is_multiple_of(8) {
            stages.push(Box::new(CastMultiShellE8::new(coarse_bits)));
        } else {
            stages.push(Box::new(CastNormal::new(coarse_bits, NormalScale::Plain)));
        }

        if wide != dim {
            stages.push(Box::new(Resize::to(dim)));
        }

        stages.push(Box::new(CastNormal::new(1, NormalScale::Plain)));
        stages.push(Box::new(qjl));

        Pipeline::new(dim, stages)
    }
}

impl Quantizer for CascadeShellEden {
    fn name() -> &'static str {
        "cascade_shell_eden"
    }

    fn display_name() -> &'static str {
        "CascadeShellEden"
    }

    fn params() -> &'static [&'static str] {
        &["b", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> Rotate -> CastMultiShellE8(b-1) -> CastNormal(1) -> QJL(1.0)"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let b = get(p, "b")?;
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;

        Ok(Self(Self::pipeline(b, rotation, seed, dim)?))
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
        let v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (8, d));

        let p = params(&[("b", json!(4))]);
        let codec = CascadeShellEden::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.25, "slope {}", se / st);
    }
}
