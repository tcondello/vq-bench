//! `leech_multishell`: Concentric Multi-Shell 24D Leech Lattice with QJL Cascade.
//! -
//! Quantizes high-dimensional vectors onto concentric spherical shells of the 24-dimensional
//! Leech lattice Lambda24 with orthogonal QJL residual refinement.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{
    CastLeech24, CastNormal, Center, Normalize, NormalScale, Params, Pipeline, Primitive,
    Quantizer, Resize,
};

/// Independent seed offset for residual rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0x24EE3;

/// The `leech_multishell` family.
pub struct LeechMultiShell(pub Pipeline);

impl LeechMultiShell {
    /// Pipeline for LeechMultiShell:
    /// `Center -> Normalize -> Rotate -> CastLeech24(b) -> Qjl(1.0)`
    pub fn pipeline(bits: u8, rotation: Rotation, seed: u64, dim: usize) -> Result<Pipeline> {
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 1..={}, got {bits}",
            CodeLayout::MAX_BITS
        );

        let wide = dim.div_ceil(24) * 24;
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

        if wide.is_multiple_of(24) {
            stages.push(Box::new(CastLeech24::new(bits)));
        } else {
            stages.push(Box::new(CastNormal::new(bits, NormalScale::Plain)));
        }

        if wide != dim {
            stages.push(Box::new(Resize::to(dim)));
        }
        stages.push(Box::new(qjl));

        Pipeline::new(dim, stages)
    }
}

impl Quantizer for LeechMultiShell {
    fn name() -> &'static str {
        "leech_multishell"
    }

    fn display_name() -> &'static str {
        "LeechMultiShell"
    }

    fn params() -> &'static [&'static str] {
        &["b", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> Rotate -> CastLeech24(b) -> QJL(1.0) [Multi-Shell 24D Leech Lattice]"
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
        let d = 24;
        let v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (8, d));

        let p = params(&[("b", json!(4))]);
        let codec = LeechMultiShell::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
