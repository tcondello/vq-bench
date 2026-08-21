//! `polar_leech_eden`: Polar Leech Lattice Anchor with Multi-Bit Gaussian Lloyd-Max Residual.
//! -
//! Decouples unit spherical direction into 24-dimensional Leech lattice Lambda24 anchors
//! and quantizes the spherical residual with multi-bit directional Gaussian Lloyd-Max.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{
    CastLeech24, CastNormal, Center, Normalize, NormalScale, Params, Pipeline, Primitive,
    Quantizer, Resize,
};

/// The `polar_leech_eden` family.
pub struct PolarLeechEden(pub Pipeline);

impl PolarLeechEden {
    /// Pipeline for PolarLeechEden:
    /// `Center -> Normalize -> Rotate -> CastLeech24(b_head) -> CastNormal(b_tail)`
    pub fn pipeline(
        bits: u8,
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 1..={}, got {bits}",
            CodeLayout::MAX_BITS
        );

        let wide = dim.div_ceil(24) * 24;
        let mut stages: Vec<Box<dyn Primitive>> = vec![
            Box::new(Center),
            Box::new(Normalize),
        ];

        if wide != dim {
            stages.push(Box::new(Resize::to(wide)));
        }

        let rot_stage = rotation.stage(seed);
        let rot_dim = rot_stage.out_dim(wide);
        stages.push(rot_stage);
        if rot_dim != wide {
            stages.push(Box::new(Resize::to(wide)));
        }

        if bits == 1 {
            stages.push(Box::new(CastLeech24::new(1)));
        } else {
            let b_head = (bits / 2).max(1);
            let b_tail = bits - b_head;
            stages.push(Box::new(CastLeech24::new(b_head)));
            stages.push(Box::new(CastNormal::new(b_tail, NormalScale::Unbiased)));
        }

        Pipeline::new(dim, stages)
    }
}

impl Quantizer for PolarLeechEden {
    fn name() -> &'static str {
        "polar_leech_eden"
    }

    fn display_name() -> &'static str {
        "PolarLeechEden"
    }

    fn params() -> &'static [&'static str] {
        &["b", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> Rotate -> CastLeech24(b_head) -> CastNormal(b_tail, Directional)"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let bits = get(p, "b")?;
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;

        Ok(Self(Self::pipeline(bits, rotation, seed, dim)?))
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
        let d = 48;
        let v: Array2<f32> = math::gaussian(&mut rng, (60, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (6, d));

        let p = params(&[("b", json!(4))]);
        let codec = PolarLeechEden::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
