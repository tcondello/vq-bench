//! `leech24`: 24-Dimensional Leech Lattice (Lambda24) Quantization.
//! -
//! Quantizes 24-dimensional blocks onto the Leech lattice Lambda24 (optimal sphere
//! packing in 24D with kissing number 196,560) using fast Golay G24 syndrome decoding.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{CastLeech24, Center, Normalize, Params, Pipeline, Primitive, Quantizer, Resize};

/// Independent seed offset for shape residual rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0x24EE;

/// The `leech24` family.
pub struct Leech24(pub Pipeline);

impl Leech24 {
    /// Pipeline for Leech24:
    /// `Center -> Normalize -> Rotation -> (Resize) -> CastLeech24(b)` (with optional QJL residual).
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
        ensure!(dim.is_multiple_of(24), "dim must be a multiple of 24, got {dim}");

        let mut stages: Vec<Box<dyn Primitive>> = vec![
            Box::new(Center),
            Box::new(Normalize),
        ];

        let rot_stage = rotation.stage(seed);
        let rot_dim = rot_stage.out_dim(dim);
        stages.push(rot_stage);
        if rot_dim != dim {
            stages.push(Box::new(Resize::to(dim)));
        }

        if residual && bits >= 2 {
            let res_seed = seed ^ RESIDUAL_ROTATION_SEED;
            let qjl = Qjl::pipeline(1.0, rotation, res_seed, dim)?;
            stages.push(Box::new(CastLeech24::new(bits - 1)));
            stages.push(Box::new(qjl));
        } else {
            stages.push(Box::new(CastLeech24::new(bits)));
        }

        Pipeline::new(dim, stages)
    }
}

impl Quantizer for Leech24 {
    fn name() -> &'static str {
        "leech24"
    }

    fn display_name() -> &'static str {
        "Leech24"
    }

    fn params() -> &'static [&'static str] {
        &["b", "residual", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> Rotate -> CastLeech24(b) [24D Leech Lattice Lambda24]"
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
        let d = 48;
        let v: Array2<f32> = math::gaussian(&mut rng, (60, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (6, d));

        let p = params(&[("b", json!(4)), ("residual", json!(true))]);
        let codec = Leech24::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
