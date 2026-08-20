//! `bitnet`: Center, optional rotation, then 1.58-bit ternary quantization (BitNet b1.58).

use anyhow::Result;

use super::catalog::get_or;
use super::rotation::Rotation;
use crate::{CastTernary, Center, Params, Pipeline, Quantizer};

/// The `bitnet` family: Microsoft BitNet b1.58 AbsMean ternary quantization.
pub struct BitNetQuant(pub Pipeline);

impl BitNetQuant {
    /// `Center -> rotation(seed) -> CastTernary` over input dimension `dim`.
    pub fn pipeline(rotation: Rotation, seed: u64, dim: usize) -> Result<Pipeline> {
        Pipeline::new(
            dim,
            vec![
                Box::new(Center),
                rotation.stage(seed),
                Box::new(CastTernary),
            ],
        )
    }
}

impl Quantizer for BitNetQuant {
    fn name() -> &'static str {
        "bitnet"
    }

    fn display_name() -> &'static str {
        "BitNet"
    }

    fn params() -> &'static [&'static str] {
        &["rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Rotate -> CastTernary(absmean)"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;
        Ok(Self(Self::pipeline(rotation, seed, dim)?))
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
        let d = 64;
        let v: Array2<f32> = math::gaussian(&mut rng, (60, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (6, d));

        for rot in ["hadamard", "full"] {
            let p = params(&[("rotation", json!(rot))]);
            let codec = BitNetQuant::build(&p, 1, d).unwrap();
            let model = codec.fit(v.view(), None);
            let codes = codec.encode(&model, v.view());
            assert_eq!(codes.len(), 60);

            let recons = codec.reconstruct(&model, &refs(&codes));
            assert_eq!(recons.dim(), (60, d));

            let scores = codec.score(&model, q.view(), &refs(&codes));
            assert_eq!(scores.dim(), (6, 60));
        }
    }
}
