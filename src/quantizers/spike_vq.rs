//! `spike_vq`: Outlier-Channel Preserving Vector Quantization.
//! -
//! Identifies high-magnitude outlier channels (e.g. 5% of dimensions), routing
//! them to high-precision quantization while rotating and quantizing the remaining
//! dense bulk with TurboQuant / QJL.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{
    CastNormal, CastUint, Center, MinMax, Normalize, NormalScale, Params, Pipeline, Primitive,
    Quantizer, SpikeSplit, Split,
};

/// Independent seed offset for bulk residual rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0x591CE;

/// The `spike_vq` family.
pub struct SpikeVQ(pub Pipeline);

impl SpikeVQ {
    /// Pipeline for SpikeVQ:
    /// `Center -> Normalize -> SpikeSplit(ratio)`
    /// - Spike branch: `MinMax -> CastUint(8)`
    /// - Bulk branch: `Rotation -> CastNormal(b - 1, Plain) -> Qjl(1.0)`
    pub fn pipeline(
        bits: u8,
        ratio: f32,
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(
            (2..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 2..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
        ensure!(ratio > 0.0 && ratio < 1.0, "ratio must be in (0.0, 1.0), got {ratio}");

        let splitter = SpikeSplit::new(ratio);
        let res_seed = seed ^ RESIDUAL_ROTATION_SEED;

        let split = Split::from_factory(splitter, move |branch, branch_dim| {
            if branch == 0 {
                // Outlier spikes: MinMax -> CastUint(8)
                Pipeline::new(
                    branch_dim,
                    vec![
                        Box::new(MinMax::default()) as Box<dyn Primitive>,
                        Box::new(CastUint::new(8)),
                    ],
                )
                .expect("valid spike branch pipeline")
            } else {
                // Bulk subspace: Rotate -> CastNormal(b-1) -> QJL residual
                let rot_stage = rotation.stage(seed);
                let mid_dim = rot_stage.out_dim(branch_dim);
                let residual = Qjl::pipeline(1.0, rotation, res_seed, mid_dim)
                    .expect("valid bulk QJL pipeline");

                Pipeline::new(
                    branch_dim,
                    vec![
                        rot_stage,
                        Box::new(CastNormal::new(bits - 1, NormalScale::Plain))
                            as Box<dyn Primitive>,
                        Box::new(residual),
                    ],
                )
                .expect("valid bulk branch pipeline")
            }
        });

        Pipeline::new(
            dim,
            vec![
                Box::new(Center),
                Box::new(Normalize),
                Box::new(split),
            ],
        )
    }
}

impl Quantizer for SpikeVQ {
    fn name() -> &'static str {
        "spike_vq"
    }

    fn display_name() -> &'static str {
        "SpikeVQ"
    }

    fn params() -> &'static [&'static str] {
        &["b", "ratio", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> SpikeSplit -> [CastUint(8), Rotate -> CastNormal(b-1) -> QJL]"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let b = get(p, "b")?;
        let ratio = get_or(p, "ratio", 0.05f32)?;
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;

        Ok(Self(Self::pipeline(b, ratio, rotation, seed, dim)?))
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
        let mut v: Array2<f32> = math::gaussian(&mut rng, (60, d));
        v.column_mut(2).mapv_inplace(|x| x * 40.0);
        let q: Array2<f32> = math::gaussian(&mut rng, (6, d));

        let p = params(&[("b", json!(4)), ("ratio", json!(0.125))]);
        let codec = SpikeVQ::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
