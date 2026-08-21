//! `cascade_lattice_eden`: Dual-Stage 24D Leech Anchor with Multi-Bit Gaussian Lloyd-Max Residual.
//! -
//! Combines a coarse 2-bit 24-dimensional Leech lattice stage to capture cross-channel
//! geometry with a high-throughput multi-bit Gaussian Lloyd-Max residual rounder.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{
    CastLeech24, CastNormal, CastUint, Center, MinMax, Normalize, NormalScale, Params, Pipeline,
    Primitive, Quantizer, Resize, SpikeSplit, Split,
};

/// The `cascade_lattice_eden` family.
pub struct CascadeLatticeEden(pub Pipeline);

impl CascadeLatticeEden {
    /// Pipeline for CascadeLatticeEden:
    /// `SpikeSplit(ratio)`:
    /// - Branch 0 (Outliers): `MinMax -> CastUint(8)`
    /// - Branch 1 (Bulk): `Center -> Normalize -> Rotate(Hadamard) -> Resize(24) -> CastLeech24(2) -> CastNormal(b, Unbiased)`
    pub fn pipeline(
        bits: u8,
        ratio: f32,
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 1..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
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
                let wide = branch_dim.div_ceil(24) * 24;
                let rot = rotation.stage(seed.wrapping_add(branch as u64));
                let mut stages: Vec<Box<dyn Primitive>> = vec![
                    Box::new(Center),
                    Box::new(Normalize),
                    rot,
                ];
                if wide != branch_dim {
                    stages.push(Box::new(Resize::to(wide)));
                }
                stages.push(Box::new(CastLeech24::new(2)));
                stages.push(Box::new(CastNormal::new(bits, NormalScale::Unbiased)));
                Pipeline::new(branch_dim, stages).expect("valid bulk cascade pipeline")
            }
        });

        Pipeline::new(dim, vec![Box::new(split)])
    }
}

impl Quantizer for CascadeLatticeEden {
    fn name() -> &'static str {
        "cascade_lattice_eden"
    }

    fn display_name() -> &'static str {
        "CascadeLatticeEden"
    }

    fn params() -> &'static [&'static str] {
        &["b", "ratio", "rotation"]
    }

    fn describe() -> &'static str {
        "SpikeSplit -> [MinMax(8), Center -> Normalize -> Rotate -> CastLeech24(2) -> CastNormal(b)]"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let b = get(p, "b")?;
        let ratio = get_or(p, "ratio", 0.05)?;
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
        let d = 48;
        let mut v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        v.column_mut(0).mapv_inplace(|x| 20.0 * x);
        v.column_mut(1).mapv_inplace(|x| 15.0 * x);
        let q: Array2<f32> = math::gaussian(&mut rng, (8, d));

        let p = params(&[("b", json!(2)), ("ratio", json!(0.0625))]);
        let codec = CascadeLatticeEden::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
