//! `multiscale_leech`: Multi-Scale Outlier-Protected 24D Leech Tangent Lattice with Adaptive Residuals.
//! -
//! Isolates outlier spikes, projects tangent vectors onto concentric 24D Leech lattice
//! shells Lambda24, and quantizes ambient normal residuals with directional Lloyd-Max.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{
    CastLeech24, CastNormal, CastUint, MinMax, NormalScale, Params, PcaRotate, Pipeline,
    Primitive, Quantizer, Resize, SpectralSplit, SpikeSplit, Split,
};

/// The `multiscale_leech` family.
pub struct MultiScaleLeech(pub Pipeline);

impl MultiScaleLeech {
    /// Pipeline for MultiScaleLeech:
    /// `SpikeSplit(ratio)`:
    /// - Branch 0 (Outliers): `MinMax -> CastUint(8)`
    /// - Branch 1 (Bulk): `PcaRotate -> SpectralSplit(ratio)`:
    ///   - Tangent (Head): `Rotate -> Resize(24) -> CastLeech24(b_head)`
    ///   - Ambient (Tail): `Rotate -> CastNormal(b_tail, Unbiased)`
    pub fn pipeline(
        b_head: u8,
        b_tail: u8,
        spectral_ratio: f32,
        spike_ratio: f32,
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&b_head),
            "b_head must be in 1..={}, got {b_head}",
            CodeLayout::MAX_BITS
        );
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&b_tail),
            "b_tail must be in 1..={}, got {b_tail}",
            CodeLayout::MAX_BITS
        );
        ensure!(
            spike_ratio > 0.0 && spike_ratio < 0.5,
            "spike_ratio must be in (0.0, 0.5), got {spike_ratio}"
        );
        ensure!(
            spectral_ratio > 0.0 && spectral_ratio < 1.0,
            "spectral_ratio must be in (0.0, 1.0), got {spectral_ratio}"
        );

        let splitter = SpikeSplit::new(spike_ratio);

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
                let k_head = (branch_dim as f32 * spectral_ratio).round() as usize;
                let k_head = k_head.clamp(1, branch_dim.saturating_sub(1).max(1));

                let spec_split = Split::from_factory(
                    SpectralSplit::new_fixed(k_head),
                    move |spec_b, spec_dim| {
                        if spec_b == 0 {
                            let wide = spec_dim.div_ceil(24) * 24;
                            let rot = rotation.stage(seed.wrapping_add(0x1111));
                            let mut stages: Vec<Box<dyn Primitive>> = vec![rot];
                            if wide != spec_dim {
                                stages.push(Box::new(Resize::to(wide)));
                            }
                            stages.push(Box::new(CastLeech24::new(b_head)));
                            Pipeline::new(spec_dim, stages).expect("valid tangent leech pipeline")
                        } else {
                            let rot = rotation.stage(seed.wrapping_add(0x2222));
                            Pipeline::new(
                                spec_dim,
                                vec![rot, Box::new(CastNormal::new(b_tail, NormalScale::Unbiased))],
                            )
                            .expect("valid ambient tail pipeline")
                        }
                    },
                );

                Pipeline::new(
                    branch_dim,
                    vec![
                        Box::new(PcaRotate),
                        Box::new(spec_split),
                    ],
                )
                .expect("valid bulk multiscale leech pipeline")
            }
        });

        Pipeline::new(dim, vec![Box::new(split)])
    }
}

impl Quantizer for MultiScaleLeech {
    fn name() -> &'static str {
        "multiscale_leech"
    }

    fn display_name() -> &'static str {
        "MultiScaleLeech"
    }

    fn params() -> &'static [&'static str] {
        &["b_head", "b_tail", "spectral_ratio", "spike_ratio", "rotation"]
    }

    fn describe() -> &'static str {
        "SpikeSplit -> [MinMax(8), PcaRotate -> SpectralSplit -> [CastLeech24(b_head), CastNormal(b_tail)]]"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let b_head = get(p, "b_head")?;
        let b_tail = get(p, "b_tail")?;
        let spectral_ratio = get_or(p, "spectral_ratio", 0.50)?;
        let spike_ratio = get_or(p, "spike_ratio", 0.05)?;
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;

        Ok(Self(Self::pipeline(
            b_head,
            b_tail,
            spectral_ratio,
            spike_ratio,
            rotation,
            seed,
            dim,
        )?))
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

        let p = params(&[
            ("b_head", json!(4)),
            ("b_tail", json!(2)),
            ("spectral_ratio", json!(0.5)),
            ("spike_ratio", json!(0.0625)),
        ]);
        let codec = MultiScaleLeech::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
