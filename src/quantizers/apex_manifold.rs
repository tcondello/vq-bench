//! `apex_manifold`: The Meta-Synthesis Peak Quantization Architecture.
//! -
//! Combines native unrotated 8-bit spike isolation, PCA covariance alignment,
//! and 3-band optimal spectral rate allocation (b_high, b_mid, b_low) using
//! directional Gaussian Lloyd-Max codebooks for maximum recall and sub-4s encoding.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{
    CastNormal, CastUint, MinMax, NormalScale, Params, PcaRotate, Pipeline, Primitive,
    Quantizer, SpectralSplit, SpikeSplit, Split,
};

/// The `apex_manifold` family.
pub struct ApexManifold(pub Pipeline);

impl ApexManifold {
    /// Pipeline for ApexManifold:
    /// `SpikeSplit(spike_ratio)`:
    /// - Branch 0 (Outliers): `MinMax -> CastUint(8)`
    /// - Branch 1 (Bulk): `PcaRotate -> 3-Band Hierarchical Split`:
    ///   - Band 1 (Top 25% Tangent): `Rotate -> CastNormal(b_high, Unbiased)`
    ///   - Band 2 (Mid 35% Tangent): `Rotate -> CastNormal(b_mid, Unbiased)`
    ///   - Band 3 (Tail 40% Ambient): `Rotate -> CastNormal(b_low, Unbiased)`
    pub fn pipeline(
        b_high: u8,
        b_mid: u8,
        b_low: u8,
        spike_ratio: f32,
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&b_high),
            "b_high must be in 1..={}, got {b_high}",
            CodeLayout::MAX_BITS
        );
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&b_mid),
            "b_mid must be in 1..={}, got {b_mid}",
            CodeLayout::MAX_BITS
        );
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&b_low),
            "b_low must be in 1..={}, got {b_low}",
            CodeLayout::MAX_BITS
        );
        ensure!(
            spike_ratio > 0.0 && spike_ratio < 0.5,
            "spike_ratio must be in (0.0, 0.5), got {spike_ratio}"
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
                let k1 = (branch_dim as f32 * 0.25).round() as usize;
                let k1 = k1.clamp(1, branch_dim.saturating_sub(2).max(1));
                let rem1 = branch_dim - k1;

                let k2 = (rem1 as f32 * 0.466667).round() as usize; // 0.35 / 0.75 ≈ 0.4667
                let k2 = k2.clamp(1, rem1.saturating_sub(1).max(1));

                let spec_split = Split::from_factory(
                    SpectralSplit::new_fixed(k1),
                    move |spec_b, spec_dim| {
                        if spec_b == 0 {
                            let rot = rotation.stage(seed.wrapping_add(0x1111));
                            Pipeline::new(
                                spec_dim,
                                vec![rot, Box::new(CastNormal::new(b_high, NormalScale::Unbiased))],
                            )
                            .expect("valid band 1 pipeline")
                        } else {
                            let inner_split = Split::from_factory(
                                SpectralSplit::new_fixed(k2),
                                move |in_b, in_dim| {
                                    if in_b == 0 {
                                        let rot = rotation.stage(seed.wrapping_add(0x2222));
                                        Pipeline::new(
                                            in_dim,
                                            vec![rot, Box::new(CastNormal::new(b_mid, NormalScale::Unbiased))],
                                        )
                                        .expect("valid band 2 pipeline")
                                    } else {
                                        let rot = rotation.stage(seed.wrapping_add(0x3333));
                                        Pipeline::new(
                                            in_dim,
                                            vec![rot, Box::new(CastNormal::new(b_low, NormalScale::Unbiased))],
                                        )
                                        .expect("valid band 3 pipeline")
                                    }
                                },
                            );
                            Pipeline::new(spec_dim, vec![Box::new(inner_split)])
                                .expect("valid inner split pipeline")
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
                .expect("valid bulk apex manifold pipeline")
            }
        });

        Pipeline::new(dim, vec![Box::new(split)])
    }
}

impl Quantizer for ApexManifold {
    fn name() -> &'static str {
        "apex_manifold"
    }

    fn display_name() -> &'static str {
        "ApexManifold"
    }

    fn params() -> &'static [&'static str] {
        &["b_high", "b_mid", "b_low", "spike_ratio", "rotation"]
    }

    fn describe() -> &'static str {
        "SpikeSplit(5%) -> [MinMax(8), PcaRotate -> 3-Band Split -> [CastNormal(b_high), CastNormal(b_mid), CastNormal(b_low)]]"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let b_high = get(p, "b_high")?;
        let b_mid = get(p, "b_mid")?;
        let b_low = get(p, "b_low")?;
        let spike_ratio = get_or(p, "spike_ratio", 0.05)?;
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;

        Ok(Self(Self::pipeline(
            b_high,
            b_mid,
            b_low,
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
        let d = 32;
        let mut v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        v.column_mut(0).mapv_inplace(|x| 20.0 * x);
        v.column_mut(1).mapv_inplace(|x| 15.0 * x);
        let q: Array2<f32> = math::gaussian(&mut rng, (8, d));

        let p = params(&[
            ("b_high", json!(6)),
            ("b_mid", json!(4)),
            ("b_low", json!(2)),
            ("spike_ratio", json!(0.0625)),
        ]);
        let codec = ApexManifold::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
