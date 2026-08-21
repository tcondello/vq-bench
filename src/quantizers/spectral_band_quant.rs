//! `spectral_band_quant`: 4-Band Analytical Variance-Weighted Bit Allocation.
//! -
//! Decomposes the PCA eigenspectrum into 4 analytical sub-bands with progressively
//! decreasing bit allocations (8b, 6b, 4b, 2b) to achieve near-optimal rate-distortion
//! without iterative codebook learning.

use anyhow::Result;

use super::catalog::get_or;
use super::rotation::Rotation;
use crate::{
    CastNormal, CastUint, MinMax, NormalScale, Params, PcaRotate, Pipeline, Primitive, Quantizer,
    SpectralSplit, Split,
};

/// The `spectral_band_quant` family.
pub struct SpectralBandQuant(pub Pipeline);

impl SpectralBandQuant {
    /// Pipeline for SpectralBandQuant:
    /// `PcaRotate -> 4-Band Hierarchical Split`:
    /// - Band 1 (Top 12.5% variance): `MinMax -> CastUint(8)`
    /// - Band 2 (Next 25.0% variance): `Rotate -> CastNormal(6, Unbiased)`
    /// - Band 3 (Next 25.0% variance): `Rotate -> CastNormal(4, Unbiased)`
    /// - Band 4 (Tail 37.5% variance): `Rotate -> CastNormal(2, Unbiased)`
    pub fn pipeline(rotation: Rotation, seed: u64, dim: usize) -> Result<Pipeline> {
        let b1_ratio = 0.125f32;
        let b2_ratio = 0.285714f32; // 0.25 / 0.875
        let b3_ratio = 0.400000f32; // 0.25 / 0.625

        let k1 = (dim as f32 * b1_ratio).round() as usize;
        let k1 = k1.clamp(1, dim.saturating_sub(3));
        let rem1 = dim - k1;

        let k2 = (rem1 as f32 * b2_ratio).round() as usize;
        let k2 = k2.clamp(1, rem1.saturating_sub(2));
        let rem2 = rem1 - k2;

        let k3 = (rem2 as f32 * b3_ratio).round() as usize;
        let k3 = k3.clamp(1, rem2.saturating_sub(1));

        let split1 = Split::from_factory(SpectralSplit::new_fixed(k1), move |branch, branch_dim| {
            if branch == 0 {
                Pipeline::new(
                    branch_dim,
                    vec![
                        Box::new(MinMax::default()) as Box<dyn Primitive>,
                        Box::new(CastUint::new(8)),
                    ],
                )
                .expect("valid band 1 pipeline")
            } else {
                let split2 = Split::from_factory(SpectralSplit::new_fixed(k2), move |b2, d2| {
                    if b2 == 0 {
                        let rot = rotation.stage(seed ^ 0x2222);
                        Pipeline::new(
                            d2,
                            vec![rot, Box::new(CastNormal::new(6, NormalScale::Unbiased))],
                        )
                        .expect("valid band 2 pipeline")
                    } else {
                        let split3 = Split::from_factory(SpectralSplit::new_fixed(k3), move |b3, d3| {
                            if b3 == 0 {
                                let rot = rotation.stage(seed ^ 0x3333);
                                Pipeline::new(
                                    d3,
                                    vec![rot, Box::new(CastNormal::new(4, NormalScale::Unbiased))],
                                )
                                .expect("valid band 3 pipeline")
                            } else {
                                let rot = rotation.stage(seed ^ 0x4444);
                                Pipeline::new(
                                    d3,
                                    vec![rot, Box::new(CastNormal::new(2, NormalScale::Unbiased))],
                                )
                                .expect("valid band 4 pipeline")
                            }
                        });
                        Pipeline::new(d2, vec![Box::new(split3)])
                            .expect("valid level 3 pipeline")
                    }
                });
                Pipeline::new(branch_dim, vec![Box::new(split2)])
                    .expect("valid level 2 pipeline")
            }
        });

        Pipeline::new(
            dim,
            vec![
                Box::new(PcaRotate),
                Box::new(split1),
            ],
        )
    }
}

impl Quantizer for SpectralBandQuant {
    fn name() -> &'static str {
        "spectral_band_quant"
    }

    fn display_name() -> &'static str {
        "SpectralBandQuant"
    }

    fn params() -> &'static [&'static str] {
        &["rotation"]
    }

    fn describe() -> &'static str {
        "PcaRotate -> 4-Band Spectral Split [B1(8b), B2(6b), B3(4b), B4(2b)]"
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

    #[test]
    fn round_trip_and_score() {
        let mut rng = math::seed(42);
        let d = 32;
        let v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (8, d));

        let p = params(&[]);
        let codec = SpectralBandQuant::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
