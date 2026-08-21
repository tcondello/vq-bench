//! `spec_eden`: Spectral-Subspace Gaussian Lloyd-Max Quantization with Water-Filling.
//! -
//! Allocates high-precision directional Gaussian Lloyd-Max codebooks and orthogonal
//! QJL residuals to the top semantic singular directions, with lower-bit Gaussian codebooks
//! for the isotropic tail subspace.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{
    CastNormal, Center, Normalize, NormalScale, Params, PcaRotate, Pipeline, Primitive, Quantizer,
    SpectralSplit, Split,
};

/// Independent seed offset for high-energy subspace residual rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0x59ED;

/// The `spec_eden` family.
pub struct SpecEden(pub Pipeline);

impl SpecEden {
    /// Pipeline for SpecEden:
    /// `Center -> Normalize -> PcaRotate -> SpectralSplit(ratio)`
    /// - Branch 0 (Semantic Subspace): `CastNormal(b, Directional) -> Qjl(1.0)`
    /// - Branch 1 (Tail Subspace): `CastNormal(b - 1, Directional)`
    pub fn pipeline(
        bits: u8,
        ratio: Option<f32>,
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(
            (2..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 2..={}, got {bits}",
            CodeLayout::MAX_BITS
        );

        let splitter = if let Some(r) = ratio {
            ensure!(r > 0.0 && r < 1.0, "ratio must be in (0.0, 1.0), got {r}");
            let k = ((dim as f32 * r).round() as usize).clamp(1, dim.saturating_sub(1));
            SpectralSplit::new_fixed(k)
        } else {
            SpectralSplit::new_dynamic()
        };

        let res_seed = seed ^ RESIDUAL_ROTATION_SEED;

        let split = Split::from_factory(splitter, move |branch, branch_dim| {
            if branch == 0 {
                // High-energy semantic singular subspace: b bits + 1-bit QJL residual
                let residual = Qjl::pipeline(1.0, rotation, res_seed, branch_dim)
                    .expect("valid QJL residual pipeline");
                Pipeline::new(
                    branch_dim,
                    vec![
                        Box::new(CastNormal::new(bits, NormalScale::Unbiased))
                            as Box<dyn Primitive>,
                        Box::new(residual),
                    ],
                )
                .expect("valid primary branch pipeline")
            } else {
                // Low-energy tail singular subspace: (b - 1) bits directional Gaussian
                Pipeline::new(
                    branch_dim,
                    vec![
                        Box::new(CastNormal::new(bits - 1, NormalScale::Unbiased))
                            as Box<dyn Primitive>,
                    ],
                )
                .expect("valid tail branch pipeline")
            }
        });

        Pipeline::new(
            dim,
            vec![
                Box::new(Center),
                Box::new(Normalize),
                Box::new(PcaRotate),
                Box::new(split),
            ],
        )
    }
}

impl Quantizer for SpecEden {
    fn name() -> &'static str {
        "spec_eden"
    }

    fn display_name() -> &'static str {
        "SpecEden"
    }

    fn params() -> &'static [&'static str] {
        &["b", "ratio", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> PcaRotate -> SpectralSplit -> [CastNormal(b)+QJL, CastNormal(b-1)]"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let b = get(p, "b")?;
        let ratio_raw: f32 = get_or(p, "ratio", 0.0)?;
        let ratio = if ratio_raw > 0.0 { Some(ratio_raw) } else { None };
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
        let mut v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        v.column_mut(0).mapv_inplace(|x| 8.0 * x);
        v.column_mut(1).mapv_inplace(|x| 4.0 * x);
        let q: Array2<f32> = math::gaussian(&mut rng, (8, d));

        let p = params(&[("b", json!(3)), ("ratio", json!(0.25))]);
        let codec = SpecEden::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
