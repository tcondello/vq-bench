//! `specbit`: Spectral-Ternary Hybrid Vector Quantization.
//! -
//! Combines PCA-based dynamic spectral splitting with high-precision QJL residual
//! quantization on the semantic subspace and fast AbsMean ternary {-1, 0, +1}
//! quantization on the tail subspace.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{
    CastNormal, CastTernary, Center, Normalize, NormalScale, Params, PcaRotate, Pipeline,
    Primitive, Quantizer, SpectralSplit, Split,
};

/// Independent seed offset for semantic residual rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0x5BC17;

/// The `specbit` family.
pub struct SpecBit(pub Pipeline);

impl SpecBit {
    /// Pipeline for SpecBit:
    /// `Center -> Normalize -> PcaRotate -> SpectralSplit`
    /// - Semantic branch: `CastNormal(b - 1, Plain) -> Qjl(1.0)`
    /// - Tail branch: `CastTernary` (AbsMean ternary {-1, 0, +1})
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
            let k = ((dim as f32 * r).round() as usize).clamp(1, dim.saturating_sub(1).max(1));
            SpectralSplit::new_fixed(k)
        } else {
            SpectralSplit::new_dynamic()
        };

        let residual_seed = seed ^ RESIDUAL_ROTATION_SEED;
        let split = Split::from_factory(splitter, move |branch, branch_dim| {
            if branch == 0 {
                let residual = Qjl::pipeline(1.0, rotation, residual_seed ^ 0x5EED, branch_dim)
                    .expect("valid QJL residual pipeline");
                Pipeline::new(
                    branch_dim,
                    vec![
                        Box::new(CastNormal::new(bits - 1, NormalScale::Plain))
                            as Box<dyn Primitive>,
                        Box::new(residual),
                    ],
                )
                .expect("valid semantic branch pipeline")
            } else {
                Pipeline::new(
                    branch_dim,
                    vec![Box::new(CastTernary) as Box<dyn Primitive>],
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

impl Quantizer for SpecBit {
    fn name() -> &'static str {
        "specbit"
    }

    fn display_name() -> &'static str {
        "SpecBit"
    }

    fn params() -> &'static [&'static str] {
        &["b", "ratio", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> PcaRotate -> SpectralSplit -> [CastNormal(b-1)+QJL, CastTernary]"
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

    fn make_test_specbit(
        bits: u8,
        ratio: Option<f32>,
        rotation: &str,
        seed: u64,
        dim: usize,
    ) -> Result<SpecBit> {
        let mut p_map = vec![("b", json!(bits)), ("rotation", json!(rotation))];
        let ratio_json;
        if let Some(r) = ratio {
            ratio_json = json!(r);
            p_map.push(("ratio", ratio_json));
        }
        let p = params(&p_map);
        SpecBit::build(&p, seed, dim)
    }

    #[test]
    fn rejects_invalid_params() {
        assert!(make_test_specbit(1, None, "hadamard", 1, 32).is_err());
        assert!(make_test_specbit(9, None, "hadamard", 1, 32).is_err());
        assert!(make_test_specbit(3, Some(1.5), "hadamard", 1, 32).is_err());
        assert!(make_test_specbit(3, None, "hadamard", 1, 32).is_ok());
    }

    #[test]
    fn round_trip_and_score() {
        let mut rng = math::seed(42);
        let d = 32;
        let mut v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        v.column_mut(0).mapv_inplace(|x| 5.0 * x);
        v.column_mut(1).mapv_inplace(|x| 3.0 * x);
        let q: Array2<f32> = math::gaussian(&mut rng, (8, d));

        let codec = make_test_specbit(3, Some(0.25), "hadamard", 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
