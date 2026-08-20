//! `spectralquant`: center, normalize, PCA-rotate, split into semantic and tail subspaces,
//! then quantize the semantic subspace with (b-1) Gaussian bits + 1 QJL residual bit, and
//! the tail subspace with b_low Gaussian bits.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{
    CastNormal, Center, Normalize, NormalScale, Params, PcaRotate, Pipeline, Primitive,
    Quantizer, SpectralSplit, Split,
};

/// Independent seed offset for the residual's QJL rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0x59EC7;

/// The `spectralquant` family.
pub struct SpectralQuant(pub Pipeline);

impl SpectralQuant {
    /// Pipeline for SpectralQuant:
    /// `Center -> Normalize -> PcaRotate -> Split(SpectralSplit)`
    /// Semantic branch: `CastNormal(b - 1, Plain) -> Qjl(1.0)`
    /// Tail branch: `CastNormal(b_low, Plain)`
    pub fn pipeline(
        bits: u8,
        b_low: Option<u8>,
        ratio: Option<f32>,
        d_sem: Option<usize>,
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(
            (2..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 2..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
        let tail_bits = b_low.unwrap_or_else(|| bits.saturating_sub(1).max(1));
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&tail_bits),
            "b_low must be in 1..={}, got {tail_bits}",
            CodeLayout::MAX_BITS
        );

        let splitter = if let Some(d) = d_sem {
            ensure!(d > 0 && d < dim, "d_sem must be in 1..{dim}, got {d}");
            SpectralSplit::new_fixed(d)
        } else if let Some(r) = ratio {
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
                    vec![
                        Box::new(CastNormal::new(tail_bits, NormalScale::Plain))
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

impl Quantizer for SpectralQuant {
    fn name() -> &'static str {
        "spectralquant"
    }

    fn display_name() -> &'static str {
        "SpectralQuant"
    }

    fn params() -> &'static [&'static str] {
        &["b", "b_low", "ratio", "d_sem", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> PcaRotate -> SpectralSplit -> [CastNormal(b-1)+QJL, CastNormal]"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let b = get(p, "b")?;
        let b_low_raw: u8 = get_or(p, "b_low", 0)?;
        let b_low = if b_low_raw > 0 { Some(b_low_raw) } else { None };
        let ratio_raw: f32 = get_or(p, "ratio", 0.0)?;
        let ratio = if ratio_raw > 0.0 { Some(ratio_raw) } else { None };
        let d_sem_raw: usize = get_or(p, "d_sem", 0)?;
        let d_sem = if d_sem_raw > 0 { Some(d_sem_raw) } else { None };
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;

        Ok(Self(Self::pipeline(b, b_low, ratio, d_sem, rotation, seed, dim)?))
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

    fn make_test_quantizer(
        bits: u8,
        ratio: Option<f32>,
        rotation: &str,
        seed: u64,
        dim: usize,
    ) -> Result<SpectralQuant> {
        let mut p_map = vec![("b", json!(bits)), ("rotation", json!(rotation))];
        let ratio_json;
        if let Some(r) = ratio {
            ratio_json = json!(r);
            p_map.push(("ratio", ratio_json));
        }
        let p = params(&p_map);
        SpectralQuant::build(&p, seed, dim)
    }

    #[test]
    fn rejects_out_of_range_bits() {
        assert!(make_test_quantizer(1, None, "hadamard", 1, 32).is_err());
        assert!(make_test_quantizer(9, None, "hadamard", 1, 32).is_err());
        assert!(make_test_quantizer(3, None, "hadamard", 1, 32).is_ok());
    }

    #[test]
    fn rejects_invalid_ratio() {
        assert!(make_test_quantizer(3, Some(0.0), "hadamard", 1, 32).is_ok());
        assert!(make_test_quantizer(3, Some(1.5), "hadamard", 1, 32).is_err());
    }

    #[test]
    fn unbiased_score_slope() {
        let mut rng = math::seed(42);
        let d = 64;
        let mut v: Array2<f32> = math::gaussian(&mut rng, (120, d));
        // Give first few columns dominant signal
        v.column_mut(0).mapv_inplace(|x| 6.0 * x);
        v.column_mut(1).mapv_inplace(|x| 4.0 * x);
        let q: Array2<f32> = math::gaussian(&mut rng, (12, d));

        for rotation in ["hadamard", "full"] {
            let codec = make_test_quantizer(4, Some(0.1), rotation, 1, d).unwrap();
            let model = codec.fit(v.view(), None);
            let codes = codec.encode(&model, v.view());
            let est = codec.score(&model, q.view(), &refs(&codes));
            let exact = q.dot(&v.t());
            let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
            let st: f32 = exact.iter().map(|t| t * t).sum();
            assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
        }
    }
}
