//! `manifold_shell`: Tangent-Bundle Spherical Shell Lattice Quantization.
//! -
//! Discovers the intrinsic manifold tangent space via PCA rotation, projecting the
//! top singular directions onto the 240 minimal root vectors of the E8 Gosset lattice on S^7
//! with orthogonal QJL residual refinement, while compressing the normal ambient space compactly.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{
    CastMultiShellE8, CastNormal, Center, Normalize, NormalScale, Params, PcaRotate, Pipeline,
    Primitive, Quantizer, SpectralSplit, Split,
};

/// Independent seed offsets for branch rotations.
const MANIFOLD_ROTATION_SEED: u64 = 0x3A0F1;
const RESIDUAL_ROTATION_SEED: u64 = 0x3A0F2;

/// The `manifold_shell` family.
pub struct ManifoldShell(pub Pipeline);

impl ManifoldShell {
    /// Pipeline for ManifoldShell:
    /// `Center -> Normalize -> PcaRotate -> SpectralSplit(ratio)`
    /// - Branch 0 (Manifold Tangent Space): `Rotate -> CastMultiShellE8(b-1) -> Qjl(1.0)`
    /// - Branch 1 (Normal Ambient Space): `Rotate -> CastMultiShellE8(1)`
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
            let mut k = ((dim as f32 * r).round() as usize).clamp(1, dim.saturating_sub(1).max(1));
            if dim >= 16 && (dim - k) >= 8 {
                let tail_8 = ((dim - k) / 8) * 8;
                if tail_8 > 0 && (dim - tail_8) > 0 {
                    k = dim - tail_8;
                }
            }
            SpectralSplit::new_fixed(k)
        } else {
            SpectralSplit::new_dynamic()
        };

        let man_seed = seed ^ MANIFOLD_ROTATION_SEED;
        let res_seed = seed ^ RESIDUAL_ROTATION_SEED;

        let split = Split::from_factory(splitter, move |branch, branch_dim| {
            if branch == 0 {
                // Manifold Tangent Space: rotate into isotropic S^(k-1) -> MultiShellE8 -> QJL
                let rot_stage = rotation.stage(man_seed);
                let mid_dim = rot_stage.out_dim(branch_dim);
                let qjl = Qjl::pipeline(1.0, rotation, res_seed, mid_dim)
                    .expect("valid QJL residual pipeline");

                if branch_dim.is_multiple_of(8) {
                    Pipeline::new(
                        branch_dim,
                        vec![
                            rot_stage,
                            Box::new(CastMultiShellE8::new(bits.saturating_sub(1).max(1)))
                                as Box<dyn Primitive>,
                            Box::new(qjl),
                        ],
                    )
                    .expect("valid manifold branch pipeline")
                } else {
                    Pipeline::new(
                        branch_dim,
                        vec![
                            rot_stage,
                            Box::new(CastNormal::new(bits.saturating_sub(1).max(1), NormalScale::Plain))
                                as Box<dyn Primitive>,
                            Box::new(qjl),
                        ],
                    )
                    .expect("valid fallback manifold branch pipeline")
                }
            } else {
                // Normal Ambient Space: compact 1-bit spherical Gosset shell
                let rot_stage = rotation.stage(seed.wrapping_add(branch as u64));
                if branch_dim.is_multiple_of(8) {
                    Pipeline::new(
                        branch_dim,
                        vec![
                            rot_stage,
                            Box::new(CastMultiShellE8::new(1)) as Box<dyn Primitive>,
                        ],
                    )
                    .expect("valid normal branch pipeline")
                } else {
                    Pipeline::new(
                        branch_dim,
                        vec![
                            rot_stage,
                            Box::new(CastNormal::new(1, NormalScale::Plain)) as Box<dyn Primitive>,
                        ],
                    )
                    .expect("valid fallback normal branch pipeline")
                }
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

impl Quantizer for ManifoldShell {
    fn name() -> &'static str {
        "manifold_shell"
    }

    fn display_name() -> &'static str {
        "ManifoldShell"
    }

    fn params() -> &'static [&'static str] {
        &["b", "ratio", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> PcaRotate -> SpectralSplit -> [Rotate -> CastMultiShellE8(b-1)+QJL, Rotate -> CastMultiShellE8(1)]"
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
        v.column_mut(0).mapv_inplace(|x| 6.0 * x);
        v.column_mut(1).mapv_inplace(|x| 4.0 * x);
        let q: Array2<f32> = math::gaussian(&mut rng, (8, d));

        let p = params(&[("b", json!(4)), ("ratio", json!(0.25))]);
        let codec = ManifoldShell::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
