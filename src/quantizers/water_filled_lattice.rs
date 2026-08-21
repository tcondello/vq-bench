//! `water_filled_lattice`: Reverse Water-Filling Gosset E8 Block Lattice Quantizer.
//! -
//! Allocates variable bit budgets across E8 Gosset lattice blocks derived from reverse
//! water-filling rate-distortion optimization on the singular value spectrum.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{
    CastMultiShellE8, CastNormal, Center, Normalize, NormalScale, Params, PcaRotate, Pipeline,
    Primitive, Quantizer, SpectralSplit, Split,
};

/// Independent seed offset for residual rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0x51E20;

/// The `water_filled_lattice` family.
pub struct WaterFilledLattice(pub Pipeline);

impl WaterFilledLattice {
    /// Pipeline for WaterFilledLattice:
    /// `Center -> Normalize -> PcaRotate -> SpectralSplit(ratio)`
    /// - Branch 0 (High-energy blocks): `CastMultiShellE8(b_high) -> Qjl(1.0)`
    /// - Branch 1 (Low-energy blocks): `CastMultiShellE8(b_low)`
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

        let res_seed = seed ^ RESIDUAL_ROTATION_SEED;
        let b_high = bits.saturating_sub(1).max(1);
        let b_low = (bits / 3).clamp(1, b_high);

        let split = Split::from_factory(splitter, move |branch, branch_dim| {
            if branch == 0 {
                // High-energy E8 lattice blocks + QJL
                let qjl = Qjl::pipeline(1.0, rotation, res_seed, branch_dim)
                    .expect("valid QJL residual pipeline");

                if branch_dim.is_multiple_of(8) {
                    Pipeline::new(
                        branch_dim,
                        vec![
                            Box::new(CastMultiShellE8::new(b_high)) as Box<dyn Primitive>,
                            Box::new(qjl),
                        ],
                    )
                    .expect("valid primary branch pipeline")
                } else {
                    Pipeline::new(
                        branch_dim,
                        vec![
                            Box::new(CastNormal::new(b_high, NormalScale::Plain))
                                as Box<dyn Primitive>,
                            Box::new(qjl),
                        ],
                    )
                    .expect("valid fallback primary branch pipeline")
                }
            } else {
                // Low-energy E8 lattice blocks
                if branch_dim.is_multiple_of(8) {
                    Pipeline::new(
                        branch_dim,
                        vec![
                            Box::new(CastMultiShellE8::new(b_low)) as Box<dyn Primitive>,
                        ],
                    )
                    .expect("valid low-energy branch pipeline")
                } else {
                    Pipeline::new(
                        branch_dim,
                        vec![
                            Box::new(CastNormal::new(b_low, NormalScale::Plain))
                                as Box<dyn Primitive>,
                        ],
                    )
                    .expect("valid fallback low-energy branch pipeline")
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

impl Quantizer for WaterFilledLattice {
    fn name() -> &'static str {
        "water_filled_lattice"
    }

    fn display_name() -> &'static str {
        "WaterFilledLattice"
    }

    fn params() -> &'static [&'static str] {
        &["b", "ratio", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> PcaRotate -> SpectralSplit -> [CastMultiShellE8(b_high)+QJL, CastMultiShellE8(b_low)]"
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
        let codec = WaterFilledLattice::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
