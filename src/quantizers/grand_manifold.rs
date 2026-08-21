//! `grand_manifold`: The Unified Quintuple-Stage Manifold Lattice Architecture.
//! -
//! Fuses outlier isolation (SpikeSplit), PCA tangent bundle projection, learned Procrustean
//! spherical Gosset shell quantization, adaptive 24D Leech ambient tails, and orthogonal QJL.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::opt_shell::OptShell;
use super::qjl::Qjl;
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{
    CastLeech24, CastNormal, CastUint, Center, MinMax, Normalize, NormalScale, Params, PcaRotate,
    Pipeline, Primitive, Quantizer, SpectralSplit, SpikeSplit, Split,
};

/// Independent seed offset for residual rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0x99A41;

/// The `grand_manifold` family.
pub struct GrandManifold(pub Pipeline);

impl GrandManifold {
    /// Pipeline for GrandManifold:
    /// `SpikeSplit(spike_ratio)`:
    /// - Branch 0 (Outliers): `MinMax -> CastUint(8)`
    /// - Branch 1 (Bulk): `Center -> Normalize -> PcaRotate -> SpectralSplit(ratio)`:
    ///   - Tangent Subspace: `OptShell(b_head) -> Qjl(1.0)`
    ///   - Ambient Subspace: `CastLeech24(b_tail)`
    pub fn pipeline(
        bits: u8,
        spike_ratio: f32,
        spectral_ratio: f32,
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(
            (2..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 2..={}, got {bits}",
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

        let spike_splitter = SpikeSplit::new(spike_ratio);
        let res_seed = seed ^ RESIDUAL_ROTATION_SEED;
        let b_head = bits.saturating_sub(1).max(1);
        let b_tail = (bits / 2).clamp(1, b_head);

        let spike_split = Split::from_factory(spike_splitter, move |branch, branch_dim| {
            if branch == 0 {
                // Outlier channels: unrotated uniform MinMax -> CastUint(8)
                Pipeline::new(
                    branch_dim,
                    vec![
                        Box::new(MinMax::default()) as Box<dyn Primitive>,
                        Box::new(CastUint::new(8)),
                    ],
                )
                .expect("valid outlier branch pipeline")
            } else {
                // Bulk: Tangent & Normal Subspace Decomposition
                let spectral_splitter = {
                    let mut k = ((branch_dim as f32 * spectral_ratio).round() as usize)
                        .clamp(1, branch_dim.saturating_sub(1).max(1));
                    if branch_dim >= 32 && k >= 8 {
                        let k_8 = (k / 8) * 8;
                        if k_8 > 0 && (branch_dim - k_8) > 0 {
                            k = k_8;
                        }
                    }
                    SpectralSplit::new_fixed(k)
                };

                let bulk_split = Split::from_factory(spectral_splitter, move |sub_branch, sub_dim| {
                    if sub_branch == 0 {
                        // Tangent Space: OptShell + QJL
                        let qjl = Qjl::pipeline(1.0, rotation, res_seed, sub_dim)
                            .expect("valid QJL pipeline");

                        if sub_dim.is_multiple_of(8) {
                            let opt_shell = OptShell::pipeline(true, rotation, seed ^ 0x11, sub_dim)
                                .expect("valid OptShell pipeline");

                            Pipeline::new(
                                sub_dim,
                                vec![
                                    Box::new(opt_shell) as Box<dyn Primitive>,
                                ],
                            )
                            .expect("valid tangent branch pipeline")
                        } else {
                            Pipeline::new(
                                sub_dim,
                                vec![
                                    Box::new(CastNormal::new(b_head, NormalScale::Plain))
                                        as Box<dyn Primitive>,
                                    Box::new(qjl),
                                ],
                            )
                            .expect("valid fallback tangent branch pipeline")
                        }
                    } else {
                        // Ambient Normal Space: CastLeech24
                        if sub_dim.is_multiple_of(24) {
                            Pipeline::new(
                                sub_dim,
                                vec![Box::new(CastLeech24::new(b_tail)) as Box<dyn Primitive>],
                            )
                            .expect("valid normal Leech branch pipeline")
                        } else {
                            Pipeline::new(
                                sub_dim,
                                vec![Box::new(CastNormal::new(b_tail, NormalScale::Plain)) as Box<dyn Primitive>],
                            )
                            .expect("valid fallback normal branch pipeline")
                        }
                    }
                });

                Pipeline::new(
                    branch_dim,
                    vec![
                        Box::new(Center),
                        Box::new(Normalize),
                        Box::new(PcaRotate),
                        Box::new(bulk_split),
                    ],
                )
                .expect("valid bulk pipeline")
            }
        });

        Pipeline::new(dim, vec![Box::new(spike_split)])
    }
}

impl Quantizer for GrandManifold {
    fn name() -> &'static str {
        "grand_manifold"
    }

    fn display_name() -> &'static str {
        "GrandManifold"
    }

    fn params() -> &'static [&'static str] {
        &["b", "spike_ratio", "spectral_ratio", "rotation"]
    }

    fn describe() -> &'static str {
        "SpikeSplit -> [MinMax(8), PcaRotate -> SpectralSplit -> [OptShell(b)+QJL, CastLeech24(b_tail)]]"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let b = get(p, "b")?;
        let spike_ratio: f32 = get_or(p, "spike_ratio", 0.05)?;
        let spectral_ratio: f32 = get_or(p, "spectral_ratio", 0.25)?;
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;

        Ok(Self(Self::pipeline(
            b,
            spike_ratio,
            spectral_ratio,
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
            ("b", json!(4)),
            ("spike_ratio", json!(0.25)),
            ("spectral_ratio", json!(0.25)),
        ]);
        let codec = GrandManifold::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
