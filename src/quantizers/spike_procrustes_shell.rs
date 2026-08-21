//! `spike_procrustes_shell`: Outlier-Protected Procrustes Gosset Shell Quantizer.
//! -
//! Combines unrotated 8-bit outlier preservation with learned Procrustean rotation
//! and concentric E8 Gosset shell lattice quantization on the bulk.

use anyhow::{ensure, Result};

use super::catalog::get_or;
use super::opt_shell::OptShell;
use super::rotation::Rotation;
use crate::{CastUint, MinMax, Params, Pipeline, Primitive, Quantizer, Resize, SpikeSplit, Split};

/// The `spike_procrustes_shell` family.
pub struct SpikeProcrustesShell(pub Pipeline);

impl SpikeProcrustesShell {
    /// Pipeline for SpikeProcrustesShell:
    /// `SpikeSplit(spike_ratio)`:
    /// - Branch 0 (Outliers): `MinMax -> CastUint(8)`
    /// - Branch 1 (Bulk): `OptShell(residual)`
    pub fn pipeline(
        residual: bool,
        spike_ratio: f32,
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(
            spike_ratio > 0.0 && spike_ratio < 0.5,
            "spike_ratio must be in (0.0, 0.5), got {spike_ratio}"
        );

        let splitter = SpikeSplit::new(spike_ratio);

        let split = Split::from_factory(splitter, move |branch, branch_dim| {
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
                // Bulk: OptShell pipeline with Resize padding if needed
                let wide = branch_dim.div_ceil(8) * 8;
                let opt_shell = OptShell::pipeline(
                    residual,
                    rotation,
                    seed.wrapping_add(branch as u64),
                    wide,
                )
                .expect("valid bulk OptShell branch pipeline");

                if wide == branch_dim {
                    Pipeline::new(branch_dim, vec![Box::new(opt_shell) as Box<dyn Primitive>])
                        .expect("valid bulk branch pipeline")
                } else {
                    Pipeline::new(
                        branch_dim,
                        vec![
                            Box::new(Resize::to(wide)) as Box<dyn Primitive>,
                            Box::new(opt_shell),
                        ],
                    )
                    .expect("valid padded bulk branch pipeline")
                }
            }
        });

        Pipeline::new(dim, vec![Box::new(split)])
    }
}

impl Quantizer for SpikeProcrustesShell {
    fn name() -> &'static str {
        "spike_procrustes_shell"
    }

    fn display_name() -> &'static str {
        "SpikeProcrustesShell"
    }

    fn params() -> &'static [&'static str] {
        &["residual", "spike_ratio", "rotation"]
    }

    fn describe() -> &'static str {
        "SpikeSplit -> [MinMax(8), OptShell(residual)] [Outlier-Protected Procrustes Shell]"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let residual = get_or(p, "residual", true)?;
        let spike_ratio: f32 = get_or(p, "spike_ratio", 0.05)?;
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;

        Ok(Self(Self::pipeline(
            residual,
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

        let p = params(&[("residual", json!(false)), ("spike_ratio", json!(0.25))]);
        let codec = SpikeProcrustesShell::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
