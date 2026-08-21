//! `spike_adapt_manifold`: Outlier-Preserved Adaptive Manifold Gosset E8 Lattice.
//! -
//! Combines 8-bit unrotated outlier preservation with adaptive tangent-tail Gosset E8
//! spherical shell lattice quantization on the remaining isotropic bulk.

use anyhow::{ensure, Result};

use super::adapt_manifold_shell::AdaptiveManifoldShell;
use super::catalog::{get, get_or};
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{CastUint, MinMax, Params, Pipeline, Primitive, Quantizer, SpikeSplit, Split};

/// The `spike_adapt_manifold` family.
pub struct SpikeAdaptiveManifold(pub Pipeline);

impl SpikeAdaptiveManifold {
    /// Pipeline for SpikeAdaptiveManifold:
    /// `SpikeSplit(spike_ratio)`:
    /// - Branch 0 (Outliers): `MinMax -> CastUint(8)`
    /// - Branch 1 (Bulk): `AdaptiveManifoldShell(b, spectral_ratio)`
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
                // Bulk: AdaptiveManifoldShell pipeline
                AdaptiveManifoldShell::pipeline(
                    bits,
                    Some(spectral_ratio),
                    rotation,
                    seed.wrapping_add(branch as u64),
                    branch_dim,
                )
                .expect("valid bulk AdaptiveManifoldShell branch pipeline")
            }
        });

        Pipeline::new(dim, vec![Box::new(split)])
    }
}

impl Quantizer for SpikeAdaptiveManifold {
    fn name() -> &'static str {
        "spike_adapt_manifold"
    }

    fn display_name() -> &'static str {
        "SpikeAdaptiveManifold"
    }

    fn params() -> &'static [&'static str] {
        &["b", "spike_ratio", "spectral_ratio", "rotation"]
    }

    fn describe() -> &'static str {
        "SpikeSplit -> [MinMax(8), AdaptiveManifoldShell(b)] [Outlier-Protected Adaptive Tangent E8]"
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
        let d = 32;
        let mut v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        v.column_mut(0).mapv_inplace(|x| 20.0 * x);
        v.column_mut(1).mapv_inplace(|x| 15.0 * x);
        let q: Array2<f32> = math::gaussian(&mut rng, (8, d));

        let p = params(&[
            ("b", json!(4)),
            ("spike_ratio", json!(0.25)),
            ("spectral_ratio", json!(0.25)),
        ]);
        let codec = SpikeAdaptiveManifold::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
