//! `spike_shell`: Outlier-Preserving Concentric Multi-Shell Lattice Quantization.
//! -
//! Identifies and preserves extreme coordinate outliers in native coordinates with 8-bit
//! uniform MinMax, and quantizes the dense rotated bulk across concentric E8 Gosset lattice shells.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{
    CastMultiShellE8, CastUint, MinMax, Params, Pipeline, Primitive, Quantizer,
    SpikeSplit, Split,
};

/// The `spike_shell` family.
pub struct SpikeShell(pub Pipeline);

impl SpikeShell {
    /// Pipeline for SpikeShell:
    /// `SpikeSplit(ratio)`:
    /// - Branch 0 (Outliers): `MinMax -> CastUint(8)` (Native unrotated)
    /// - Branch 1 (Bulk): `Rotate -> CastMultiShellE8(b)`
    pub fn pipeline(
        bits: u8,
        ratio: f32,
        rotation: Rotation,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 1..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
        ensure!(ratio > 0.0 && ratio < 0.5, "ratio must be in (0.0, 0.5), got {ratio}");

        let splitter = SpikeSplit::new(ratio);

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
                // Dense bulk: rotate and quantize with concentric E8 lattice shells
                let stage = rotation.stage(seed.wrapping_add(branch as u64));
                Pipeline::new(
                    branch_dim,
                    vec![
                        stage,
                        Box::new(CastMultiShellE8::new(bits)),
                    ],
                )
                .expect("valid bulk branch pipeline")
            }
        });

        Pipeline::new(dim, vec![Box::new(split)])
    }
}

impl Quantizer for SpikeShell {
    fn name() -> &'static str {
        "spike_shell"
    }

    fn display_name() -> &'static str {
        "SpikeShell"
    }

    fn params() -> &'static [&'static str] {
        &["b", "ratio", "rotation"]
    }

    fn describe() -> &'static str {
        "SpikeSplit -> [MinMax -> CastUint(8), Rotate -> CastMultiShellE8(b)]"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let b = get(p, "b")?;
        let ratio: f32 = get_or(p, "ratio", 0.05)?;
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
        v.column_mut(0).mapv_inplace(|x| 20.0 * x);
        v.column_mut(1).mapv_inplace(|x| 15.0 * x);
        let q: Array2<f32> = math::gaussian(&mut rng, (8, d));

        let p = params(&[("b", json!(4)), ("ratio", json!(0.0625))]);
        let codec = SpikeShell::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.35, "slope {}", se / st);
    }
}
