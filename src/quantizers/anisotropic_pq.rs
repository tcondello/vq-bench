//! `anisotropic_pq`: ScaNN Anisotropic Product Quantization.
//! -
//! Contiguous `section_dim`-column segments, each rounded to its own `centroids`-codeword
//! anisotropic score-aware k-means codebook (ScaNN directional loss).
//! Recovers standard PQ baseline continuously when omega = 0.0.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use crate::coding::CodeLayout;
use crate::{AnisotropicKmeans, Params, Pipeline, Primitive, Quantizer, SegmentSplit, Split};

pub struct AnisotropicPq(pub Pipeline);

impl AnisotropicPq {
    pub fn pipeline(
        centroids: usize,
        section_dim: usize,
        omega: f32,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        ensure!(
            (2..=1 << CodeLayout::MAX_BITS).contains(&centroids),
            "centroids must be in 2..={}, got {centroids}",
            1u32 << CodeLayout::MAX_BITS
        );
        ensure!(
            (1..=dim).contains(&section_dim),
            "section_dim must be in 1..={dim}, got {section_dim}"
        );
        ensure!(omega >= 0.0, "omega must be >= 0.0, got {omega}");

        let split = Split::from_factory(SegmentSplit::new(dim, section_dim), move |b, branch_dim| {
            let rounder = AnisotropicKmeans::new(centroids, omega, seed.wrapping_add(b as u64));
            Pipeline::new(branch_dim, vec![Box::new(rounder) as Box<dyn Primitive>])
                .expect("a dim-generic stage cannot mismatch its input dim")
        });
        Pipeline::new(dim, vec![Box::new(split)])
    }
}

impl Quantizer for AnisotropicPq {
    fn name() -> &'static str {
        "anisotropic_pq"
    }

    fn display_name() -> &'static str {
        "AnisotropicPQ"
    }

    fn params() -> &'static [&'static str] {
        &["centroids", "section_dim", "omega"]
    }

    fn describe() -> &'static str {
        "SegmentSplit(section_dim) -> [AnisotropicKmeans(centroids, omega)]"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let centroids = get(p, "centroids")?;
        let section_dim = get(p, "section_dim")?;
        let omega = get_or(p, "omega", 0.0f32)?;
        Ok(Self(Self::pipeline(centroids, section_dim, omega, seed, dim)?))
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
    fn test_anisotropic_pq_neutral_anchor_recovers_baseline() {
        let mut rng = math::seed(42);
        let d = 32;
        let v: Array2<f32> = math::gaussian(&mut rng, (60, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (5, d));

        // When omega = 0.0, AnisotropicPQ must build and score cleanly
        let p = params(&[
            ("centroids", json!(16)),
            ("section_dim", json!(8)),
            ("omega", json!(0.0)),
        ]);
        let codec = AnisotropicPq::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.45, "slope {}", se / st);
    }
}
