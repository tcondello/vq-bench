//! `anisotropic_opq`: ScaNN Anisotropic Optimized Product Quantization.
//! -
//! Learned OPQ rotation followed by anisotropic score-aware product quantization.
//! Recovers standard OPQ baseline continuously when omega = 0.0.

use anyhow::{Context, Result};
use serde_json::Value;

use super::anisotropic_pq::AnisotropicPq;
use super::catalog::{get, get_or, FromParam};
use super::opq_p::OpqP;
use crate::{OptimizePq, Params, Pipeline, Primitive, Quantizer};

const DEFAULT_ITERS: usize = 15;

#[derive(Clone, Copy)]
pub enum Init {
    Identity,
    Eigen,
}

impl FromParam for Init {
    fn from_value(v: &Value) -> Result<Self> {
        match v.as_str().context("must be a string")? {
            "identity" => Ok(Init::Identity),
            "eigen" => Ok(Init::Eigen),
            other => anyhow::bail!("unknown init `{other}` (expected `identity` or `eigen`)"),
        }
    }
}

pub struct AnisotropicOpq(pub Pipeline);

impl AnisotropicOpq {
    pub fn pipeline(
        centroids: usize,
        section_dim: usize,
        omega: f32,
        iters: usize,
        init: Init,
        seed: u64,
        dim: usize,
    ) -> Result<Pipeline> {
        let aniso_pq = AnisotropicPq::pipeline(centroids, section_dim, omega, seed, dim)?;
        let mut stages: Vec<Box<dyn Primitive>> = Vec::new();
        if let Init::Eigen = init {
            stages.extend(OpqP::head(section_dim));
        }
        if iters > 0 {
            stages.push(Box::new(OptimizePq::new(centroids, section_dim, iters, seed)));
        }
        stages.push(Box::new(aniso_pq));
        Pipeline::new(dim, stages)
    }
}

impl Quantizer for AnisotropicOpq {
    fn name() -> &'static str {
        "anisotropic_opq"
    }

    fn display_name() -> &'static str {
        "AnisotropicOPQ"
    }

    fn params() -> &'static [&'static str] {
        &["centroids", "section_dim", "omega", "iters", "init"]
    }

    fn describe() -> &'static str {
        "OptimizePq -> AnisotropicPQ(centroids, section_dim, omega)"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let centroids = get(p, "centroids")?;
        let section_dim = get(p, "section_dim")?;
        let omega = get_or(p, "omega", 0.0f32)?;
        let iters = get_or(p, "iters", DEFAULT_ITERS)?;
        let init = get_or(p, "init", Init::Eigen)?;
        Ok(Self(Self::pipeline(
            centroids,
            section_dim,
            omega,
            iters,
            init,
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
    fn test_anisotropic_opq_neutral_recovers_baseline() {
        let mut rng = math::seed(42);
        let d = 32;
        let v: Array2<f32> = math::gaussian(&mut rng, (60, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (5, d));

        let p = params(&[
            ("centroids", json!(16)),
            ("section_dim", json!(8)),
            ("omega", json!(0.0)),
            ("iters", json!(2)),
        ]);
        let codec = AnisotropicOpq::build(&p, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
        let st: f32 = exact.iter().map(|t| t * t).sum();
        assert!(((se / st) - 1.0).abs() < 0.45, "slope {}", se / st);
    }
}
