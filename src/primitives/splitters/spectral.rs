//! `split(spectral)`: slice each vector into semantic and tail spectral subspaces.
//! -
//! Model: k, the semantic subspace dimension (learned from participation ratio if not fixed)
//! Code for vector x: empty
//! Apply: x --> [x[..k], x[k..]]
//! Reconstruct: [y_sem, y_tail] --> [y_sem; y_tail]
//! Score: [s_sem, s_tail] --> s_sem + s_tail

use ndarray::{s, Array2, ArrayView2, Axis};

use crate::{coding, Splitter};

/// Split a `d`-dim vector into a leading semantic subspace (`k` columns) and a
/// trailing tail subspace (`d - k` columns). `k` is either fixed at construction or
/// learned dynamically at `fit` time from the participation-ratio effective rank
/// `d_eff = (sum lambda_i)^2 / sum lambda_i^2`.
pub struct SpectralSplit {
    /// Fixed semantic dimension if specified, or `None` if learned from data at fit.
    fixed_k: Option<usize>,
}

impl SpectralSplit {
    /// Dynamic spectral splitter that computes `k = ceil(d_eff)` at `fit` time.
    pub fn new_dynamic() -> Self {
        Self { fixed_k: None }
    }

    /// Fixed spectral splitter with a predetermined semantic dimension `k`.
    pub fn new_fixed(k: usize) -> Self {
        assert!(k > 0, "semantic dimension must be > 0");
        Self { fixed_k: Some(k) }
    }

    /// Read the fitted semantic dimension `k` from model bytes.
    pub fn k(&self, model: &[u8], in_dim: usize) -> usize {
        if model.is_empty() {
            self.fixed_k.unwrap_or(1).min(in_dim.saturating_sub(1).max(1))
        } else {
            coding::unpack_model::<usize>(model).min(in_dim.saturating_sub(1).max(1))
        }
    }
}

impl Default for SpectralSplit {
    fn default() -> Self {
        Self::new_dynamic()
    }
}

impl Splitter for SpectralSplit {
    fn describe() -> &'static str {
        "slice each vector into semantic (top-k) and tail subspaces"
    }

    fn n_branches(&self) -> usize {
        2
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let dim = vectors.ncols();
        let k = if let Some(k_fixed) = self.fixed_k {
            k_fixed.min(dim.saturating_sub(1).max(1))
        } else {
            // Participation-ratio effective dimension over column variances
            let var = vectors.var_axis(Axis(0), 0.0);
            let sum_v: f32 = var.sum();
            let sum_sq: f32 = var.iter().map(|&x| x * x).sum();
            let d_eff = if sum_sq > 0.0 {
                (sum_v * sum_v) / sum_sq
            } else {
                1.0
            };
            (d_eff.ceil() as usize).clamp(1, dim.saturating_sub(1).max(1))
        };
        coding::pack_model(k)
    }

    fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
        Some(0)
    }

    fn apply(&self, model: &[u8], vectors: ArrayView2<f32>, _codes: &[&[u8]]) -> Vec<Array2<f32>> {
        let k = self.k(model, vectors.ncols());
        vec![
            vectors.slice(s![.., ..k]).to_owned(),
            vectors.slice(s![.., k..]).to_owned(),
        ]
    }

    fn apply_queries(&self, model: &[u8], queries: ArrayView2<f32>) -> Vec<Array2<f32>> {
        let k = self.k(model, queries.ncols());
        vec![
            queries.slice(s![.., ..k]).to_owned(),
            queries.slice(s![.., k..]).to_owned(),
        ]
    }

    fn reconstruct(
        &self,
        model: &[u8],
        _codes: &[&[u8]],
        child_recons: &[Array2<f32>],
    ) -> Array2<f32> {
        let k = self.k(model, child_recons[0].ncols() + child_recons[1].ncols());
        let n_v = child_recons[0].nrows();
        let total_dim = k + child_recons[1].ncols();
        let mut out = Array2::zeros((n_v, total_dim));
        out.slice_mut(s![.., ..k]).assign(&child_recons[0]);
        out.slice_mut(s![.., k..]).assign(&child_recons[1]);
        out
    }

    fn score(
        &self,
        _model: &[u8],
        _codes: &[&[u8]],
        _query: ArrayView2<f32>,
        child_scores: &[Array2<f32>],
    ) -> Array2<f32> {
        &child_scores[0] + &child_scores[1]
    }

    fn branch_in_dim(&self, model: &[u8], in_dim: usize, branch: usize) -> usize {
        let k = self.k(model, in_dim);
        if branch == 0 {
            k
        } else {
            in_dim - k
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, refs};
    use crate::{math, CastUint, Pipeline, Primitive, Split};
    use ndarray::Array2;

    fn make_skewed_data(n: usize, d: usize) -> Array2<f32> {
        let mut v = math::gaussian(&mut math::seed(1), (n, d));
        // Make first 2 columns have large variance
        v.column_mut(0).mapv_inplace(|x| 10.0 * x);
        v.column_mut(1).mapv_inplace(|x| 8.0 * x);
        v
    }

    #[test]
    fn dynamic_fit_learns_effective_dimension() {
        let v = make_skewed_data(1000, 32);
        let splitter = SpectralSplit::new_dynamic();
        let model = splitter.fit(v.view(), None);
        let k = splitter.k(&model, 32);
        assert!((1..=5).contains(&k), "expected small d_eff for skewed data, got {k}");
    }

    #[test]
    fn fixed_split_round_trips() {
        let v = make_skewed_data(100, 16);
        let q = math::gaussian(&mut math::seed(2), (5, 16));
        let split = Split::from_factory(SpectralSplit::new_fixed(4), |_b, branch_dim| {
            Pipeline::new(
                branch_dim,
                vec![Box::new(CastUint::new(4)) as Box<dyn Primitive>],
            )
            .unwrap()
        });

        let model = split.fit(v.view(), None);
        let codes = split.encode(&model, v.view());
        let r = refs(&codes);
        let recon = split.reconstruct(&model, &r, None);
        assert_eq!(recon.dim(), (100, 16));
        let scores = split.score(&model, q.view(), &r, None);
        assert_close(&scores, &q.dot(&recon.t()), 1e-2);
    }
}
