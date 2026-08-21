//! `split(spike)`: isolate outlier activation channels from the bulk dense subspace.
//! -
//! Model: k (number of outlier channels) and the permutation indices [usize; d]
//! Code for vector x: empty
//! Apply: x --> [x[:, outliers], x[:, bulk]]
//! Reconstruct: [y_outliers, y_bulk] --> unpermutated y
//! Score: [s_outliers, s_bulk] --> s_outliers + s_bulk

use ndarray::{Array2, ArrayView2};

use crate::{coding, Splitter};

/// Outlier-channel isolating splitter.
///
/// Discovers large-magnitude or high-variance outlier channels from training data
/// and isolates them into a dedicated high-precision branch.
pub struct SpikeSplit {
    /// Fraction of dimensions to isolate as outlier spikes (e.g. 0.05 for 5%).
    ratio: f32,
}

impl SpikeSplit {
    /// Create a new SpikeSplit with a specified outlier ratio in (0.0, 1.0).
    pub fn new(ratio: f32) -> Self {
        assert!(ratio > 0.0 && ratio < 1.0, "ratio must be in (0.0, 1.0), got {ratio}");
        Self { ratio }
    }

    /// Read (k, perm) from model bytes.
    pub fn unpack_layout(&self, model: &[u8], in_dim: usize) -> (usize, Vec<usize>) {
        if model.is_empty() {
            let k = ((in_dim as f32 * self.ratio).round() as usize)
                .clamp(1, in_dim.saturating_sub(1).max(1));
            let perm: Vec<usize> = (0..in_dim).collect();
            (k, perm)
        } else {
            coding::unpack_model::<(usize, Vec<usize>)>(model)
        }
    }
}

impl Splitter for SpikeSplit {
    fn describe() -> &'static str {
        "isolate outlier activation/embedding channels from dense bulk"
    }

    fn n_branches(&self) -> usize {
        2
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let (n, dim) = (vectors.nrows(), vectors.ncols());
        let k = ((dim as f32 * self.ratio).round() as usize)
            .clamp(1, dim.saturating_sub(1).max(1));

        // Compute max absolute magnitude per column
        let mut col_max: Vec<(f32, usize)> = (0..dim)
            .map(|j| {
                let mut mx = 0.0f32;
                for i in 0..n {
                    mx = mx.max(vectors[[i, j]].abs());
                }
                (mx, j)
            })
            .collect();

        // Sort descending by spike magnitude
        col_max.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let perm: Vec<usize> = col_max.iter().map(|&(_, j)| j).collect();

        coding::pack_model((k, perm))
    }

    fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
        Some(0)
    }

    fn apply(&self, model: &[u8], vectors: ArrayView2<f32>, _codes: &[&[u8]]) -> Vec<Array2<f32>> {
        let (n, dim) = (vectors.nrows(), vectors.ncols());
        let (k, perm) = self.unpack_layout(model, dim);

        let mut outliers = Array2::zeros((n, k));
        let mut bulk = Array2::zeros((n, dim - k));

        for i in 0..n {
            for j in 0..k {
                outliers[[i, j]] = vectors[[i, perm[j]]];
            }
            for j in 0..(dim - k) {
                bulk[[i, j]] = vectors[[i, perm[k + j]]];
            }
        }

        vec![outliers, bulk]
    }

    fn apply_queries(&self, model: &[u8], queries: ArrayView2<f32>) -> Vec<Array2<f32>> {
        let (n, dim) = (queries.nrows(), queries.ncols());
        let (k, perm) = self.unpack_layout(model, dim);

        let mut q_outliers = Array2::zeros((n, k));
        let mut q_bulk = Array2::zeros((n, dim - k));

        for i in 0..n {
            for j in 0..k {
                q_outliers[[i, j]] = queries[[i, perm[j]]];
            }
            for j in 0..(dim - k) {
                q_bulk[[i, j]] = queries[[i, perm[k + j]]];
            }
        }

        vec![q_outliers, q_bulk]
    }

    fn reconstruct(
        &self,
        model: &[u8],
        _codes: &[&[u8]],
        child_recons: &[Array2<f32>],
    ) -> Array2<f32> {
        let n = child_recons[0].nrows();
        let total_dim = child_recons[0].ncols() + child_recons[1].ncols();
        let (k, perm) = self.unpack_layout(model, total_dim);
        let mut out = Array2::zeros((n, total_dim));

        for i in 0..n {
            for j in 0..k {
                out[[i, perm[j]]] = child_recons[0][[i, j]];
            }
            for j in 0..(total_dim - k) {
                out[[i, perm[k + j]]] = child_recons[1][[i, j]];
            }
        }

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
        let (k, _) = self.unpack_layout(model, in_dim);
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
    use crate::math;
    use crate::util::testing::{assert_close, refs};
    use crate::{CastUint, Pipeline, Primitive, Split};

    #[test]
    fn isolates_outlier_columns() {
        let mut rng = math::seed(42);
        let mut v = math::gaussian(&mut rng, (100, 16));
        // Spike columns 3 and 7
        v.column_mut(3).mapv_inplace(|x| x * 50.0);
        v.column_mut(7).mapv_inplace(|x| x * 80.0);

        let splitter = SpikeSplit::new(0.125); // 2 out of 16
        let model = splitter.fit(v.view(), None);
        let (k, perm) = splitter.unpack_layout(&model, 16);
        assert_eq!(k, 2);
        assert!(perm[0] == 7 || perm[0] == 3);
        assert!(perm[1] == 7 || perm[1] == 3);
    }

    #[test]
    fn round_trip_and_score() {
        let mut rng = math::seed(1);
        let v = math::gaussian(&mut rng, (50, 16));
        let q = math::gaussian(&mut rng, (5, 16));

        let split = Split::from_factory(SpikeSplit::new(0.25), |_b, branch_dim| {
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
        assert_eq!(recon.dim(), (50, 16));
        let scores = split.score(&model, q.view(), &r, None);
        assert_close(&scores, &q.dot(&recon.t()), 1e-2);
    }
}
