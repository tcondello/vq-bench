//! ANISOTROPIC_KMEANS(k, omega): ScaNN anisotropic score-aware k-means codebook
//! -
//! Minimizes ScaNN anisotropic loss:
//! L_aniso(p, c) = ||p - c||^2 + omega * (||p|| - <p, c>/||p||)^2
//! Recovers standard Euclidean k-means continuously as omega -> 0.

use ndarray::{Array1, Array2, ArrayView2};

use crate::coding::CodeLayout;
use crate::{coding, math, Primitive};

pub struct AnisotropicKmeans {
    centroids: usize,
    omega: f32,
    seed: u64,
}

impl AnisotropicKmeans {
    pub fn new(centroids: usize, omega: f32, seed: u64) -> Self {
        debug_assert!((2..=256).contains(&centroids));
        Self {
            centroids,
            omega,
            seed,
        }
    }

    fn index_bits(&self) -> u8 {
        (self.centroids as u32).next_power_of_two().trailing_zeros() as u8
    }

    fn centroids(model: &[u8]) -> Array2<f32> {
        let (_dim, centroids): (usize, Array2<f32>) = coding::unpack_model(model);
        centroids
    }

    fn layout(&self) -> CodeLayout {
        CodeLayout::new().bits(1, self.index_bits())
    }

    /// Assign points to nearest centroid under ScaNN anisotropic loss
    pub fn nearest_anisotropic(
        points: ArrayView2<f32>,
        centroids: ArrayView2<f32>,
        omega: f32,
    ) -> Array1<u32> {
        let dots = math::matmul(points, centroids.t());
        let cnorms: Array1<f32> = centroids.rows().into_iter().map(|c| c.dot(&c)).collect();
        let pnorms_sq: Array1<f32> = points.rows().into_iter().map(|p| p.dot(&p).max(1e-9)).collect();

        dots.outer_iter()
            .enumerate()
            .map(|(i, row)| {
                let p_norm_sq = pnorms_sq[i];
                let mut best = 0usize;
                let mut best_val = f32::INFINITY;
                for (k, &d) in row.iter().enumerate() {
                    // Loss = ||c_k||^2 - 2(1 + omega) <p, c_k> + omega * <p, c_k>^2 / ||p||^2
                    let val = cnorms[k] - 2.0 * (1.0 + omega) * d + omega * (d * d) / p_norm_sq;
                    if val < best_val {
                        best_val = val;
                        best = k;
                    }
                }
                best as u32
            })
            .collect()
    }

    /// Train k-means with anisotropic Lloyd iterations
    fn fit_anisotropic(
        points: ArrayView2<f32>,
        k: usize,
        omega: f32,
        seed: u64,
    ) -> Array2<f32> {
        // Initial standard Lloyd k-means
        let mut centroids = math::lloyd_kmeans(points, k, 15, seed);
        if omega <= 1e-6 {
            return centroids;
        }

        let d = points.ncols();
        let iters = 10;

        for _ in 0..iters {
            let assignments = Self::nearest_anisotropic(points, centroids.view(), omega);
            let mut counts = vec![0usize; k];
            let mut sum_p = Array2::<f32>::zeros((k, d));

            for (i, &c_idx) in assignments.iter().enumerate() {
                let c = c_idx as usize;
                counts[c] += 1;
                let p = points.row(i);
                for j in 0..d {
                    sum_p[[c, j]] += p[j];
                }
            }

            for c in 0..k {
                if counts[c] > 0 {
                    let factor = 1.0 / (counts[c] as f32);
                    for j in 0..d {
                        centroids[[c, j]] = sum_p[[c, j]] * factor;
                    }
                }
            }
        }

        centroids
    }

    fn dequant(&self, model: &[u8], codes: &[&[u8]]) -> Array2<f32> {
        let centroids = Self::centroids(model);
        let (idx, []) = self.layout().unpack::<0>(codes);
        let mut out = Array2::zeros((codes.len(), centroids.ncols()));
        for (i, row) in idx.outer_iter().enumerate() {
            out.row_mut(i).assign(&centroids.row(row[0] as usize));
        }
        out
    }
}

impl Primitive for AnisotropicKmeans {
    fn describe() -> &'static str {
        "round to nearest centroid in an anisotropic score-aware codebook"
    }

    fn in_dim(&self) -> Option<usize> {
        None
    }

    fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
        Some(self.layout().byte_len())
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let d = vectors.ncols();
        let centroids = Self::fit_anisotropic(vectors, self.centroids, self.omega, self.seed);
        coding::pack_model((d, centroids))
    }

    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let centroids = Self::centroids(model);
        let idx = Self::nearest_anisotropic(vectors, centroids.view(), self.omega);
        self.layout().pack(idx.into_shape_with_order((vectors.nrows(), 1)).unwrap().view(), &[])
    }

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        let recon = self.dequant(model, codes);
        *vectors -= &recon;
    }

    fn reconstruct(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let mut out = self.dequant(model, codes);
        if let Some(child) = child_recons {
            out += &child;
        }
        out
    }

    fn score(
        &self,
        model: &[u8],
        queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let centroids = Self::centroids(model);
        let (idx, []) = self.layout().unpack::<0>(codes);
        let table = math::matmul(queries, centroids.t());
        let mut scores = Array2::zeros((queries.nrows(), codes.len()));
        for (j, row) in idx.outer_iter().enumerate() {
            let c = row[0] as usize;
            for i in 0..queries.nrows() {
                scores[[i, j]] = table[[i, c]];
            }
        }
        if let Some(child) = child_scores {
            scores += &child;
        }
        scores
    }
}
