//! CAST_TERNARY: ternary quantization into {-1, 0, +1} using per-vector absmean scale.
//! -
//! Model: input dim d
//! Code for vector x: d 2-bit levels {-1 -> 0, 0 -> 1, +1 -> 2}, then 1 trailing f32 absmean scale
//! Apply: x --> x - hat(x)          (residual for downstream stages)
//! Reconstruct: y --> scale * ternary(x) + y
//! Score: s --> scale * <q, ternary(x)> + s

use ndarray::{Array1, Array2, ArrayView2};

use crate::coding::CodeLayout;
use crate::{coding, math, Primitive};

/// Ternary rounder: rounds coordinates to `{-1, 0, +1}` scaled by per-vector `mean(|x|)`.
pub struct CastTernary;

/// Two bits per coordinate (encoding {-1 -> 0, 0 -> 1, +1 -> 2}).
const BITS: u8 = 2;

/// The code layout: `d` 2-bit levels, plus 1 trailing f32 absmean scale.
fn layout(d: usize) -> CodeLayout {
    CodeLayout::new().bits(d, BITS).scalars(1)
}

/// Decode packed 2-bit ternary levels into real vectors: `scale * {-1, 0, +1}`.
fn decode_ternary(codes: &[&[u8]], d: usize) -> Array2<f32> {
    let (levels, [scales]) = layout(d).unpack::<1>(codes);
    let mut out = Array2::zeros((codes.len(), d));
    for (i, mut row) in out.rows_mut().into_iter().enumerate() {
        let s = scales[i];
        for j in 0..d {
            let t = levels[[i, j]] as f32 - 1.0;
            row[j] = s * t;
        }
    }
    out
}

impl Primitive for CastTernary {
    fn describe() -> &'static str {
        "round vector into {-1, 0, +1} with per-vector absmean scale (BitNet b1.58)"
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        coding::pack_model(vectors.ncols())
    }

    fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let (n, d) = (vectors.nrows(), vectors.ncols());
        let mut levels = Array2::<u32>::zeros((n, d));
        let mut scales = Array1::<f32>::zeros(n);

        for (i, row) in vectors.rows().into_iter().enumerate() {
            let absmean = row.iter().map(|&x| x.abs()).sum::<f32>() / d as f32;
            let inv_scale = if absmean > 1e-12 { 1.0 / absmean } else { 1.0 };
            for j in 0..d {
                let q = (row[j] * inv_scale).round().clamp(-1.0, 1.0) as i32;
                levels[[i, j]] = (q + 1) as u32;
            }
            scales[i] = absmean;
        }

        layout(d).pack(levels.view(), &[scales.view()])
    }

    fn apply(&self, _model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        *vectors -= &decode_ternary(codes, vectors.ncols());
    }

    fn reconstruct(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let d = super::code_dim(model, child_recons);
        let mut out = decode_ternary(codes, d);
        if let Some(child) = child_recons {
            out += &child;
        }
        out
    }

    fn score(
        &self,
        _model: &[u8],
        queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let mut out = math::matmul(queries, decode_ternary(codes, queries.ncols()).t());
        if let Some(child) = child_scores {
            out += &child;
        }
        out
    }

    fn code_bytes(&self, _model: &[u8], in_dim: usize) -> Option<usize> {
        Some(layout(in_dim).byte_len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, refs};
    use ndarray::array;

    #[test]
    fn exact_at_ternary_grid() {
        let v = array![[2.0, -2.0, 2.0, -2.0], [-1.5, 1.5, -1.5, 1.5]];
        let q = array![[1.0, 0.5, -1.0, 2.0]];
        let cast = CastTernary;
        let model = cast.fit(v.view(), None);
        let codes = cast.encode(&model, v.view());
        let r = refs(&codes);
        assert_close(&cast.reconstruct(&model, &r, None), &v, 1e-5);
        assert_close(&cast.score(&model, q.view(), &r, None), &q.dot(&v.t()), 1e-4);
    }

    #[test]
    fn size_accounting() {
        let v = array![[0.1, -0.2, 0.3, 0.4, 0.0]]; // d = 5
        let cast = CastTernary;
        let model = cast.fit(v.view(), None);
        let codes = cast.encode(&model, v.view());
        assert_eq!(codes[0].len(), cast.code_bytes(&[], 5).unwrap());
        // 5 * 2 = 10 bits -> 2 bytes + 4 bytes scale = 6 bytes
        assert_eq!(codes[0].len(), 6);
    }
}
