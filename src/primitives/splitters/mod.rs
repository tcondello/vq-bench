//! Splitters: fan-out stages. A [`Split`] adapts a [`Splitter`] + one child pipeline
//! per branch into a single terminal [`Primitive`].

primitives! { Splitter:
    segment => SegmentSplit,
    spectral => SpectralSplit,
}

mod split;
pub use split::Split;
