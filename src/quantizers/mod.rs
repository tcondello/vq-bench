//! Quantizers: the in-tree catalog the harness runs.

pub mod catalog;
mod rotation;

/// Register the in-tree quantizer families: declare each `module => Type` and collect
/// it into the registry. Add a family by writing its module (a type implementing
/// [`Quantizer`](crate::Quantizer)) and adding it here — the only registration edit.
macro_rules! quantizers {
    ($($module:ident => $ty:ident),+ $(,)?) => {
        $(mod $module;)+
        /// Every quantizer family the harness can build.
        pub fn quantizers() -> Vec<catalog::QuantizerSpec> {
            vec![$(catalog::QuantizerSpec::of::<$module::$ty>()),+]
        }
    };
}

quantizers! {
    minmax => MinMax,
    scalar => Scalar,
    qjl => Qjl,
    itq => Itq,
    itq_asym => ItqAsym,
    simhash => SimHash,
    eden_mse => EdenMse,
    eden_prod => EdenProd,
    turboquant_mse => TurboquantMse,
    turboquant_prod => TurboquantProd,
    rabitq => RaBitQ,
    e_rabitq => ERaBitQ,
    pq => Pq,
    opq => Opq,
    opq_p => OpqP,
    spectralquant => SpectralQuant,
    anchorquant => AnchorQuant,
    bitnet => BitNetQuant,
    e8quant => E8Quant,
    specbit => SpecBit,
    turbobitnet => TurboBitNet,
    spike_vq => SpikeVQ,
    polar_e8 => PolarE8,
    aatc => AATCQuant,
    shell_e8 => ShellE8,
    multi_shell_e8 => MultiShellE8,
    leech24 => Leech24,
    spec_shell => SpecShell,
    spike_shell => SpikeShell,
    spec_eden => SpecEden,
    spike_eden => SpikeEden,
    polar_shell => PolarShell,
    leech_eden => LeechEden,
    bitnet_shell => BitNetShell,
    spectral_leech => SpectralLeech,
    opt_shell => OptShell,
    spike_aatc => SpikeAATC,
    turbo_shell => TurboShell,
    grand_hybrid => GrandHybrid,
    manifold_shell => ManifoldShell,
    spike_manifold_shell => SpikeManifoldShell,
    manifold_leech => ManifoldLeech,
    spike_manifold_leech => SpikeManifoldLeech,
    adapt_manifold_shell => AdaptiveManifoldShell,
    adapt_manifold_leech => AdaptiveManifoldLeech,
    spike_adapt_manifold => SpikeAdaptiveManifold,
    hilbert_shell => HilbertShell,
    cascade_shell_eden => CascadeShellEden,
    water_filled_lattice => WaterFilledLattice,
    procrustes_leech => ProcrustesLeech,
    spike_procrustes_shell => SpikeProcrustesShell,
    leech_multishell => LeechMultiShell,
    grand_manifold => GrandManifold,
    fast_spike_eden => FastSpikeEden,
    polar_leech_eden => PolarLeechEden,
    spectral_band_quant => SpectralBandQuant,
    spike_leech_fast => SpikeLeechFast,
    cascade_lattice_eden => CascadeLatticeEden,
    anisotropic_polar_shell => AnisotropicPolarShell,
    spike_water_filled_eden => SpikeWaterFilledEden,
    multiscale_leech => MultiScaleLeech,
    fast_gosset_compand => FastGossetCompand,
    apex_manifold => ApexManifold,
    progressive_eden => ProgressiveEden,
    task_aware_eden => TaskAwareEden,
    hierarchical_shell => HierarchicalShell,
}
