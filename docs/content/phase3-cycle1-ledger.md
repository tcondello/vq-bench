# Phase 3 Cycle 1 Master Experiment Ledger

**Program:** Phase 3 — Validate the Law, Explore the Territory  
**Cycle:** Cycle 1 (Five Axes Executed across Multi-Seed Sweeps)  

---

## The Rule of Five Ledger

| Run # | Axis | Hypothesis | Configuration | Empirical Result | Adjudication |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | **Attack** | Non-convex concentric shells have $\Gamma \ge 2.2\times$ but break scalar quantization recall ($R_{10} \le 0.50$). | `synthetic-shells-128`, $N=5$ seeds | $R_{10} = \mathbf{0.3527 \pm 0.0042}$ @ 2 b/d, $0.5024$ @ 2.5 b/d | **SUCCESSFUL BOUNDARY MAP** ($\Gamma$ requires convex/cluster-connected topology) |
| **R2** | **Task Unit** | Late-Interaction MaxSim recovers code retrieval sharpness ($R_{10} \ge 0.70$ at 1.35 b/d) by absorbing token noise. | `colbert-python-128`, 2000 multi-token docs, MaxSim | MaxSim $R_{10} = \mathbf{77.90\%}$ @ 1 b/d, $\mathbf{85.25\%}$ @ 2 b/d (vs 23.9% / 27.5% single-vec) | **ACCEPTED** (Task unit resolution proves MaxSim absorbs token noise) |
| **R3** | **Source** | Static-typed Rust code has higher redundancy and lower distortion than Python ($\Gamma \ge 1.80\times$, $\text{MSE} \le 0.220$). | `colbert-rust-128`, $N=5$ seeds | Redundancy = $\mathbf{42.43\%}$ ($\epsilon \le 0.25$), $\text{MSE} = \mathbf{0.1500}$ @ 1.35 b/d (-60.6% vs text) | **ACCEPTED** (Stronger domain-entropy dividend in statically-typed code) |
| **R4** | **Mechanism** | Normalized query margin $\bar{M} = \mu_\Delta / \sigma_s$ explains ranking hardness across diverse tasks. | 7 benchmark datasets, $\mu_\Delta / \sigma_s$ vs $R_{10}$ | $\bar{M}_{\text{Text}} = \mathbf{2.14}$ vs $\bar{M}_{\text{Code}} = \mathbf{0.21}$ (10x sharper margin on text) | **ACCEPTED** (Margin scale $\bar{M}$ physically explains task retrieval difficulty) |
| **R5** | **Application** | Two-stage candidate filter (1.35 b/d) + exact rescoring retains $\ge 90\%$ of uncompressed recall on text. | 1.35 b/d filter $\to$ top-100 exact rescore | Text Rescored $R_{10} = \mathbf{97.98\%}$ ($\mathbf{98.0\%}$ retention with 95.8% memory saving) | **ACCEPTED** (Production feasibility verified; code requires MaxSim filter) |
