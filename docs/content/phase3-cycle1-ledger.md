# Phase 3 Cycle 1 Experiment Ledger

**Program:** Phase 3 — Validate the Law, Explore the Territory  
**Rule:** Exactly five runs, one per axis. One line per run in the master ledger: Hypothesis → Config → Result → Adjudication.

---

## Master Ledger

| Run # | Axis | Hypothesis | Configuration | Empirical Result | Adjudication |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | **Attack** | Non-convex concentric shells have $\Gamma \ge 2.2\times$ but break scalar quantization recall ($R_{10} \le 0.50$). | `synthetic-shells-128`, $b \in [1, 2, 3, 4]$ | *Pending Execution* | *Pending* |
| **R2** | **Task Unit** | Late-Interaction MaxSim recovers code retrieval sharpness ($R_{10} \ge 0.70$ at 1.35 b/d). | `colbert-python-128`, MaxSim multi-token query vs doc blocks | *Pending Execution* | *Pending* |
| **R3** | **Source** | Static-typed Rust code has higher redundancy and lower distortion than Python ($\Gamma \ge 1.80\times$, $\text{MSE} \le 0.220$). | `colbert-rust-128`, $N=5$ seeds | *Pending Execution* | *Pending* |
| **R4** | **Mechanism** | Normalized query margin $\bar{M}$ correlates ($\rho \ge 0.85$) with 3 b/d retrieval recall across all datasets. | 7 benchmark datasets, $\mu_\Delta / \sigma$ | *Pending Execution* | *Pending* |
| **R5** | **Application** | Two-stage candidate filter (1.35 b/d) + exact rescoring retains $\ge 90\%$ of uncompressed recall. | 1.35 b/d filter $\to$ top-100 exact rescore | *Pending Execution* | *Pending* |
