# Quantization Research Charter

## The Grounding Thesis

Every quantizer ever tested on this benchmark — fifty-nine public configurations and twenty of ours, across every geometric family anyone has tried — buys retrieval quality with reconstruction fidelity at the same exchange rate: they all sit on one recall-versus-distortion curve, and the information-theoretic headroom left along that curve is measured in fractions of a point. **The research program is therefore not to move along that curve, but to find and exploit the structure the curve ignores.** Everything the exchange rate is blind to is an open lane: the *task's* structure (ranking margins, not global error), the *data's* structure (the joint, coordinate-aligned regularities that made imagenet quantize differently from its statistical twin), the *corpus's* structure (correlation across a document's token vectors, which every method today ignores by encoding vectors independently), and the *system's* structure (retrieval happens in stages and at multiple rates, while today's codes serve exactly one of each).

One sentence to ground every decision: **if an experiment cannot, even in principle, place a point above the recall-versus-distortion envelope or explain why nothing can, it is not part of this program.**

---

## The Single Success Measure

**Height above the envelope**: recall gained at *matched measured distortion and matched honest bits*, evaluated on held-out queries, mean ± spread over $\ge 5$ seeds. This is the loop's objective function and the referee for every claim. It cannot be gamed by spending bits (the envelope is distortion-indexed), by diagonal comparisons (matching is required), or by query overfitting (held-out is required).

---

## The Four Lanes

### Lane 1: Task-Aware Coding (Highest Ceiling)
* **Hypothesis**: Codes that preserve score margins near ranking decision boundaries beat MSE-optimal codes at equal bits — the envelope exists only because everyone optimizes the wrong loss.
* **First Experiment**: Anisotropic score-aware loss (ScaNN-style) inside the existing EDEN pipeline.
* **Kill Criterion**: $<+1$ point above the envelope at 2–4 b/d on three datasets within two weeks $\to$ lane closes.

### Lane 2: A Theory of Quantizability (Highest Scientific Value)
* **Hypothesis**: What separates imagenet from laion — statistically indistinguishable marginals, opposite responses — is measurable joint structure, plausibly the gap between the data's achievable distortion and the Gaussian bound at matched covariance. The anomaly now has three independent confirmations: SpikeEden's channel routing (+1.9 to +8.2 at mid rates), and HierarchicalShell's shell hierarchy (+13.2 at 1 b/d), both win dramatically on imagenet-clip-512 and lose everywhere else, by unrelated mechanisms invisible to marginal statistics.
* **First Experiment**: $k$-means pilot distortion vs Gaussian bound on all eleven datasets, correlated with measured deltas; plus one pre-registered categorical-CLIP prediction (Food-101 or CIFAR-100: call the outcome before running).
* **Kill Criterion**: If neither statistic separates the matched pair, the mechanism goes in the paper as *unknown*, and the fit-time gate stands as the permanent answer.

### Lane 3: Joint Coding of Multi-Vector Documents (Most Pinecone-Shaped)
* **Hypothesis**: A document's token embeddings are highly correlated, and coding them jointly beats independent coding at equal bits — a regime no method on the board can express.
* **First Experiment**: Within-document residual or shared-centroid coding on the ColBERT dataset.
* **Kill Criterion**: $<+2$ points at matched b/d over independent EDEN $\to$ lane closes.

### Lane 4: Hierarchical, Stage-Aware Codes (Progressive Bit-Plane Residuals on EDEN-prod)
* **Hypothesis**: H3's real virtues — prefix hierarchy and cheap neighborhoods — matter where the headroom math says geometry still can: coarse candidate generation below 1.5 b/d, and one code serving many rates.
* **Status**: 
  - Test (a), candidate generation vs QJL and RaBitQ, ran and **failed** (1 win — imagenet, now lane-2 evidence — against 2 losses; closed).
  - Test (b) as first run was tautological — the "retrained" codes were the truncations by construction ($\Delta = \text{exactly } 0.00$ everywhere), verifying an implementation property, not the hypothesis.
  - The real metric is the **price of progressiveness**: one truncatable index vs the *frontier* code retrained at each rate. On the shell code that price is 2–8 points — no bargain.
* **The Lane's One Remaining Pre-Registered Run**: Build progressive bit-plane residuals on top of the frontier method (`EDEN-prod`), truncate $4 \to 2 \to 1$ b/d, and measure recall penalty vs retrained EDEN at each rate, on three datasets.
* **Kill Criterion**: 
  - Penalty $\ge 1$ point at any rate $\to$ lane closes completely, certificate airtight (packing headroom bounded, SFC locality dead, candidate generation lost 2-of-3, progressiveness priced out on the best code).
  - Penalty $< 1$ point everywhere $\to$ the systems contribution survives: one index, every rate tier, built on the frontier.

---

## Operating Rules (The Discipline That Made the Good Day Good)

1. **Bit-Matched or Envelope-Indexed**: Comparisons ship bit-matched or envelope-indexed, never diagonal.
2. **Pre-Registered Kill Criteria**: Every hypothesis is written down, with its kill criterion, *before* the run — and the run adjudicates on the exact metric the criterion names, not a neighboring one.
3. **Falsifiability by Construction**: A test must be able to fail: if a design makes the pass outcome true by construction, it verifies the implementation, not the hypothesis — exact zeros and perfect scores are the tell, and they trigger a redesign, not a celebration.
4. **One Test for Post-Hoc Hypotheses**: A post-hoc explanation earns exactly one pre-registered test or it dies.
5. **Negative Results Archived by Name**: Negative results are archived by name, never rebranded.
6. **Same-Machine Timings**: Timings come from one machine in one run.
6. **The Instrument Rule (Two-Tier Claims)**: Any claim about a source must either name its encoder (Tier 1: encoder-conditional product claim) or demonstrate invariance across at least two contrasting encoders (Tier 2: encoder-invariant science claim). Instruments are hypotheses too.
7. **Same-Machine Timings**: Timings come from one machine in one run.
8. **Transparent Correction Logs**: Corrections are logged with the bug, the affected numbers, and the new values.
9. **Consolidation is Continuous**: Results get written up when a lane closes, because the writeups are the maps the next lane runs on.
10. **Closure Requires Countersignature**: A kill criterion firing closes a lane provisionally; the closure becomes final only after an implementation-validity review by the advisor. Kill criteria protect against wishful positives; they do not protect against strawman negatives.
11. **Baseline Recovery is Mandatory**: Every method with a knob must reproduce the baseline at the knob's neutral setting (e.g., \(\omega=0\)), within seed noise, in the same table. No recovery, no adjudication.
12. **Existence Proofs Outrank Experiments**: Where the literature demonstrates a hypothesis in some form (ScaNN's anisotropic gain, ColBERTv2's residual compression), the validity gate is reproducing the known result before testing the variant. Failure to reproduce indicts the implementation, not the hypothesis.
13. **Published Bit Accounting**: Bit accounting is published per configuration — anchors, residuals, scales, model bits, per line — and configurations that must differ (\(L=16\) vs \(L=32\)) must differ.
14. **Banned Vocabulary**: *Definitive, unshakeable, inviolable, all-time, SOTA, record* — unless the sentence also contains the measurement that licenses it.

---

## What "Earth-Shattering" Means Here

* A point above the envelope.
* A validated predictor of quantizability.
* A joint-coding regime the benchmark cannot currently express.
* A single code that serves every rate and stage of retrieval.

Any one of these changes how vector databases store embeddings; all four are open; none of them is reachable by inventing another point on the curve.

