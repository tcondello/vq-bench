# Phase 2 Research Charter: Domain-Entropy Compression of Code Embeddings

**Program:** Domain-Entropy Compression of Code Embeddings  
**Prepared for:** Quantization Research (`VQ-bench`, `experiment/all-benchmarks`)  
**Status:** Active  

---

## 1. Grounding Thesis

Every quantizer tested — 59 public configurations and all of ours — buys retrieval quality with reconstruction fidelity at one fixed exchange rate, and the headroom along that curve is measured in fractions of a point. The program therefore targets the structure the exchange rate ignores. Phase 1 established *which* structure pays: quantizability is a property of the **source**, not the quantizer (the \(\Gamma\) result: imagenet 1.85× vs laion 1.32× under an identical encoder). Phase 2 pushes that finding to its logical extreme: **narrow, highly structured domains are low-entropy sources, and their embeddings should compress far below generic budgets — with code as the strongest available test case.**

The limit case that frames everything: the embeddings of a text can carry at most the entropy of the text plus the (shared, amortized) model. For rigid, repetitive languages — and source code is the most rigid language humans write at scale — the floor is dramatically lower than for open text, and the endgame designs are dictionary-shaped: recurring tokens map to shared centroids, and only residual, context-specific information costs bits.

**Why this is the right bet commercially as well as scientifically:** the agent-driven code-retrieval market is exploding and is extraordinarily cost-sensitive. Semble runs code search on a 16M-parameter *static* embedding model at 0.854 NDCG@10 — evidence that the code domain tolerates radical model-side compression. Voyage-code-3 ships binary, Matryoshka-truncated embeddings that retain ~91% retrieval quality at **1/96th the storage** — evidence that code embeddings tolerate radical vector-side quantization. Firecrawl's Developer Index spans 70M+ artifacts. Whoever quantifies — with theory and measurement — exactly how few bits code retrieval actually needs, owns the storage-cost argument for that entire market.

**The PII program is parked, deliberately.** Its mathematical core (identifying information is model-incompressible entropy; low-density vectors resist population-fit codebooks) shares machinery with this program, so the code work accrues PII evidence for free. Goal G6 below specifies the passive signal collection; no dedicated PII experiments run in Phase 2.

---

## 2. Assigned Goals

### G0 — Repairs before New Construction *(Completed in Phase 2A)*
Lane 1 reopened with validity gates (reproduced published anisotropic-PQ gain before porting; \(\omega \to 0\) baseline recovery mandatory). Lane 3 reopened with corrected line-item bit accounting and the corpus-centroid + residual construction. Lane 2: reconciled the top-5%-variance discrepancy in-repo and ran the categorical-CLIP forward test (CIFAR-100 CLIP: \(+4.63\%\) and \(+2.18\%\) gains confirmed). Lane 4 certificate and master synthesis rewritten with claims sized to evidence. Gated SpikeSplit and ColBERT quantizers committed.

### G1 — Build `colbert-python-128`, Forecasts First
Using the existing dataset pipeline, embed a Python corpus (\(\ge 250\text{K}\) token vectors from real repositories; hold out \(\ge 20\%\) of *repositories*, not files, for eval) with `colbert-ir/colbertv2.0`. **Before the dataset exists**, commit a forecast file to the repo: predicted quantizability gap \(\Gamma\) (with a band), predicted \(d_{\text{eff}}\), predicted Hopkins statistic, predicted fraction of tokens within \(\epsilon\) of their nearest corpus-centroid (vocabulary redundancy), and a predicted operating point of the form *"recall@10 \(\ge R\) at \(\le B\) bits/dim with corpus-fit coding."* The forecast is the experiment; the dataset merely grades it.

### G2 — Grade the Quantizability Predictor Out-of-Sample
Run the full Lane 2 diagnostic suite on `colbert-python-128` and score it against the G1 forecast. Together with the G0 categorical-CLIP test this gives the \(\Gamma\) predictor its second forward test. Adjudication is symmetric: within-band results strengthen the predictor; out-of-band results are reported as misses against the committed forecast, not re-explained.

### G3 — The Headline Experiment: The Domain-Entropy Dividend
Using the repaired Lane 3 machinery (corpus-level token centroids + \(b\)-bit residuals, ColBERTv2-style; validity gate = reproducing ColBERTv2's published compression quality on msmarco-colbert first), measure the full rate-recall curve on `colbert-python-128` versus `msmarco-colbert-128` under identical protocols. The claim under test, stated as a number: **bits/dim needed to reach matched recall on code vs. general text.** Report the dividend as *"code retrieval at recall \(R\) costs \(X\) b/d where general text costs \(Y\)."* Kill criterion: if code needs \(\ge 80\%\) of the general-text budget at matched recall, the domain-entropy thesis is dead as a practical matter and the writeup says so.

### G4 — Approach the Dictionary Floor
Push toward the limit case: token-vocabulary coding (deduplicate near-identical token vectors; code recurring subword-context pairs as centroid references; charge honest bits for the dictionary, amortized and reported separately, per the bit-accounting rule). Measure how far real coding closes the gap between the G3 operating point and the text-entropy floor estimate. Deliverable: a single chart, bits/dim vs recall, with four curves — independent EDEN, corpus-centroid+residual, dictionary coding, and the estimated floor.

### G5 — Prove It's the Source, Not the Model
Repeat G2's diagnostics and G3's headline measurement with a second, radically different encoder on the same corpus split — `potion-code-16M` (open, static, 16M params) is the designated choice because it maximizes encoder contrast. If the domain-entropy dividend appears under both a late-interaction transformer and a static embedding model, the result is a property of *code*, and generalizes; if it appears under only one, the boundary is a finding in itself. Forecast committed before running, as always.

### G6 — Passive PII Signal Collection
While G3–G5 run, log per-vector quantization error and density diagnostics (distance to nearest centroid, local codebook occupancy) for every corpus processed. Deliverable at phase end: distributions of these statistics with the top-percentile outlier vectors characterized (what kinds of tokens resist the population codebook?). This is the parked PII program's evidence base accruing at zero marginal cost; no detector is built, no claims are made.

### G7 — Consolidate
The Phase 2 writeup: forecasts vs outcomes for G1–G5, the dividend chart, the repaired-lane adjudications from G0, and the negative-results ledger. Upstream: `colbert-python-128` and the corpus-centroid coder as VQ-bench contributions. Every table carries seeds, honest bits, and protocol.

---

## 3. Standing Operating Rules

1. **Closure Requires Countersignature**: A kill criterion firing closes a lane provisionally; the closure becomes final only after an implementation-validity review by the advisor.
2. **Baseline Recovery is Mandatory**: Every method with a knob must reproduce the baseline at the knob's neutral setting (\(\omega=0\)), within seed noise, in the same table.
3. **Existence Proofs Outrank Experiments**: Where the literature demonstrates a hypothesis in some form, the validity gate is reproducing the known result before testing variants.
4. **Published Bit Accounting**: Bit accounting is published per configuration — anchors, residuals, scales, model bits, per line.
5. **Banned Vocabulary**: *Definitive, unshakeable, inviolable, all-time, SOTA, record* — unless the sentence also contains the measurement that licenses it.
