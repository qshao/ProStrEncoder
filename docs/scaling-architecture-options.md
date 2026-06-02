# ProStrEncoder — Scaling Architecture Options

**Date:** 2026-06-02  
**Context:** Evaluating architectural directions for scaling ProStrEncoder from 2.2M to ~1B parameters.  
**Compute target:** 2–4 × H200 (single node, DDP/FSDP)  
**Primary downstream tasks:** Function prediction (GO terms, EC numbers, binding sites) and sequence-structure design (inverse folding, mutation effect prediction)  
**Pre-training data:** PDB (~220K structures) first; AFDB augmentation later

---

## Current Architecture (Baseline)

- **2.2M parameters**
- **GVP-GNN backbone** (3 layers, hidden_scalar=128, hidden_vector=16) — SE(3)-equivariant message passing
- **WalkEncoder** (4-layer transformer, embed_dim=64) — topology + AA-type context from random walks
- **Pre-training objective:** BERT-style masked residue prediction (15% mask rate, cross-entropy)
- **Output:** (N, 512) rotation-invariant per-residue embeddings

---

## Option A — Scale Depth and Width of the Existing GVP-GNN + WalkEncoder

### What It Is

Increase the hidden dimensions and number of layers within the current architecture:
- `hidden_scalar`: 128 → 512–1024  
- `hidden_vector`: 16 → 64–128  
- `num_layers`: 3 → 12–24  
- `walk_embed_dim`: 64 → 256  
- `walk_layers`: 4 → 12  

~1B parameters is reachable with `hidden_scalar=1024, hidden_vector=128, num_layers=20`.

### Advantages

- **Lowest implementation risk.** All training infrastructure already works; only config values change.
- **SE(3)-equivariance maintained throughout.** Theoretically principled for 3D structure tasks.
- **Walk encoder naturally captures sequence-topology context,** which benefits function prediction for residues in structurally conserved motifs.
- **Predictable iteration cycle.** No new modules to debug; scaling experiments can start immediately.
- **GVP has been validated** at moderate scale in the original GVP-GNN paper (Jing et al. 2021).

### Concerns

- **Sum-aggregation has an expressiveness ceiling.** GVP-GNN uses fixed graph-structure-weighted aggregation; adding layers yields diminishing returns faster than attention-based architectures, which learn to weight neighbors adaptively.
- **No long-range residue communication.** A catalytic residue 200 positions away from an active-site loop cannot directly influence it — information must propagate one hop per layer. At 20 layers with k=30 neighbors this covers a large graph neighborhood but is still hop-limited and diluted.
- **Long-range context is critical for function prediction.** Binding sites, allosteric communication pathways, and domain interfaces are global properties. A local aggregation model captures them poorly regardless of depth.
- **Scaling GVP-GNN beyond ~12 layers is largely unexplored.** Gradient flow, representational collapse, and overfitting behavior at that depth are unknown for protein graphs.
- **Limited benefit from Flash Attention / FSDP.** The GVP message-passing kernels are not attention-based, so GPU efficiency tools designed for transformers provide little uplift. Memory scaling is manual.

---

## Option B — Replace GVP with a Full Equivariant Transformer

### What It Is

Replace `GVPConv` layers with attention-based equivariant layers such as **Equiformer** (equivariant multi-head attention on irreducible representations of SO(3)) or **SE(3)-Transformer**. Each residue attends to all others with geometry-aware, equivariant attention weights. Requires a full rewrite of `gvp.py` and the GNN backbone.

### Advantages

- **Attention is the correct inductive bias for function prediction.** An active-site residue can directly attend to a catalytic partner 300 sequence positions away in a single layer — no hop limits.
- **Theoretically the most expressive equivariant architecture available.** Higher-order spherical harmonics in Equiformer encode richer geometric relationships than GVP's vector features.
- **Long-range dependencies captured in O(1) layers** rather than O(depth), which means shallower models can be more expressive.
- **Aligns with the current research frontier.** AlphaFold2's IPA, Equiformer2, FrameDiff, and RFdiffusion all use attention-based equivariant architectures; building in this direction is consistent with where the field is heading.

### Concerns

- **O(N²) memory and compute in sequence length.** Full equivariant attention over N residues requires N² pair computations per layer. For a 500-residue protein this is 250K pairs; for proteins >1000 residues this becomes prohibitive without sparse attention approximations, which add further complexity.
- **High implementation complexity.** Correct equivariant attention with spherical harmonic features (Equiformer) is substantially harder to implement and debug than GVP. Custom CUDA kernels are typically needed for practical speed.
- **Scaling behavior on proteins is under-explored.** Most Equiformer papers target small molecules (<100 atoms). How the architecture behaves at protein scale (hundreds to thousands of residues) in terms of expressiveness, stability, and sample efficiency is an open research question.
- **Longest path to a working prototype.** Estimated 4–8 weeks of engineering before the first training run — high risk for a research group with limited engineering bandwidth.
- **Standard GPU efficiency tools do not apply directly.** Flash Attention 2 and FSDP work with standard transformers; equivariant attention requires adapted or custom implementations.

---

## Option C — Hybrid: GVP Local Encoder + Global Transformer (Recommended)

### What It Is

Keep the GVP-GNN layers for **local geometry encoding** — they are well-validated, fast, and correctly equivariant. After GVP produces per-residue invariant scalar representations, add a **standard multi-head self-attention transformer** (pre-norm, RoPE or ALiBi positional encoding) over all N residue representations. The global transformer operates on invariant scalar features output by GVP — equivariance is not required at this stage because GVP already extracted rotation-invariant representations.

The walk encoder output continues to augment GVP input as in the current design. The transformer layer on top handles global context.

### Advantages

- **Best of both worlds.** GVP handles the hard part (3D geometry, equivariance) efficiently; the standard transformer handles global context with proven scaling laws and mature tooling.
- **Flash Attention 2, FSDP, gradient checkpointing all work out of the box.** The global transformer is standard `nn.TransformerEncoderLayer` — every GPU efficiency technique designed for LLMs applies directly.
- **Proven scaling laws.** Transformer depth/width scaling is well understood from GPT, BERT, Chinchilla, and ESM literature. You get predictable, reliable returns from adding layers or widening hidden dimensions.
- **Directly captures long-range residue interactions.** Binding sites, allosteric effects, and domain interfaces — the key drivers of function prediction — are captured by attention in O(1) layers regardless of protein length.
- **Walk encoder + GVP already provide rich invariant features.** The transformer on top is a well-conditioned learning problem with strong local geometry already encoded.
- **Scales from 2.2M → 1B+ primarily through the transformer.** Parameter counting is straightforward: a 24-layer, 1024-dim transformer over 512-dim GVP outputs is ~860M parameters — well within 4 × H200.
- **Inference flexibility.** For fast deployment, the global transformer can be distilled into a GVP-only model, or the transformer layers can be pruned. The GVP layers can serve as a lightweight standalone encoder.
- **Closest architectural parallel to ESM-2.** ESM-2 is a pure sequence transformer; adding GVP for 3D geometry on top is a natural extension that the research community will recognize and compare to.

### Concerns

- **Global transformer is O(N²) in protein length.** For very long proteins (>2000 residues), standard attention becomes memory-prohibitive. Mitigation: windowed attention, sparse attention (Longformer-style), or simply truncating at a maximum length during pre-training.
- **Two architectural phases require care in training.** GVP local features must be meaningful inputs for the transformer. If GVP is poorly initialized or undertrained, the transformer sees noisy inputs and converges slowly. **Mitigation:** staged training — first pre-train GVP alone for a few epochs, then add the transformer and continue jointly.
- **Equivariance holds only at the GVP level.** The global transformer operates on invariant scalars, not equivariant vectors — so the final output is invariant but not equivariant. This is correct for function prediction (function is invariant to rotation) but means the model cannot directly output equivariant quantities (e.g., 3D coordinates for design tasks without a separate decoder).
- **Slightly more implementation work than Option A** — new module for the global transformer, integration with existing encoder, training schedule changes.

---

## Comparison Table

| Criterion | Option A | Option B | Option C |
|-----------|----------|----------|----------|
| Long-range residue context | Weak (hop-limited) | Strong | Strong |
| SE(3)-equivariance | Full throughout | Full throughout | Local (GVP layers only) |
| Implementation risk | Low | High | Medium |
| Path to 1B parameters | Possible, uncertain | Possible, expensive | Well-charted |
| Flash Attention / FSDP benefit | Limited | Requires custom kernels | Full benefit |
| Function prediction suitability | Moderate | High | High |
| Design task suitability | Moderate | High | High |
| Time to first working prototype | 1–2 weeks | 4–8 weeks | 2–4 weeks |
| Scaling behavior known | Partially | Poorly | Well understood |

---

## Recommendation

**Option C** is the recommended path for the stated goals (function prediction + design, 2–4 × H200, PDB scale).

Function prediction and design both require global context — binding sites and allosteric pathways are intrinsically long-range. Option A cannot capture this well regardless of depth. Option B captures it correctly but carries prohibitive implementation risk and unknown protein-scale behavior. Option C achieves the same long-range expressiveness as Option B using standard transformer technology, benefits fully from modern GPU efficiency tooling, and follows a well-understood scaling trajectory — at materially lower engineering risk than Option B.

The staged training strategy (GVP pre-training → joint fine-tuning) de-risks the two-phase architecture and is consistent with how ESM-2 and similar models were developed.
