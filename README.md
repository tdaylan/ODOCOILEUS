# ODOCOILEUS-Σ

**Connectome-Constrained Whole-Brain Embodiment of *Odocoileus virginianus* in a Closed-Loop Sensorimotor Virtual Environment**

`v0.1.0-alpha` · `numpy` + `scipy` only · 5 files · no GPU required

---

## Abstract

We specify a pipeline for inducing a signed, weighted, directed multigraph
**G = (V, E, σ, w)** over the cervid central nervous system, instantiating it
as a conductance-free leaky integrate-and-fire (LIF) dynamical system **Φ_G**,
and coupling Φ_G to a rigid-body proxy of the animal through a bilateral
Poisson afferent encoder **𝓔** and a low-pass population-rate efferent decoder
**𝓓**. The construction follows the logic of whole-brain connectome-to-dynamics
work in *Drosophila* (FlyWire → LIF → NeuroMechFly-class embodiment), with one
decisive substitution: because no synaptic-resolution cervid connectome exists,
**G is sampled from a mesoscale generative prior** (degree-corrected stochastic
block model, DC-SBM, with hemispheric projection bias) rather than measured.
The repository therefore realizes the *inference-and-embodiment scaffold* and
leaves the *measurement stage* as an explicit, documented interface.

---

## 0. Epistemic status (read first)

| Claim | Status |
|---|---|
| A deer connectome has been imaged / reconstructed | **False.** None exists. |
| `connectome.py` outputs a real deer wiring diagram | **False.** It samples a *prior*. |
| The simulated agent exhibits deer-like cognition | **False.** Weights are unfitted; behavior is a plumbing test. |
| The pipeline from a graph `W` to embodied closed-loop dynamics runs end-to-end | **True.** `python run.py` |
| Swapping in a measured `W` requires no changes downstream | **True by construction** (§5). |

Treat this repo as a *typed interface plus a null model*, not a result.

---

## 1. Lineage and problem statement

Connectome-constrained simulation has been demonstrated at three scales:

* ***C. elegans*** — |V| = 302, |E| ≈ 10³ (White et al., 1986; OpenWorm lineage).
* ***Drosophila*** — |V| ≈ 1.4 × 10⁵, |E| ≈ 5 × 10⁷ synapses (FlyWire; Dorkenwald et al., 2024),
  with LIF whole-brain models reproducing sensorimotor circuit predictions
  (Shiu et al., 2024) and biomechanical bodies (NeuroMechFly; Lobato-Rios et al., 2022).
* ***Mus musculus* (cubic-mm fragment)** — ~10⁵ cells, ~10⁸–10⁹ synapses (MICrONS), i.e. a *volume*, not a whole brain.

**Problem.** Given tissue 𝒯 from species *s*, recover **G_s**, define dynamics
Φ_G, and close the loop through a body **B**:

```
   world W ──𝓔──▶ Φ_G ──𝓓──▶ B ──▶ world W'        (Δt_act = 20 ms, Δt_sim = 0.5 ms)
        ▲                                  │
        └──────────── proprio / exteroception ─────┘
```

**Why deer is qualitatively harder than fly.** The scaling is unfavorable on
every axis simultaneously:

| Quantity | *Drosophila* | *O. virginianus* (order-of-magnitude, **unmeasured**) |
|---|---|---|
| Neurons | ~10⁵ | ~10⁹ (ungulate scaling assumption) |
| Brain volume | ~10⁻¹ mm³ | ~10² cm³ = ~10⁵ mm³ |
| EM volume at ~4×4×40 nm | ~10⁻¹ PB | ~10⁵ PB (~100 EB) |
| Proofreading | crowdsourced, ~10⁷ edits | intractable without ≥10³× automation gain |
| Wiring variance across individuals | low (stereotyped) | high (cortical, plastic) |

The last row is the killer. Insect neurons are (largely) *genetically
identified*, so one reconstruction generalizes. Mammalian cortex is
statistically stereotyped but individually idiosyncratic, so even a perfect
single-deer **G** would be an *instance*, not the *species*. This motivates
sampling from a generative prior instead of chasing a point estimate.

---

## 2. Formal model

### 2.1 Graph

Let N = |V|. Partition V into R = 12 mesoscale compartments
𝒞 = {C_r}, with hemisphere label h_i ∈ {0, 1} and neurotransmitter sign
σ_i ∈ {+1, −1} (Dale's law; Pr[σ_i = −1] = 0.2).

Edge multiplicity for pre *j* → post *i*, with *j* ∈ C_a, *i* ∈ C_b:

```
   M_ij  ~  Poisson-thinned Binomial( n_a n_b , p_ab ) placements,
            weighted by  θ^out_j θ^in_i,     θ ~ LogNormal(0, s²)          (degree correction)

   accept(j→i) = ½ · [ 2(1−κ_ab)·𝟙(h_i = h_j) + 2κ_ab·𝟙(h_i ≠ h_j) ]        (hemispheric bias, κ = contralateral fraction)
```

Duplicate placements sum, so M_ij ∈ ℕ models multi-synaptic contacts.
Signed PSP amplitude matrix:

```
   W_ij = w_syn · σ_j · M_ij         [mV],       w_syn = 0.275 mV (borrowed insect placeholder)
```

### 2.2 Dynamics (Φ_G)

Current-based LIF with a single synaptic state variable **g**:

```
   τ_m  dv_i/dt =  (v_rest − v_i) + g_i
   τ_s  dg_i/dt = −g_i + τ_m Σ_j W_ij s_j(t) + τ_m w_ext ξ_i(t)
   s_i(t) = δ(t − t_k)   if v_i ≥ v_th  →  v_i ← v_reset, refractory t_ref
```

with τ_m = 20 ms, τ_s = 5 ms, v_rest = v_reset = −52 mV, v_th = −45 mV,
t_ref = 2.2 ms, and ξ_i a Poisson process of rate ν_i (afferent).
The τ_m/τ_s prefactor normalizes the *peak* PSP to ≈ W_ij (valid for τ_s ≪ τ_m).
Integration: exact exponential decay for **g**, forward Euler for **v**,
Δt = 0.5 ms.

### 2.3 Sensorimotor interface

**Afferent encoder 𝓔** (`run.py`). Stimulus → rate → Poisson spikes:

```
   ν_i = ν_bg                                   ∀ i ∈ V
   ν_i += G_olf · c_{h(i)}(x_t)                 ∀ i ∈ OB       (c_L, c_R: nostril-resolved plume concentration)
   ν_i += G_vis · ℓ_{β(i)}(x_t)                 ∀ i ∈ LGN      (ℓ_β: looming in retinotopic bin β ∈ {0..11}, ±150° panorama)
```

Laterality: hemisphere 0 ≡ left. Left nostril / left hemifield → hemisphere-0 afferents.

**Efferent decoder 𝓓.** Exponentially filtered population rates (τ_r = 100 ms):

```
   turn   = tanh( (r_{M1,L} − r_{M1,R}) / 5 Hz )         (+ ≡ counter-clockwise)
   fwd    = clip( r_CPG / 10 Hz , 0, 1 )
   flight = 𝟙[ r_MLR > 15 Hz ]                           (gait switch: walk → gallop-proxy)
```

### 2.4 Mesoscale projection prior

Neuron budget and projection densities are **placeholders** chosen to induce
two hardwired sensorimotor arcs; they are *not* anatomical measurements.

| Arc | Path | Lateralization design |
|---|---|---|
| Olfactory taxis | OB → PIR → STR → M1 → CPG | ipsilateral (κ ≈ 0.15–0.20) ⇒ scent-left drives turn-left |
| Predator evasion | LGN ⇒ SC ⇒ M1 (crossed), SC → AMY → MLR → CPG | contralateral (κ = 0.85) ⇒ threat-left drives turn-right; MLR gates flight |
| Cognitive loop (unconstrained) | V1, HPC, PFC, PIR ↔ AMY, STR | κ = 0.5 (no laterality) |

Compartments: `OB` olfactory bulb · `LGN` lateral geniculate · `V1` primary visual ·
`SC` superior colliculus · `PIR` piriform · `AMY` amygdala · `HPC` hippocampus ·
`PFC` prefrontal · `STR` striatum · `M1` motor cortex · `MLR` mesencephalic locomotor
region · `CPG` brainstem/spinal pattern generator (rate proxy).

---

## 3. Full measurement pipeline (specified, **not implemented**)

The stages that would replace the prior with data:

1. **Acquisition.** Perfusion-fixation, heavy-metal en-bloc staining, serial
   ATUM/GCIB or multibeam SEM at ≈ 4 × 4 × 40 nm; complementarily,
   expansion microscopy (ExM) with barcoded-tract sparse labeling for long-range
   projections that EM sectioning cannot economically span.
2. **Volume assembly.** Elastic rigid → nonrigid stitching, then alignment
   minimizing a section-to-section photometric residual.
3. **Segmentation.** Affinity prediction with a 3D U-Net, followed by
   agglomeration (flood-filling networks, Januszewski et al., 2018, or
   watershed + hierarchical merge), yielding supervoxel graph 𝒮.
4. **Synapse & transmitter inference.** Presynaptic-site and cleft detection;
   per-terminal neurotransmitter classification from ultrastructure
   (vesicle morphology) → σ_i.
5. **Proofreading.** Human-in-the-loop split/merge on 𝒮 with error-rate
   budget ε; active learning to route edits to high-influence nodes.
6. **Graph assembly.** Contract 𝒮 → V, aggregate contacts → **M**, threshold
   at a minimum synapse count m_min to control false-positive edges.
7. **Registration to ontology.** Assign each i ∈ V to (C_r, h_i) via
   atlas registration → interfaces directly with `connectome.Connectome`.

The interface contract for stage 7 is precisely the `Connectome` dataclass
(`W`, `slices`, `hemi`, `sign`).

---

## 4. Identifiability

Even with G known, Φ_G is **underdetermined** by structure:

* **Weight–sign–magnitude degeneracy.** Synapse count ≠ efficacy; w_ij is a
  latent. Approaches: fit a low-dimensional cell-type-level gain vector
  **γ** ∈ ℝ^{R×R} against recorded population statistics (*constrain, don't fit
  every edge*), as in structure-constrained-network work on the fly visual system
  (Lappalainen et al., 2024).
* **Neuromodulation** (ACh/NE/5-HT/DA) is a slow, non-synaptic, volume-transmitted
  degree of freedom that a wiring diagram cannot encode.
* **Gap junctions / electrical coupling** are absent from Φ_G.
* **Dendritic computation** is collapsed to a point neuron.

Observed here: the closed-loop network has a **sharp gain transition**.
With this repo's defaults, sweeping ν_bg from 800 → 1000 → 1200 Hz takes
population rates from ≈ 0 Hz → O(1–30 Hz) → saturating (>50 Hz). The default
ν_bg = 1000 Hz sits on the sensitive side of that transition. Treat this as an
empirical marker of proximity to a critical regime in a feedforward-biased
excitable network, not as a physiological claim.

---

## 5. Repository layout

```
.
├── README.md       this document
├── connectome.py   DC-SBM sampler → Connectome(W, slices, hemi, sign)
├── lif.py          LIFNetwork: sparse matvec + exact-exp synapse + Euler membrane
├── env.py          ForestEnv: plumes, stalking predator, bilateral sensors, kinematic body
└── run.py          𝓔 / 𝓓 and the closed loop; prints per-region rates
```

Dependency graph: `run.py → {connectome, lif, env}`; the three leaves are mutually independent.

## 6. Usage

```bash
pip install numpy scipy
python connectome.py                      # sanity: N, |E|, <k_out>, inhibitory fraction
python run.py --n 4000 --seconds 20       # closed loop; ~8 s wall on a laptop-class CPU
python run.py --n 4000 --seconds 5 --seed 3   # different graph draw
```

Default build (N = 4000, seed 0): ≈ 1.3 × 10⁵ edges, ⟨k_out⟩ ≈ 33, ≈ 21 % inhibitory.
Expected output: an edge/degree summary, terminal `food` / `caught` counters, and
a per-region mean-rate dictionary showing a rate gradient rising along the
feedforward hierarchy (sensory ≈ 2–4 Hz → M1 ≈ 20 Hz → CPG ≈ 30 Hz).

**Scaling caveat.** Block densities p_ab are N-independent, so ⟨k_out⟩ ∝ N and the
ν_bg default is tuned only for N ≈ 4000. At N = 20000 (⟨k_out⟩ ≈ 164) the network
saturates (rates in the hundreds of Hz). Rescale p_ab ∝ 1/N (constant in-degree)
or retune ν_bg when changing N.

**Scaling.** Per step cost ≈ O(|E| + N); memory ≈ O(|E|). At biological
N ~ 10⁹ with ⟨k⟩ ~ 10⁴ this is ~10¹³ edges, so realistic full-scale runs
require distributed sparse ops on the order of exascale memory. Not attempted.

To inject a measured graph, construct a `Connectome` from your own CSR matrix,
region slices, hemisphere labels and signs, and pass its fields into `run.py`
unchanged.

---

## 7. Validation plan (proposed)

1. **Null-model comparison.** Reciprocity, clustering, in/out-degree
   distributions and 3-node motif Z-scores against a configuration-model
   baseline.
2. **In-silico lesion.** Ablate an arc (e.g. SC → M1) and measure the
   behavioral deficit (evasion latency, capture rate); the repo's arcs
   predict a *selective* loss of turn-away behavior.
3. **Rate-statistics matching.** Fit **γ** so that per-compartment firing-rate
   distributions match awake-cervid electrophysiology (where available).
4. **Perturbation-response.** Compare simulated optogenetic-style stimulation
   maps to recorded ones (ungulate data is sparse; rodent proxy admissible).

## 8. Known limitations

* Prior is hand-specified; densities are order-of-magnitude guesses.
* No plasticity (STDP / neuromodulated), so no learning.
* No proprioceptive or interoceptive channel; body is a kinematic point mass with no mechanics or muscle model.
* Behavior is **not** validated and, at these defaults, is not expected to be adaptive.
* Point-neuron LIF; no heterogeneity in τ_m, v_th.

## 9. Roadmap

- [ ] Replace the kinematic body with a MuJoCo quadruped (cf. NeuroMechFly / flybody-style embodiment).
- [ ] Add a learned readout / cell-type-level gain fitting (**γ**) against a reward signal.
- [ ] Add STDP with dopaminergic gating (three-factor rule).
- [ ] Sparse GPU backend (CSR SpMV via CuPy/JAX) to reach N ≥ 10⁶.
- [ ] Importer for real EM-derived graphs (CAVE / neuPrint-style APIs).

## References

*Cited from memory; verify details before formal citation.*

- White, J. G. et al. (1986). The structure of the nervous system of *C. elegans*. *Phil. Trans. R. Soc. B*.
- Januszewski, M. et al. (2018). High-precision automated reconstruction of neurons with flood-filling networks. *Nature Methods*.
- Lobato-Rios, V. et al. (2022). NeuroMechFly, a neuromechanical model of adult *Drosophila*. *Nature Methods*.
- Dorkenwald, S. et al. (2024). Neuronal wiring diagram of an adult brain (FlyWire). *Nature*.
- Shiu, P. K. et al. (2024). A *Drosophila* computational brain model reveals sensorimotor processing. *Nature*.
- Lappalainen, J. K. et al. (2024). Connectome-constrained networks predict neural activity across the fly visual system. *Nature*.
- MICrONS Consortium. Functional connectomics spanning multiple areas of mouse visual cortex.

## License

MIT (proposed).
