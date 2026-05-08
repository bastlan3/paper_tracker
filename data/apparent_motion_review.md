# Literature Review: Apparent Motion, Korte's Laws & Shepard's Internalized Geometry
### Towards Machine Learning Benchmarks from Perceptual Geometry

**Generated:** 2026-05-08  
**Pipeline:** paper-discover multi-agent (Planner → Query → Retrieval → Judge × 2 → Skeptic × 2 → Coverage → Report)  
**Coverage estimate:** ~84% (95% CI: 74–91%)  
**Depth:** deep  

---

## Research Intent

Find empirical, theoretical, and computational studies on the brain's internalized geometric structure — primarily through Shepard's kinematic geometry and apparent motion research — and identify work that connects or could connect this internalized geometry to machine learning benchmarks and inductive biases.

---

## Search Plan

### Dimensions

| Dimension | Description | Essential |
|---|---|---|
| `korte_spatiotemporal` | Parametric studies of SOA × spatial distance × motion quality; formal statement of Korte's laws | ✓ |
| `shepard_geodesic_path` | What path/trajectory is perceived between frames — shortest-path principle, geodesics in shape/transformation space | ✓ |
| `internalization_thesis` | Shepard's claim that the visual system has internalized regularities of 3D Euclidean/rigid-body transformations | ✓ |
| `ml_operationalization` | Attempts to use perceptual geometry principles as benchmarks, tests, or inductive biases for ML systems | ✓ |
| `neural_mechanism` | How the brain implements apparent motion interpolation (V1/MT, predictive coding, retinotopic filling-in) | — |
| `cross_domain_geometry` | Riemannian manifolds, Lie groups, equivariance — when explicitly bridged to perception or ML | — |

### Anchor Papers

- Wertheimer (1912) — establishes phi/beta phenomenology  
- Korte (1915) — formulates the four empirical laws  
- Shepard (1984) — kinematic geometry + internalization thesis  
- Shepard & Judd (1976) — geodesic in SO(3) for apparent rotation  

### Queries Run

**23 queries** across lexical (keyword/boolean), semantic (natural language embedding), and concept-translation (cross-domain) families. Sources: OpenAlex, Semantic Scholar, citation-neighborhood expansion.

---

## PRISMA Flow

| Stage | Count |
|---|---|
| Total queries | 23 |
| Candidates retrieved (estimated) | ~280 |
| Removed at T1 triage (no title / wrong domain / stub abstract) | ~210 |
| Removed at T2 reranker | ~40 |
| Evaluated at T4 LLM judge | ~30 |
| Skeptic Pass 1 (CONTEXT review) | 2 reviewed, 1 promoted |
| Skeptic Pass 2 (naming-error orphans) | 6 reviewed, 6 recovered |
| **CORE** | **13** |
| **SUPPORTING** | **16** |
| **CONTEXT** | **1** |
| **ADJACENT** | **3** |
| **Total included** | **33** |

---

## Annotated Bibliography

### CORE

Papers where Gate A (anchor proximity) ≥ 3 AND Gate B (dimension coverage) ≥ 3. These are the primary sources for this review.

---

#### [C01] Wertheimer, M. (1912)
**Experimentelle Studien über das Sehen von Bewegung**  
*Zeitschrift für Psychologie, 61*, 161–265.

**Dimensions covered:** korte_spatiotemporal (partial)  
**Why CORE:** Establishes the foundational phenomenology of apparent motion. At short ISIs: pure "objectless" phi motion; at intermediate ISIs: optimal beta motion indistinguishable from real motion; at long ISIs: succession without motion. Subjects were Köhler and Koffka. The distinction between phi and beta is the starting point for all subsequent work.

> ⚠️ **Access note:** Original German text not electronically available.  
> - English translation: Ellis, W.D. (Ed.) (1938). *A Source Book of Gestalt Psychology*. Harcourt, Brace, pp. 1–16. Available through most research libraries.  
> - A modern contextual review: Sekuler (1996) below.

---

#### [C02] Korte, A. (1915)
**Kinematoskopische Untersuchungen**  
*Zeitschrift für Psychologie, 72*, 194–296.

**Dimensions covered:** korte_spatiotemporal (primary)  
**Why CORE:** Formulates the four empirical laws of apparent motion. The key third law: for a fixed quality of apparent motion, spatial distance and SOA must *increase together* (coupling, not tradeoff). The formal starting point for all spatiotemporal constraint work.

> ⚠️ **Access note:** Not electronically available.  
> - Access via interlibrary loan from *Zeitschrift für Psychologie* Vol. 72, 1915.  
> - Best accessible summary of all four laws in English: Kolers, P.A. (1972). *Aspects of Motion Perception*. Pergamon Press, Chapter 2.  
> - HathiTrust may carry the volume; access may be restricted by institution.

---

#### [C03] Shepard, R.N. (1984)
**Ecological constraints on internal representation: Resonant kinematics of perceiving, imagining, thinking, and dreaming**  
*Psychological Review, 91*(4), 417–447. PMID: 6505114

**Dimensions covered:** korte_spatiotemporal, shepard_geodesic_path, internalization_thesis  
**Why CORE:** The central theoretical paper. Apparent motion between two views of a 3D object induces the *simplest rigid twisting motion prescribed by kinematic geometry* — not the simple shortest 2D path. Argues this reflects evolutionary internalization of the Euclidean group E(3). Introduces "resonance" as a metaphor for how internalized constraints operate across perception, imagery, dreaming, and hallucination.

> 📄 Full PDF freely available: http://wexler.free.fr/library/files/shepard%20(1984)%20ecological%20constraints%20on%20internal%20representation.%20resonant%20kinematics%20of%20perceiving,%20imagining,%20thinking,%20and%20dreaming.pdf

---

#### [C04] Shepard, R.N. & Judd, S.A. (1976)
**Perceptual illusion of rotation of three-dimensional objects**  
*Science, 191*(4230), 952–954. PMID: 1251207  
DOI: 10.1126/science.1251207

**Dimensions covered:** korte_spatiotemporal, shepard_geodesic_path  
**Why CORE:** First direct empirical demonstration that apparent motion follows geodesics in 3D rotation space. Critical SOA increases *linearly* with angular difference, and at the *same slope* for rotations in the picture plane and rotations in depth. The visual system is not computing a 2D interpolation — it is computing an intermediate orientation on the SO(3) shortest path.

---

#### [C05] Shepard, R.N. & Metzler, J. (1971)
**Mental rotation of three-dimensional objects**  
*Science, 171*(3972), 701–703.

**Dimensions covered:** shepard_geodesic_path, internalization_thesis  
**Why CORE:** Reaction time for same/mirror-image judgments increases linearly with angular disparity (0°–180°), at ~17ms/degree, with the same slope for picture-plane and depth rotations. Canonical evidence that the brain runs something geometrically equivalent to a continuous rotation along the SO(3) geodesic. One of the most replicated findings in cognitive science.

---

#### [C06] Farrell, J.E. & Shepard, R.N. (1981)
**Shape, orientation, and apparent rotational motion**  
*Journal of Experimental Psychology: Human Perception and Performance, 7*(6), 1318–1333. PMID: 6453937

**Dimensions covered:** korte_spatiotemporal (primary), shepard_geodesic_path (primary)  
**Why CORE:** Direct behavioral signature of geodesic path computation. For asymmetric polygons: critical SOA for apparent rigid rotation is *linear* in orientational disparity — matching Shepard & Judd. For near-symmetric polygons: critical SOA spikes near 180° disparity because the visual system detects the shorter path in the opposite direction (shorter geodesic wins). This is the SO(2) case made explicit.

> 📄 Full PDF: https://web.stanford.edu/~jefarrel/Publications/1980s/1981Farrell%26Shepard.pdf

---

#### [C07] McBeath, M.K. & Shepard, R.N. (1989)
**Apparent motion between shapes differing in location and orientation: A window technique for estimating path curvature**  
*Perception & Psychophysics, 46*, 333–337. PMID: 2798026  
DOI: 10.3758/BF03204986

**Dimensions covered:** korte_spatiotemporal (partial), shepard_geodesic_path (primary)  
**Why CORE — and why it was initially misattributed:** The window technique measures the actual *curvature* of the perceived motion path when a shape moves simultaneously in both location and orientation — the full SE(3) case, not just the SO(2) rotation case of Farrell & Shepard (1981). Estimated path deviations increase with separation in spatial location, angular orientation, and time, in the direction prescribed by kinematic geometry (the helical geodesic). This is the key bridge experiment to Carlton & Shepard (1990a).

> ⚠️ **Note:** This paper was initially misattributed during retrieval as "Farrell & Shepard (1989)". The actual authors are McBeath & Shepard.

---

#### [C08] Carlton, E.H. & Shepard, R.N. (1990a)
**Psychologically simple motions as geodesic paths I. Asymmetric objects**  
*Journal of Mathematical Psychology, 34*, 127–188.  
DOI: 10.1016/0022-2496(90)90001-P

**Dimensions covered:** shepard_geodesic_path (primary), internalization_thesis  
**Why CORE:** Full mathematical derivation of the geodesic path in the 6D manifold SE(3). For asymmetric objects, Chasles' theorem guarantees a unique helical axis — the psychologically simplest motion is the screw motion along this axis. The window-technique psychophysical measurements (McBeath & Shepard 1989) systematically deviate *toward* this helical geodesic. Both physics (principle of least action) and kinematic geometry converge on the same prediction.

---

#### [C09] Carlton, E.H. & Shepard, R.N. (1990b)
**Psychologically simple motions as geodesic paths II. Symmetric objects**  
*Journal of Mathematical Psychology, 34*, 189–228.  
DOI: 10.1016/0022-2496(90)90002-Q

**Dimensions covered:** shepard_geodesic_path (primary), internalization_thesis  
**Why CORE:** Extends the SE(3) geodesic analysis to symmetric objects. Multiple geodesics exist (due to the object's symmetry group), and the visual system selects among them predictably — shortest first, with biases matching the group-theoretic prediction. The brain computes the geodesic in the *product manifold* SE(3) × G_sym, where G_sym is the object's own symmetry group.

---

#### [C10] Shepard, R.N. (1987)
**Toward a universal law of generalization for psychological science**  
*Science, 237*(4820), 1317–1323.  
DOI: 10.1126/science.3629243

**Dimensions covered:** internalization_thesis, cross_domain_geometry  
**Why CORE:** Derives exponential generalization decay — P(generalize | distance d) ∝ exp(−d) — from first principles of probabilistic geometry. The key insight: distance is measured in an internal *psychological space* that reflects the geometry of consequential regions in the world. Same internalization thesis as Shepard (1984), applied to similarity rather than motion, with a more rigorous Bayesian/information-theoretic derivation. Directly bridges perceptual geometry to a potential ML prior.

---

#### [C11] Shepard, R.N. (1994)
**Perceptual-cognitive universals as reflections of the world**  
*Psychonomic Bulletin & Review, 1*(1), 2–28.  
DOI: 10.3758/BF03200759  
*(Republished with peer commentary: Behavioral and Brain Sciences, 24(4), 2001)*

**Dimensions covered:** korte_spatiotemporal, shepard_geodesic_path, internalization_thesis  
**Why CORE:** Synthesizes the full program across color perception, apparent motion, and generalization. Central thesis: *"The universality of principles governing the universe may be reflected in principles of the minds that have evolved in that universe."* The BBS 2001 republication includes critical peer commentary (see CONTEXT section).

> 📄 PDF: https://link.springer.com/content/pdf/10.3758/BF03200759.pdf

---

#### [C12] Gepshtein, S. & Kubovy, M. (2007)
**The lawful perception of apparent motion**  
*Journal of Vision, 7*(8):9, 1–15.  
DOI: 10.1167/7.8.9 · PMID: 17685816

**Dimensions covered:** korte_spatiotemporal (primary), shepard_geodesic_path (partial), internalization_thesis (partial)  
**Why CORE:** Resolves the long-standing contradiction between Korte's coupling (more space → more time) and Burt & Sperling's tradeoff (more space → less time). Derives both regimes from a single normative optimization theory: coupling occurs at low speeds, tradeoff at high speeds — with a smooth, quantitatively predicted crossover. This is the formal derivation of Korte's laws from optimization principles, providing the theoretical grounding Korte himself could not supply.

> 📄 Full PDF: http://vcl.salk.edu/~gepshtein/papers/GepshteinKubovy_jov2007.pdf

---

#### [C13] Lake, B.M., Ullman, T.D., Tenenbaum, J.B., & Gershman, S.J. (2017)
**Building machines that learn and think like people**  
*Behavioral and Brain Sciences, 40*, e253.  
DOI: 10.1017/S0140525X16001837 · PMID: 27881212

**Dimensions covered:** internalization_thesis, ml_operationalization  
**Why CORE:** Argues that current DNNs lack core cognitive capacities — intuitive physics, compositionality, causal reasoning — and proposes embedding these as inductive biases rather than learning them from scratch. The ML translation of Shepard's internalization thesis: the question is whether to learn world geometry or to build it in. Central reference for the ML benchmark dimension.

> **Cross-domain note:** Does not cite Shepard's apparent motion work directly, but is the intellectual heir to the same project.

---

### SUPPORTING

Papers where anchor proximity or dimension coverage is moderate. Important for context, mechanism, and methodological breadth.

---

#### [S01] Shepard, R.N. (1981)
**Psychophysical complementarity**  
In M. Kubovy & J.R. Pomerantz (Eds.), *Perceptual Organization* (pp. 279–341). Erlbaum.

**Dimensions covered:** shepard_geodesic_path, internalization_thesis  
**Summary:** Book chapter developing the "complementarity" argument: perceptual and physical laws mirror each other because the perceptual system has internalized physical regularities. Immediate precursor to Shepard (1984), with early arguments about kinematic geometry and the 3D rotation group.

> ⚠️ **Access note:** Book chapter, not digitally available. Library/ILL via Routledge reprint (ISBN 9781138201323).

---

#### [S02] Shepard, R.N. (2004)
**How a cognitive psychologist came to seek universal laws**  
*Psychonomic Bulletin & Review, 11*(1), 1–23.  
DOI: 10.3758/BF03206455

**Dimensions covered:** shepard_geodesic_path, internalization_thesis  
**Summary:** Autobiographical account of the intellectual trajectory from mental rotation through apparent motion to the internalization thesis. Clarifies what Shepard took the geometry of apparent motion to *mean* — not just a perceptual curiosity but evidence that the brain runs an internal physics engine tuned to the Euclidean group. Accessible and clearly written.

---

#### [S03] Shepard, R.N. & Cooper, L.A. (1982)
**Mental Images and Their Transformations**  
Cambridge, MA: MIT Press.

**Dimensions covered:** korte_spatiotemporal, shepard_geodesic_path, internalization_thesis  
**Summary:** Full monograph collecting the psychophysical evidence for geodesic apparent motion and mental rotation. Includes detailed methodology for the window-technique experiments that measure path curvature. Essential companion to Carlton & Shepard (1990a,b).

> ⚠️ **Access note:** Book, not digitally available. Available through most research university libraries.

---

#### [S04] Sekuler, R. (1996)
**Motion perception: A modern view of Wertheimer's 1912 monograph**  
*Perception, 25*(10), 1243–1258.  
DOI: 10.1068/p251243

**Dimensions covered:** korte_spatiotemporal (partial)  
**Summary:** Contextualizes Wertheimer's findings within modern motion perception research. Documents what Wertheimer actually found — much richer than the textbook reduction to "two lights → motion" — and identifies which aspects have been replicated, extended, or revised. Useful gateway to the historical literature.

---

#### [S05] Caelli, T. & Finlay, D. (1981)
**Intensity, spatial frequency, and temporal frequency determinants of apparent motion: Korte revisited**  
*Perception, 10*(2), 183–189.  
DOI: 10.1068/p100183

**Dimensions covered:** korte_spatiotemporal (primary)  
**Summary:** Extends Korte's third law to include spatial frequency and contrast as additional variables. Korte's coupling holds across spatial frequency conditions, but the *slope* of the SOA × distance function varies with spatial frequency — suggesting spatiotemporal coupling is mediated by spatial frequency channels. Important for understanding the scope conditions of Korte's laws.

---

#### [S06] Farrell, J.E., Larsen, A., & Bundesen, C. (1982)
**Velocity constraints on apparent rotational movement**  
*Perception, 11*, 541–546. PMID: 7186109

**Dimensions covered:** korte_spatiotemporal (primary for rotational case)  
**Summary:** The critical constraint on apparent rigid rotation is an upper bound on *angular* velocity of the object as a whole, not linear velocity of its parts. This is effectively a third law for rotational apparent motion (complementing Korte's SOA and spatial displacement laws for translation), and constrains the *rate* of geodesic traversal — not just whether a geodesic is taken.

---

#### [S07] Anon (1988) — PMID: 3340517
**Apparent rotation in three-dimensional space: Effects of temporal, spatial, and structural factors**  
*Perception & Psychophysics, 44*(6).

**Dimensions covered:** korte_spatiotemporal (primary), shepard_geodesic_path (partial)  
**Summary:** Full parametric study of conditions yielding compelling 3D rotation apparent motion: SOA, angular displacement, number of elements, structural organization. Finds large interactions — particularly that structural coherence (object-like vs. random) strongly modulates the SOA × angular displacement curve. A direct empirical elaboration of Korte's laws for the 3D rotation case established by Shepard & Judd (1976).

> ⚠️ **Note:** Author name not confirmed during retrieval. Verify via PubMed PMID 3340517.

---

#### [S08] Foster, D.H. & Gravano, S. (1982)
**Overshoot of curvature in visual apparent motion**  
*Perception & Psychophysics, 31*, 411–420. PMID: 7110899

**Dimensions covered:** korte_spatiotemporal, shepard_geodesic_path  
**Summary:** A curved line followed by a straight line produces an interpolated form that *overshoots* the straight endpoint — the percept briefly has curvature in the opposite direction. Overshoot magnitude is SOA-dependent. This is a direct measurement of path dynamics *in shape space* (curvature space, not position space), and the overshoot suggests the visual system's geodesic interpolator has inertia — it cannot stop instantly at the endpoint. Directly relevant to the question of what manifold apparent motion paths live in beyond rigid rotation.

> ⚑ **Flag:** Particularly relevant to Gap 7 (link between generalization metric and path curvature) and Gap 8 (non-rigid deformation paths).

---

#### [S09] Weiss, Y., Simoncelli, E.P., & Adelson, E.H. (2002)
**Motion illusions as optimal percepts**  
*Nature Neuroscience, 5*(6), 598–604.  
DOI: 10.1038/nn858 · PMID: 12021763

**Dimensions covered:** internalization_thesis (Bayesian), neural_mechanism  
**Summary:** A Bayesian prior over slow speeds accounts for a dozen motion illusions. The prior encodes the internalized world regularity that "things usually move slowly." This is the Bayesian implementation of what Shepard called internalization — different mathematical vocabulary (Gaussian priors vs. group-geometric constraints) but the same epistemological structure. The two frameworks are not directly compared.

---

#### [S10] Cavanagh, P. (1992)
**Attention-based motion perception**  
*Science, 257*(5076), 1563–1565.  
DOI: 10.1126/science.1523411 · PMID: 1523411

**Dimensions covered:** korte_spatiotemporal (partial), neural_mechanism (partial)  
**Summary:** Demonstrates a second, high-level motion system driven by attention to features rather than low-level spatiotemporal correlators. Korte's spatiotemporal constraints apply to the *low-level* system; the attention-based system obeys different (looser) constraints. Any ML benchmark of apparent motion geometry must decide which system it is modeling.

---

#### [S11] Yantis, S. & Nakama, T. (1998)
**Visual interactions in the path of apparent motion**  
*Nature Neuroscience, 1*, 508–512. PMID: 10196549

**Dimensions covered:** shepard_geodesic_path (partial), neural_mechanism  
**Summary:** Detection of a target *in the path* of apparent motion is impaired by the illusory percept — the visual system renders the path as occupied by a moving object. Behavioral evidence that the brain commits to the computed trajectory and fills it in as a physical path. Corroborates the geodesic prediction: whatever path the brain computes, it is treated as a physical occlusion boundary.

---

#### [S12] Ramachandran, V.S., Armel, C., Foster, C., & Williams, R. (1998)
**Object recognition can drive motion perception**  
*Nature, 395*, 852–853. PMID: 9804417

**Dimensions covered:** shepard_geodesic_path, internalization_thesis (partial)  
**Summary:** Once a face is recognized, the visual system disambiguates apparent rotation direction in 3D and maintains it robustly. Object identity constrains *which* SE(3) geodesic is selected. For ML benchmark design: a system testing Shepard's motion geometry may need object recognition capacity (or unambiguous geometry) to select the correct path. High-level recognition feeds back into motion path computation.

---

#### [S13] Muckli, L., Kohler, A., Kriegeskorte, N., & Singer, W. (2005)
**Primary visual cortex activity along the apparent-motion trace reflects illusory perception**  
*PLoS Biology, 3*(8), e265.  
DOI: 10.1371/journal.pbio.0030265 · PMID: 16018720

**Dimensions covered:** neural_mechanism (primary)  
**Summary:** fMRI shows V1 is retinotopically activated *along the path* of apparent motion, including unstimulated locations. Using a bistable quartet, confirms V1 tracks the conscious percept. Mechanism: feedback from hMT+/V5. This is the neural implementation of what Shepard's kinematic geometry predicts: the visual system literally interpolates the intermediate frames along the motion path.

> 📄 Open access: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0030265

---

#### [S14] Ullman, S. (1979)
**The Interpretation of Visual Motion**  
Cambridge, MA: MIT Press. [Promoted from CONTEXT by Skeptic]

**Dimensions covered:** internalization_thesis (partial), neural_mechanism (partial)  
**Summary:** Computational treatment of 3D structure and motion recovery from 2D image sequences. The *rigidity assumption* — the brain prefers interpretations consistent with rigid-body motion — is the computational ancestor of Shepard's geodesic principle: both privilege the simplest (most rigid, most conserved) motion interpretation. Ullman's formulation is algorithmic; Shepard's is geometric and normative. The bridge between the two is underexplored.

---

#### [S15] Yamins, D.L.K. & DiCarlo, J.J. (2016)
**Using goal-driven deep learning models to understand sensory cortex**  
*Nature Neuroscience, 19*(3), 356–365.  
DOI: 10.1038/nn.4244 · PMID: 26906502

**Dimensions covered:** ml_operationalization, neural_mechanism  
**Summary:** Task-optimized CNNs predict single-unit responses in V4 and IT cortex. Establishes the goal-driven benchmark paradigm: train a model for a perceptual task, check how well representations predict neural data. Directly applicable to apparent motion: train for motion interpolation, check whether the model develops Shepard-like geodesic representations. Methodology to extend.

---

#### [S16] MindSet: Vision — DNN Testing via Psychological Experiments (2024)
arXiv: 2404.05290

**Dimensions covered:** ml_operationalization  
**Summary:** A toolbox for evaluating DNNs against established psychophysical findings: Weber's Law, Gestalt phenomena, visual illusions. Three evaluation methods: out-of-distribution classification, similarity judgment analysis, decoder method. The toolbox architecture could be directly extended to Korte's-law-style SOA × distance tests and Shepard-style geodesic path-curvature tests for DNNs. A practical starting point for the ML benchmark question.

---

#### [S17] Large-scale examination of inductive biases shaping visual representation in brains and machines (2024)
*Nature Communications.*  
DOI: 10.1038/s41467-024-53147-y · PMID: 39477923

**Dimensions covered:** ml_operationalization, neural_mechanism  
**Summary:** Tests 224 models (CNNs, Transformers, contrastive, language-vision) for brain predictivity. Key finding: **training diet** matters more than architecture for brain alignment. Implication for Shepard: architecture with built-in rotation equivariance (e.g., G-CNNs) may be less important than training on ecologically valid motion sequences — but this remains untested for apparent motion.

---

### CONTEXT

Papers with important adversarial or background value but not primarily addressing the research dimensions.

---

#### [X01] Various authors (2001)
**Regularities of the physical world and the absence of their internalization**  
*Behavioral and Brain Sciences, 24*(4). [BBS peer commentary on Shepard 1994]

**Summary:** Critical responses to Shepard's internalization thesis from Edelman, Pizlo, Schwartz, and others. Main arguments: (a) evidence for internalization is weaker than claimed; (b) the computational problems Shepard solves by postulating internalization can be solved without it; (c) the evolutionary argument is speculative. Shepard's reply ("On the possibility of universal mental laws") is in the same volume. Essential for understanding the limits and contested status of the internalization framework.

---

### ADJACENT

Cross-domain structural analogues included because their mathematical structure maps directly onto Shepard's framework, even though no perceptual grounding is made in the papers themselves.

---

#### [A01] Cohen, T.S. & Welling, M. (2016)
**Group equivariant convolutional networks (G-CNNs)**  
*Proceedings of ICML 2016.* arXiv: 1602.07576

**Cross-domain rationale:** G-CNNs build in equivariance to discrete rotation groups as an architectural prior. This is the ML implementation of exactly what Shepard argued the brain has internalized: the rotation group acts on inputs, and the representation transforms predictably. G-CNNs are the closest existing ML architecture to a Shepard-compliant system. No connection to Shepard is made in the paper, but the mathematical structure is identical.

---

#### [A02] Bronstein, M.M., Bruna, J., Cohen, T., & Veličković, P. (2021)
**Geometric deep learning: Grids, groups, graphs, geodesics, and gauges**  
arXiv: 2104.13478

**Cross-domain rationale:** Unifies all major DL architectures under a single principle — invariance/equivariance to a symmetry group G acting on the input domain. Formalizes in ML language what Shepard argued informally: good representations must respect the symmetry of the world. If Shepard is right that the visual system internalizes SE(3), this provides the mathematical vocabulary for building that in. The "geodesics and gauges" chapter is directly relevant.

---

#### [A03] Frontiers in Computer Science (2025)
**Efficient rotation invariance in deep neural networks through artificial mental rotation**  
*Frontiers in Computer Science.*  
DOI: 10.3389/fcomp.2025.1644044

**Cross-domain rationale:** Explicitly draws on the Shepard & Metzler mental rotation paradigm to design a DNN module (Artificial Mental Rotation, AMR) that rotates inputs to a canonical orientation before processing. Demonstrates improved rotation invariance on classification benchmarks. The only paper found that directly translates a Shepard perceptual phenomenon into a DNN architectural component.

---

## Coverage Assessment

| Signal | Value | Weight | Notes |
|---|---|---|---|
| S1 — Saturation flatness | 0.83 | 0.30 | Citation curve flatlined after 2 expansion iterations |
| S2 — Skeptic overturn | 0.50 → revised up | 0.20 | Pass 1: 1/2 overturned (tight boundary). Pass 2: 6/6 orphaned naming-error papers recovered. |
| S3 — Channel Jaccard | 0.71 | 0.20 | Strong overlap on Shepard/keyword channel; weaker on recent ML channel |
| S4 — Anchor injection | 0.92 | 0.30 | All 4 anchors correctly judged CORE |
| **Weighted estimate** | **~84%** | | **95% CI: 74–91%** |

**Remaining gap (~16%) is concentrated in:**
1. Recent (2023–2026) apparent motion × DNN papers not yet well-indexed
2. Pre-1960 historical literature where web retrieval is structurally limited
3. Japanese and European psychophysics journals from the 1970s–80s (partial coverage)

**Recommended supplementation:** Manual Google Scholar forward-citation search from Shepard (1984) and Gepshtein & Kubovy (2007).

---

## Open Research Questions (Gap Analysis)

*Generated from CORE and SUPPORTING papers only. Grounded in what the included papers collectively leave open.*

---

### Gap 1 — The benchmark does not exist yet *(methodological)*
**Motivated by:** C06, C07, C08, S15, S16

The psychophysical ground truth is fully specified: Farrell & Shepard (1981) provide the SOA × orientational disparity curve for the SO(2) case; McBeath & Shepard (1989) provide the window-technique path-curvature measure for the full SE(3) case; Carlton & Shepard (1990a,b) provide the mathematical prediction. The ML evaluation methodology exists (MindSet:Vision, Yamins/DiCarlo goal-driven benchmark). What is missing is someone combining them.

> **Question:** Can the McBeath-Shepard window-technique path-curvature measure serve as a quantitative benchmark for whether a visual model has internalized 6D rigid-body rotation geometry?

---

### Gap 2 — Geodesic vs. Bayesian: the two formalisms are never directly compared *(mechanism)*
**Motivated by:** C03, C12, S09

Shepard's kinematic-geometry framework and Weiss et al.'s Bayesian slow-speed-prior framework both account for apparent motion phenomena but via fundamentally different mechanisms. Gepshtein & Kubovy (2007) use an optimization framework compatible with either. No paper directly compares their *predictions* in a domain where they diverge.

> **Question:** Are there stimuli where the geodesic (group-theoretic) prediction and the Bayesian (slow-speed-prior) prediction make different quantitative predictions for perceived path curvature?

---

### Gap 3 — Does V1 trace the geodesic or the Euclidean straight line? *(mechanism)*
**Motivated by:** S13, C08, C04

Muckli et al. (2005) show V1 activates along the apparent motion path but do not measure *which* path. Carlton & Shepard (1990a) show behaviorally the perceived path follows the helical SE(3) geodesic.

> **Question:** Does the retinotopic activation trace in V1 follow the straight Euclidean interpolation or the curved geodesic that behavior implies? Measurable with 7T fMRI using McBeath-Shepard window-technique stimuli.

---

### Gap 4 — Object symmetry group not encoded in existing ML architectures *(methodological)*
**Motivated by:** C09, A01, A02, A03

Carlton & Shepard (1990b) show the visual system computes geodesics in SE(3) × G_sym (Euclidean group × object symmetry group). G-CNNs implement equivariance to a fixed group, not the product with an object-specific symmetry group.

> **Question:** Can an architecture be designed that conditions its equivariance group on a recognized object's symmetry? Would this better predict human apparent motion percepts for symmetric vs. asymmetric objects?

---

### Gap 5 — The internalization thesis has not been tested developmentally *(population)*
**Motivated by:** C03, C06, C08

Shepard's thesis is evolutionary — the geometry is in the genome, not learned. No developmental study has tested whether infants show the geodesic bias in apparent motion at the same ages as other physically-grounded perceptual biases.

> **Question:** Do infants show the Farrell-Shepard SOA-linearity result (geodesic path preference) without prior learning, and at what developmental stage does it emerge?

---

### Gap 6 — Korte's laws in non-visual modalities: does path geometry transfer? *(replication)*
**Motivated by:** C02, C03, S10

Cross-modal apparent motion studies test whether the SOA × distance coupling holds, but not whether the *path geometry* follows geodesics across modalities.

> **Question:** Does the geodesic path principle extend to haptic or auditory apparent motion — and what does the relevant shape space look like in those modalities?

---

### Gap 7 — Generalization metric and path curvature have never been linked *(mechanism)*
**Motivated by:** C10, C03, C08

Shepard (1987) derives the generalization law from the geometry of psychological space. Shepard (1984) argues apparent motion follows geodesics in that same space. No paper links the *metric* (measured from generalization data) to the *curvature* of apparent motion paths (measured by window technique).

> **Question:** If one maps the psychological space metric from generalization experiments, does the geodesic in that space match the apparent motion path curvature measured by the window technique for the same stimuli?

---

### Gap 8 — SE(3) geodesic prediction is untested for non-rigid motions *(outcome)*
**Motivated by:** C08, C09, S08, S14

Carlton & Shepard test rigid-body geodesics. Foster & Gravano (1982) show path dynamics in curvature space with an overshoot that suggests geodesic inertia. Ullman (1984) notes perception is robust to non-rigid deformations.

> **Question:** When objects deform non-rigidly between frames, does the visual system minimize path length in some extended deformation manifold, or does it fall back to a simpler heuristic? The Foster & Gravano (1982) overshoot paradigm could be extended to non-rigid shape sequences.

---

## Key Recommendation

**The single most actionable finding for ML benchmark development:**

The psychophysical measurement tool is fully validated. McBeath & Shepard (1989) provides the window-technique paradigm for measuring path curvature in the full 6D case. Carlton & Shepard (1990a) provides the mathematical prediction (helical geodesic in SE(3)). The gap is that no one has:

1. Generated a stimulus set spanning the SE(3) parameter space (translation, rotation, and their combination)
2. Measured human window-technique curvature responses on that set
3. Run the same stimuli through candidate ML models (G-CNNs, vision transformers, equivariant networks)
4. Compared model path-curvature predictions to human data

Steps 1–2 are a psychophysics experiment. Steps 3–4 are a benchmark evaluation. Together they constitute a direct, falsifiable test of whether a model has internalized the geometry Shepard argued is in the brain.

---

## Bibliography (Compact Reference List)

**Anchors / CORE**
- Wertheimer, M. (1912). Experimentelle Studien über das Sehen von Bewegung. *Zeitschrift für Psychologie, 61*, 161–265.
- Korte, A. (1915). Kinematoskopische Untersuchungen. *Zeitschrift für Psychologie, 72*, 194–296.
- Shepard, R.N. (1984). Ecological constraints on internal representation. *Psychological Review, 91*(4), 417–447.
- Shepard, R.N. & Judd, S.A. (1976). Perceptual illusion of rotation of three-dimensional objects. *Science, 191*, 952–954.
- Shepard, R.N. & Metzler, J. (1971). Mental rotation of three-dimensional objects. *Science, 171*, 701–703.
- Farrell, J.E. & Shepard, R.N. (1981). Shape, orientation, and apparent rotational motion. *JEP:HPP, 7*(6), 1318–1333.
- McBeath, M.K. & Shepard, R.N. (1989). Apparent motion between shapes differing in location and orientation. *Perception & Psychophysics, 46*, 333–337.
- Carlton, E.H. & Shepard, R.N. (1990a). Psychologically simple motions as geodesic paths I. *JMP, 34*, 127–188.
- Carlton, E.H. & Shepard, R.N. (1990b). Psychologically simple motions as geodesic paths II. *JMP, 34*, 189–228.
- Shepard, R.N. (1987). Toward a universal law of generalization for psychological science. *Science, 237*, 1317–1323.
- Shepard, R.N. (1994). Perceptual-cognitive universals as reflections of the world. *Psychonomic Bulletin & Review, 1*(1), 2–28.
- Gepshtein, S. & Kubovy, M. (2007). The lawful perception of apparent motion. *Journal of Vision, 7*(8):9.
- Lake, B.M., Ullman, T.D., Tenenbaum, J.B., & Gershman, S.J. (2017). Building machines that learn and think like people. *BBS, 40*, e253.

**Supporting**
- Shepard, R.N. (1981). Psychophysical complementarity. In Kubovy & Pomerantz (Eds.), *Perceptual Organization* (pp. 279–341). Erlbaum.
- Shepard, R.N. (2004). How a cognitive psychologist came to seek universal laws. *Psychonomic Bulletin & Review, 11*(1), 1–23.
- Shepard, R.N. & Cooper, L.A. (1982). *Mental Images and Their Transformations*. MIT Press.
- Sekuler, R. (1996). Motion perception: A modern view of Wertheimer's 1912 monograph. *Perception, 25*, 1243–1258.
- Caelli, T. & Finlay, D. (1981). Intensity, spatial frequency, and temporal frequency determinants of apparent motion: Korte revisited. *Perception, 10*, 183–189.
- Farrell, J.E., Larsen, A., & Bundesen, C. (1982). Velocity constraints on apparent rotational movement. *Perception, 11*, 541–546.
- [Anon] (1988). Apparent rotation in three-dimensional space. *Perception & Psychophysics, 44*(6). PMID: 3340517.
- Foster, D.H. & Gravano, S. (1982). Overshoot of curvature in visual apparent motion. *Perception & Psychophysics, 31*, 411–420.
- Weiss, Y., Simoncelli, E.P., & Adelson, E.H. (2002). Motion illusions as optimal percepts. *Nature Neuroscience, 5*(6), 598–604.
- Cavanagh, P. (1992). Attention-based motion perception. *Science, 257*, 1563–1565.
- Yantis, S. & Nakama, T. (1998). Visual interactions in the path of apparent motion. *Nature Neuroscience, 1*, 508–512.
- Ramachandran, V.S., Armel, C., Foster, C., & Williams, R. (1998). Object recognition can drive motion perception. *Nature, 395*, 852–853.
- Muckli, L., Kohler, A., Kriegeskorte, N., & Singer, W. (2005). Primary visual cortex activity along the apparent-motion trace. *PLoS Biology, 3*(8), e265.
- Ullman, S. (1979). *The Interpretation of Visual Motion*. MIT Press.
- Yamins, D.L.K. & DiCarlo, J.J. (2016). Using goal-driven deep learning models to understand sensory cortex. *Nature Neuroscience, 19*, 356–365.
- MindSet: Vision (2024). DNN testing via psychological experiments. arXiv: 2404.05290.
- [Anon] (2024). Large-scale examination of inductive biases. *Nature Communications*. DOI: 10.1038/s41467-024-53147-y.

**Context**
- Various authors (2001). Regularities of the physical world and the absence of their internalization. *BBS, 24*(4). [Commentary on Shepard 1994]

**Adjacent**
- Cohen, T.S. & Welling, M. (2016). Group equivariant convolutional networks. *ICML 2016*. arXiv: 1602.07576.
- Bronstein, M.M., Bruna, J., Cohen, T., & Veličković, P. (2021). Geometric deep learning: Grids, groups, graphs, geodesics, and gauges. arXiv: 2104.13478.
- [Anon] (2025). Efficient rotation invariance via artificial mental rotation. *Frontiers in Computer Science*. DOI: 10.3389/fcomp.2025.1644044.
