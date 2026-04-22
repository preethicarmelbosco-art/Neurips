# Datasheet for CogBench v1.1.0

Following the "Datasheets for Datasets" template of Gebru et al. (2018,
*Commun. ACM*, 2021). All author-identifying information has been redacted
for double-blind review.

---

## Motivation

### For what purpose was the dataset created?

CogBench was created to support **mechanistic interpretability** and
**contrastive fine-tuning** research on language models. Existing capability
benchmarks conflate raw task accuracy with the strength of the scaffolding
(context, chain-of-thought, instruction templates) that surrounds the task.
CogBench introduces matched **target / retain** pairs that let researchers
quantify how much of a model's competence is internalized vs. scaffolding-
dependent — operationalized as **Cognitive Absorption (CA)**.

### Who funded the creation of the dataset?

Redacted for double-blind review. A non-anonymous statement will be added
at camera-ready.

### Any other comments?

The suite covers nine cognitive primitives (theory of mind, causal-temporal
reasoning, applied scientific reasoning, moral reasoning, strategic
reasoning, stepwise planning, mathematics) plus a style-only negative
control (NULL-CC) and a five-domain COIN (Contrast) family whose target/retain pairs are complete opposites — like sides of a coin.

---

## Composition

### What do the instances represent?

Each instance is a **contrastive pair of short English passages**: one
`target` that exercises the corpus-specific cognitive skill, and one
`retain` that is content-matched but lacks that skill. Passages are machine-
generated natural language (plus math symbols for SPL-CC and CoreMath).

### How many instances are there in total?

145 390 pairs across 39 JSONL files:

| Split   | Pairs     |
| ------- | --------: |
| train   |    55 094 |
| bench   |    60 079 |
| holdout |     5 659 |
| coin    |    24 556 |
| **total** | **145 390** |

Per-file record counts are in `SHA256SUMS` / `croissant.json`.

### Does the dataset contain all possible instances or is it a sample?

Sample. Each corpus is drawn from a hand-curated set of scenario / topic
templates, expanded via the generator. Templates reflect designer choices
and do not exhaustively cover the cognitive primitive.

### What data does each instance consist of?

Plain UTF-8 text plus light metadata (see README § "Record schema" and
`croissant.json` `recordSet` entries for authoritative field lists).

### Is there a label or target associated with each instance?

There is no supervised *task* label; labels are the **corpus membership**
(target vs. retain within a pair) and the **taxonomic metadata** (category,
difficulty, domain). The pair structure is the annotation.

### Is any information missing from individual instances?

No. Every record has all schema-required fields populated.

### Are relationships between individual instances made explicit?

Yes. Each pair is joined by `id` (UUID) and by `scenario_id` / `topic_id` /
`seed_topic`, which link target and retain passages derived from the same
seed.

### Are there recommended data splits?

Yes. Three pre-computed splits per corpus:

- `train/` — used for contrastive fine-tuning.
- `bench/` — used for absorption measurement (disjoint seeds from train).
- `holdout/` — reserved for late-stage validation only.

Splits share **no scenario seeds**. Users performing additional cross-
validation should preserve this seed-level disjointness.

### Are there any errors, sources of noise, or redundancies?

Known sources of noise:

- **Generator bias**: single-model family, single decoding configuration.
- **Judge / generator correlation**: LLM judges share pre-training
  substrates with the generator, creating correlated false negatives.
- **Template coverage gaps**: scenarios outside the hand-curated templates
  are under-represented.
- **NULL-CC formality contrast** is systematic, not sampled from naturalistic
  distributions.

### Is the dataset self-contained, or does it link to external resources?

Self-contained. No external resources are required at load time.

### Does the dataset contain data that might be considered confidential?

No. All content is synthetically generated; scenarios reference fictional
placeholders (e.g. "Senior Analyst A", "Department Head B").

### Does the dataset contain data that might be offensive, insulting, threatening, or cause anxiety?

Some corpora (MOR-CC, CTR-CC, STR-CC) include scenarios involving deception,
harm, or morally-charged situations. Content is generated under a
moderation-filtered prompt and was not observed to contain explicit or
graphic material, but mild distress-adjacent content (workplace conflict,
safety incidents, legal disputes) is present by design.

### Does the dataset relate to people?

Indirectly. Scenarios describe fictional agents by role (Analyst A, Patient
X, Student 1). No real people are depicted.

### Does the dataset identify any sub-populations?

No. No demographic attributes (gender, age, ethnicity, geography, socio-
economic status) are solicited or embedded. See Croissant RAI field
`personalSensitiveInformation`.

### Is it possible to identify individuals?

No. See above.

### Does the dataset contain sensitive data?

No PII. No health, financial, or biometric records. No political or
religious affiliations are encoded.

---

## Collection

### How was the data associated with each instance acquired?

Machine-generated via a single open-weights teacher LLM served through a
local inference stack, with structured-output JSON enforced by a schema-
validating wrapper. Each pair underwent:

1. **Deterministic regex / structural gates** per corpus (math-symbol
   checks for SPL-CC and CoreMath; cognitive-term blacklist for NULL-CC;
   counterfactual pattern matching for CTR-CC; etc.).
2. **Multi-judge LLM panel** applying unanimous (context-dependent corpora:
   ToM, CTR, MOR, STR, STP, COIN) or majority (NULL) consensus. Pairs
   failing either stage were rejected and regenerated.

### Over what timeframe was the data collected?

**2026-03-19 → 2026-04-14.** Later corpora (MOR-CC, NULL-CC re-validation,
COIN) continued into April following generator-pipeline iteration.

### What mechanisms or procedures were used to collect the data?

A Python pipeline orchestrating the generator LLM, structural validators,
and judge panel. All prompts, regex gates, and judge configurations are
released with the code bundle at camera-ready.

### If the dataset is a sample from a larger set, what was the sampling strategy?

Template-driven. Each corpus has a fixed list of seed templates; the
generator samples over templates then over stylistic variations within each.
Near-duplicates (by embedding distance) are filtered within each corpus.

### Who was involved in the data collection process and how were they compensated?

No human annotators were used. Pipeline design and prompt engineering were
performed by the authors (redacted for review).

### Were any ethical review processes conducted?

Not required: no human subjects data, no PII, no scraped material. Content
moderation is enforced by the generator's safety filter and by manual spot
inspection.

### Does the dataset relate to people?

No.

---

## Preprocessing / Cleaning / Labeling

### Was any preprocessing / cleaning / labeling of the data done?

Yes:

- Deterministic regex gates per corpus (see Collection §2.1).
- Embedding-based near-duplicate filtering.
- **Cross-benchmark decontamination scan** against public holdouts
  (BigBench, BigToM, BCOPA-CE, Hendrycks MATH, FinQA, LegalBench,
  ScienceQA, CyberMetric). Pairs with n-gram overlap above threshold to any
  public test item were rejected.
- Split assignment by scenario-seed id so that train / bench / holdout
  share no seed.

### Was the "raw" data saved in addition to the cleaned data?

The pipeline preserved intermediate `accepted.jsonl` / `rejected.jsonl`
files with judge verdicts. These are *not* included in this release but
are available on request (camera-ready).

### Is the software that was used to preprocess / clean / label the data available?

Yes — released alongside the dataset at camera-ready. The
`code/generate_croissant.py` included here regenerates the metadata only;
the full generation + validation pipeline is a separate release artifact.

---

## Uses

### Has the dataset been used for any tasks already?

Yes — used in the accompanying submission to compute Cognitive Absorption
scores across a 42-model zoo.

### Is there a repository that links to any or all papers or systems that use the dataset?

Will be maintained at the camera-ready hosting URL.

### What (other) tasks could the dataset be used for?

- Contrastive fine-tuning ablations.
- Probing / steering-vector experiments.
- Feature-attribution method evaluation (where ground-truth counterfactuals
  are required).
- Curriculum design for cognitive primitives.

### Is there anything about the composition of the dataset or the way it was collected that might impact future uses?

The single-generator bias means downstream work should **not** claim
capability measurement on CogBench alone; pair with human-authored
capability benchmarks. The diagnostic contract assumes paired use of
`target` and `retain` splits — using `target` as a stand-alone training
corpus is out of scope.

### Are there tasks for which the dataset should not be used?

- Ranking model "intelligence" on a public leaderboard (it is diagnostic,
  not a capability leaderboard).
- Evaluating frontier research mathematics (CoreMath is textbook /
  competition-grade).
- High-stakes deployment decisions without human review.

---

## Distribution

### Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?

Yes — public release under CC BY 4.0.

### How will the dataset be distributed?

Anonymous review copy via the submission platform. Camera-ready release
will be mirrored to a DOI-issuing archive (Hugging Face Hub, Harvard
Dataverse, or Zenodo) preserving the file layout and checksums.

### When will the dataset be distributed?

Anonymous review bundle: with the submission. Public release: at
camera-ready.

### Will the dataset be distributed under a copyright or other IP license, and/or under applicable terms of use?

Dual-licensed: **data** under **CC BY 4.0** (`LICENSE-DATA`); **code** under **Apache 2.0** (`LICENSE`).

### Have any third parties imposed IP-based or other restrictions?

No.

### Do any export controls or other regulatory restrictions apply?

No.

---

## Maintenance

### Who will be supporting / hosting / maintaining the dataset?

Redacted for review. Camera-ready will list a maintainer contact and
long-term hosting DOI.

### How can the owner / curator / manager of the dataset be contacted?

Via the submission platform during review; a public contact will be
provided at camera-ready.

### Is there an erratum?

No known errata at v1.1.0 release.

### Will the dataset be updated?

Bug-fix patches will follow semantic versioning. Breaking schema changes
bump MAJOR; superseded versions remain accessible via DOI.

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?

N/A — no personal data.

### Will older versions of the dataset continue to be supported / hosted / maintained?

Yes — via archival DOI at camera-ready.

### If others want to extend / augment / build on / contribute to the dataset, is there a mechanism for them to do so?

Yes — the release will include the generation pipeline and
`generate_croissant.py` so third parties can extend corpora with consistent
schema. Contribution process will be documented with the camera-ready
release.