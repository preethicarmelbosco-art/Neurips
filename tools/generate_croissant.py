#!/usr/bin/env python3
"""Generate Croissant 1.1 + RAI metadata for the CogBench anonymous submission.

Re-runnable: deterministically walks `data/` to emit per-file FileObjects,
per-file RecordSets, checksums, sizes, and record counts. Run from the
repository root (the directory that contains `data/`) or pass `--root`.

Spec refs:
  - https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html
  - https://docs.mlcommons.org/croissant/docs/croissant-rai-spec.html
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

CORPORA = {
    "spl_cc": {
        "title": "Symbolic Physics Logic (SPL-CC)",
        "skill": "applied scientific reasoning rendered as symbolic / mathematical derivations",
        "target_field": ("target", "Mathematical derivation with equations and units"),
        "retain_field": ("retain", "Conceptual/historical prose of the same physics topic, zero math"),
        "extra_fields": [
            ("seed_topic", "sc:Text", "Seed physics topic drawn from the curated template set"),
        ],
    },
    "tom_cc": {
        "title": "Theory of Mind (ToM-CC)",
        "skill": "higher-order belief, deception, and knowledge-state reasoning",
        "target_field": ("target", "Target passage containing ToM-bearing content (beliefs, deceptions, knowledge states)"),
        "retain_field": ("retain", "Content-matched control passage with ToM attributions removed"),
        "extra_fields": [
            ("scenario_id", "sc:Text", "Scenario seed identifier (shared by target and retain)"),
            ("category", "sc:Text", "ToM sub-category (e.g. deception, false_belief, epistemic_modal)"),
            ("difficulty", "sc:Text", "Difficulty tier: simple / moderate / complex / adversarial"),
            ("character_a", "sc:Text", "Primary character whose mental state is tracked"),
            ("belief_object", "sc:Text", "Proposition or object held in the character's belief"),
        ],
    },
    "ctr_cc": {
        "title": "Causal-Temporal Reasoning (CTR-CC)",
        "skill": "counterfactual causal inference beyond mere temporal succession",
        "target_field": ("target_text", "Counterfactual causal reasoning passage"),
        "retain_field": ("retain_text", "Temporal-sequence-only control (same events, no causal attribution)"),
        "extra_fields": [
            ("scenario_id", "sc:Text", "Scenario seed identifier"),
            ("category", "sc:Text", "Causal reasoning sub-category (e.g. preventive_causation, sufficient_cause)"),
            ("domain", "sc:Text", "Applied domain (medicine, law, engineering, ...)"),
            ("difficulty", "sc:Text", "Difficulty tier"),
        ],
    },
    "mor_cc": {
        "title": "Moral Reasoning (MOR-CC)",
        "skill": "moral norm application, permissibility, and blame attribution",
        "target_field": ("target", "Morally-loaded passage requiring normative reasoning"),
        "retain_field": ("retain", "Content-matched neutral descriptive passage"),
        "extra_fields": [
            ("scenario_id", "sc:Text", "Scenario seed identifier"),
            ("category", "sc:Text", "Moral reasoning sub-category"),
            ("difficulty", "sc:Text", "Difficulty tier"),
        ],
    },
    "str_cc": {
        "title": "Strategic Reasoning (STR-CC)",
        "skill": "goal-directed strategic planning, coordination, and adversarial play",
        "target_field": ("target", "Strategic reasoning passage"),
        "retain_field": ("retain", "Content-matched non-strategic control"),
        "extra_fields": [
            ("scenario_id", "sc:Text", "Scenario seed identifier"),
            ("category", "sc:Text", "Strategic reasoning sub-category"),
            ("difficulty", "sc:Text", "Difficulty tier"),
        ],
    },
    "stp_cc": {
        "title": "Stepwise Planning (STP-CC)",
        "skill": "multi-step plan construction and decomposition",
        "target_field": ("target", "Explicit stepwise plan"),
        "retain_field": ("retain", "Unordered/flat recounting of the same steps"),
        "extra_fields": [
            ("scenario_id", "sc:Text", "Scenario seed identifier"),
            ("category", "sc:Text", "Planning sub-category"),
            ("difficulty", "sc:Text", "Difficulty tier"),
        ],
    },
    "null_cc": {
        "title": "Null Contrastive Corpus (NULL-CC, negative control)",
        "skill": "style-only contrast (formal vs. informal) with no cognitive content",
        "target_field": ("target_formal", "Formal-register description of the scenario (control target)"),
        "retain_field": ("retain_informal", "Informal-register description of the same scenario (control retain)"),
        "extra_fields": [
            ("scenario_id", "sc:Text", "Scenario seed identifier"),
            ("category", "sc:Text", "Scenario category (e.g. workplace_interaction)"),
            ("complexity", "sc:Text", "Complexity tier: simple / moderate / complex"),
        ],
    },
    "core_math": {
        "title": "Core Mathematics (CoreMath)",
        "skill": "formal mathematical proof construction",
        "target_field": ("target_proof", "Formal proof with explicit structure"),
        "retain_field": ("retain_intuition", "Intuitive / hand-wavy explanation of the same theorem"),
        "extra_fields": [
            ("topic_id", "sc:Text", "Math topic identifier"),
            ("category", "sc:Text", "Mathematics sub-category (algebra, analysis, ...)"),
            ("difficulty", "sc:Text", "Difficulty tier"),
        ],
    },
}

COIN_DOMAINS = {
    "CAU_COIN": "Causal contrast partner for CTR-CC (complete-opposition pairs)",
    "MOR_COIN": "Moral contrast partner for MOR-CC (complete-opposition pairs)",
    "STP_COIN": "Spatial-temporal contrast partner for STP-CC (complete-opposition pairs)",
    "STR_COIN": "Strategic contrast partner for STR-CC (complete-opposition pairs)",
    "TOM_COIN": "Theory-of-mind contrast partner for ToM-CC (complete-opposition pairs)",
}

SPLITS = ("train", "bench", "holdout")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def file_object(fid: str, name: str, description: str, path: Path, rel_url: str) -> dict:
    return {
        "@type": "cr:FileObject",
        "@id": fid,
        "name": name,
        "description": description,
        "contentUrl": rel_url,
        "encodingFormat": "application/jsonlines",
        "contentSize": f"{path.stat().st_size} B",
        "sha256": sha256_of(path),
        "md5": md5_of(path),
    }


def field(name: str, description: str, dtype: str, file_id: str, column: str) -> dict:
    return {
        "@type": "cr:Field",
        "@id": f"{file_id}/{name}",
        "name": name,
        "description": description,
        "dataType": dtype,
        "source": {
            "fileObject": {"@id": file_id},
            "extract": {"column": column},
        },
    }


def corpus_fields(file_id: str, spec: dict) -> list[dict]:
    out = [field("id", "UUID v4 unique identifier for the record", "sc:Text", file_id, "id")]
    for ename, edtype, edesc in spec["extra_fields"]:
        out.append(field(ename, edesc, edtype, file_id, ename))
    tname, tdesc = spec["target_field"]
    rname, rdesc = spec["retain_field"]
    out.append(field(tname, tdesc, "sc:Text", file_id, tname))
    out.append(field(rname, rdesc, "sc:Text", file_id, rname))
    out.append(field("timestamp", "ISO-8601 generation timestamp (UTC)", "sc:DateTime", file_id, "timestamp"))
    return out


def coin_fields(file_id: str) -> list[dict]:
    return [
        field("id", "UUID v4 unique identifier for the record", "sc:Text", file_id, "id"),
        field("scenario_id", "Scenario seed identifier", "sc:Text", file_id, "scenario_id"),
        field("theme", "Theme matched to the paired primary corpus", "sc:Text", file_id, "theme"),
        field("category", "Contrast sub-category", "sc:Text", file_id, "category"),
        field("difficulty", "Difficulty tier", "sc:Text", file_id, "difficulty"),
        field("target", "Target passage asserting the cognitive claim", "sc:Text", file_id, "target"),
        field("retain", "Retain passage asserting the complete opposite of target", "sc:Text", file_id, "retain"),
        field("timestamp", "ISO-8601 generation timestamp (UTC)", "sc:DateTime", file_id, "timestamp"),
    ]


def build(root: Path) -> dict:
    file_objects: list[dict] = []
    record_sets: list[dict] = []

    for corpus, spec in CORPORA.items():
        for split in SPLITS:
            path = root / "data" / split / f"{corpus}_{split}.jsonl"
            if not path.exists():
                continue
            fid = f"{corpus}_{split}_file"
            rel_url = f"data/{split}/{corpus}_{split}.jsonl"
            file_objects.append(
                file_object(
                    fid,
                    f"{corpus}_{split}.jsonl",
                    f"{spec['title']} — {split} split ({count_lines(path)} records)",
                    path,
                    rel_url,
                )
            )
            record_sets.append(
                {
                    "@type": "cr:RecordSet",
                    "@id": f"{corpus}_{split}_records",
                    "name": f"{corpus}_{split}_records",
                    "description": f"{spec['title']} — {split} split. Target exercises {spec['skill']}; retain is a content-matched control lacking that skill.",
                    "key": {"@id": f"{fid}/id"},
                    "field": corpus_fields(fid, spec),
                }
            )

    for domain, description in COIN_DOMAINS.items():
        for split in SPLITS:
            path = root / "data" / "coin" / domain / f"{split}.jsonl"
            if not path.exists():
                continue
            fid = f"coin_{domain.lower()}_{split}_file"
            rel_url = f"data/coin/{domain}/{split}.jsonl"
            file_objects.append(
                file_object(
                    fid,
                    f"coin/{domain}/{split}.jsonl",
                    f"COIN {domain} — {split} split ({count_lines(path)} records). {description}.",
                    path,
                    rel_url,
                )
            )
            record_sets.append(
                {
                    "@type": "cr:RecordSet",
                    "@id": f"coin_{domain.lower()}_{split}_records",
                    "name": f"coin_{domain.lower()}_{split}_records",
                    "description": f"COIN {domain} — {split} split. {description}.",
                    "key": {"@id": f"{fid}/id"},
                    "field": coin_fields(fid),
                }
            )

    ctx = {
        "@language": "en",
        "@vocab": "https://schema.org/",
        "citeAs": "cr:citeAs",
        "column": "cr:column",
        "conformsTo": "dct:conformsTo",
        "cr": "http://mlcommons.org/croissant/",
        "rai": "http://mlcommons.org/croissant/RAI/",
        "data": {"@id": "cr:data", "@type": "@json"},
        "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
        "dct": "http://purl.org/dc/terms/",
        "examples": {"@id": "cr:examples", "@type": "@json"},
        "extract": "cr:extract",
        "field": "cr:field",
        "fileProperty": "cr:fileProperty",
        "fileObject": "cr:fileObject",
        "fileSet": "cr:fileSet",
        "format": "cr:format",
        "includes": "cr:includes",
        "isLiveDataset": "cr:isLiveDataset",
        "jsonPath": "cr:jsonPath",
        "key": "cr:key",
        "md5": "cr:md5",
        "parentField": "cr:parentField",
        "path": "cr:path",
        "recordSet": "cr:recordSet",
        "references": "cr:references",
        "regex": "cr:regex",
        "repeated": "cr:repeated",
        "replace": "cr:replace",
        "samplingRate": "cr:samplingRate",
        "sc": "https://schema.org/",
        "separator": "cr:separator",
        "source": "cr:source",
        "subField": "cr:subField",
        "transform": "cr:transform",
    }

    doc = {
        "@context": ctx,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "name": "CogBench",
        "description": (
            "CogBench: a contrastive corpus suite for cognitive-circuit analysis in language models. "
            "Eight primary synthetic contrastive corpora spanning applied scientific reasoning (SPL-CC), "
            "theory of mind (ToM-CC), causal-temporal reasoning (CTR-CC), moral reasoning (MOR-CC), "
            "strategic reasoning (STR-CC), stepwise planning (STP-CC), core mathematics (CoreMath), "
            "and a negative-control null corpus (NULL-CC); plus a COIN (Contrast) family "
            "with five theme-matched partner sub-corpora (CAU_COIN, MOR_COIN, STP_COIN, STR_COIN, TOM_COIN) "
            "whose target/retain pairs are complete opposites — like sides of a coin. "
            "Each record is a matched target/retain text pair: the target exercises a specific cognitive "
            "skill and the retain is a content-matched control lacking that skill. The suite supports "
            "mechanistic interpretability via Sparse Autoencoder (SAE) analysis and quantitative Cognitive "
            "Absorption (CA) measurement of scaffolding dependence. "
            "Data files are licensed under CC BY 4.0; accompanying generation, evaluation, and training "
            "code (shipped alongside in the same bundle) is separately licensed under the Apache License 2.0."
        ),
        "url": "https://anonymous.4open.science/r/12321_cogbench_primitive",
        "sameAs": "https://anonymous.4open.science/r/12321_cogbench_primitive",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "sdLicense": "https://creativecommons.org/licenses/by/4.0/",
        "citeAs": (
            "@inproceedings{anonymous2026cogbench,\n"
            "  title={CogBench: Contrastive Corpora for Cognitive-Circuit Analysis in Language Models},\n"
            "  author={Anonymous Authors},\n"
            "  booktitle={Submitted to the NeurIPS 2026 Track on Datasets and Benchmarks},\n"
            "  year={2026},\n"
            "  note={Under double-blind review}\n"
            "}"
        ),
        "creator": {"@type": "sc:Person", "name": "Anonymous"},
        "publisher": {"@type": "sc:Organization", "name": "Anonymous"},
        "datePublished": "2026-04-22",
        "dateCreated": "2026-03-19",
        "dateModified": "2026-04-22",
        "version": "1.1.0",
        "inLanguage": "en",
        "isLiveDataset": False,
        "keywords": [
            "contrastive corpus",
            "mechanistic interpretability",
            "sparse autoencoders",
            "theory of mind",
            "causal reasoning",
            "moral reasoning",
            "strategic reasoning",
            "planning",
            "mathematical reasoning",
            "negative control",
            "cognitive absorption",
            "benchmark",
            "NLP",
        ],
        "rai:dataCollection": (
            "All pairs are synthetically generated using a single open-weights teacher model served via a "
            "local inference stack, with structured JSON output enforced by a schema-validating wrapper. "
            "Each pair undergoes a two-stage validation pipeline: (1) deterministic regex gates enforcing "
            "structural constraints (e.g. a cognitive-term blacklist for NULL-CC, math-symbol presence "
            "checks for SPL-CC and CoreMath, counterfactual pattern matching for CTR-CC); (2) a multi-judge "
            "LLM panel applying unanimous- or majority-vote consensus on context-dependent content (ToM, "
            "CTR, MOR, STR, STP, COIN). Pairs failing either stage are rejected and regenerated. The bench, "
            "train, and holdout splits are disjoint by scenario-seed id."
        ),
        "rai:dataCollectionType": ["Synthetic generation", "Software/API", "LLM panel validation"],
        "rai:dataCollectionTimeframe": ["2026-03-19", "2026-04-14"],
        "rai:dataPreprocessingProtocol": [
            "Deterministic regex and structural gates per corpus (math-symbol checks, cognitive blacklist, counterfactual patterns).",
            "Embedding-based near-duplicate filtering within each corpus.",
            "Cross-benchmark decontamination scan against public holdouts (BigBench, BigToM, BCOPA-CE, Hendrycks MATH, FinQA, LegalBench, ScienceQA, CyberMetric).",
            "Split assignment by scenario-seed id so that train / bench / holdout share no seed.",
        ],
        "rai:dataAnnotationProtocol": (
            "Annotations are entirely model-based: a panel of open-weights LLM judges votes accept/reject on "
            "each pair along corpus-specific criteria (target contains the named cognitive skill; retain is "
            "a content-matched control that does not). Consensus rule: unanimous for context-dependent "
            "corpora (ToM, CTR, MOR, STR, STP, COIN), majority for style-only corpora (NULL). No human "
            "annotation was used to construct the release; a human audit protocol over a stratified sample "
            "is described in the accompanying datasheet for reviewer verification."
        ),
        "rai:annotationsPerItem": "2 to 3 LLM-judge votes per pair depending on corpus",
        "rai:machineAnnotationTools": [
            "Open-weights generator LLM served via a local inference stack",
            "Open-weights judge LLMs in a panel-vote configuration",
            "Schema-validating structured-output wrapper",
        ],
        "rai:dataUseCases": [
            "Mechanistic interpretability: identification of cognitive-skill features and circuits in language models via Sparse Autoencoder (SAE) analysis.",
            "Cognitive Absorption (CA) measurement: quantifying scaffolding dependence by comparing target vs. retain activation / prediction statistics.",
            "Contrastive fine-tuning: LoRA / Q-DoRA training on matched target/retain pairs.",
            "Two-dimensional model profiling: raw capability × cognitive-absorption across a model zoo.",
            "Negative-control / null-hypothesis testing: NULL-CC isolates style-only variation from cognitive-content variation.",
        ],
        "rai:dataBiases": [
            "Single-generator bias: all text is produced by one teacher model family, so stylistic and reasoning distributional biases of that family propagate into every corpus.",
            "Judge-generator correlation: validation judges are also LLMs and share pre-training substrates with the generator, which can create correlated false negatives / positives.",
            "Template coverage: scenarios are seeded from hand-curated template categories; domains outside those templates are under-represented.",
            "Language: English-only.",
            "Register asymmetry in NULL-CC: formal/informal contrast is systematic rather than sampled from naturalistic distributions.",
        ],
        "rai:dataLimitations": [
            "The suite is diagnostic, not a capability benchmark; high retain accuracy is expected in some corpora by construction (e.g. NULL-CC) and does not imply model failure.",
            "Cognitive-skill labels are intended properties of the generation prompt and judge consensus, not empirically verified human annotations on the final corpus.",
            "Corpora are in English and reflect the distribution of the single generator model; cross-lingual and cross-model generalisation is out of scope for v1.",
            "CoreMath covers textbook and competition-style topics and should not be used to evaluate frontier mathematical research ability.",
        ],
        "rai:personalSensitiveInformation": [
            "No personally identifying information (PII) is included.",
            "No demographic attributes (gender, age, ethnicity, geography, socio-economic status) are solicited or embedded.",
            "All characters in scenarios are fictional placeholders (e.g. 'Senior Analyst A').",
        ],
        "rai:dataSocialImpact": (
            "The dataset is intended to support interpretability research that makes cognitive-skill "
            "representations in language models more legible and auditable. Because it is fully synthetic "
            "with no human data collection, it poses minimal direct privacy risk. Misuse risk is analogous "
            "to any capability-eliciting corpus: text from target pairs could in principle be used as "
            "training data to boost narrow cognitive skills without the accompanying absorption controls, "
            "undermining the diagnostic contract. We recommend paired use of target and retain splits."
        ),
        "rai:dataReleaseMaintenancePlan": (
            "Versioning follows semantic versioning (MAJOR.MINOR.PATCH); v1.1.0 is the NeurIPS 2026 "
            "submission snapshot. Maintenance window: bug fixes and checksum re-issuance for at least "
            "24 months post-acceptance. Superseded versions remain accessible via the hosting platform DOI; "
            "breaking schema changes bump MAJOR. For camera-ready we will publish the dataset on a "
            "long-term archive with a DOI (Hugging Face Hub, Harvard Dataverse, or Zenodo) preserving the "
            "train / bench / holdout / coin file layout."
        ),
        "distribution": file_objects,
        "recordSet": record_sets,
    }
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.root / "croissant.json")
    doc = build(args.root)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n_files = len(doc["distribution"])
    n_records = sum(1 for _ in doc["recordSet"])
    print(f"wrote {out} — {n_files} FileObjects / {n_records} RecordSets")


if __name__ == "__main__":
    main()