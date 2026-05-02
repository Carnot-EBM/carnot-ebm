"""Tests for the Exp 1135 position-paper findings update.

Spec traces: REQ-PUBLISH-004, SCENARIO-PUBLISH-004.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts import experiment_1135_position_paper_v3_findings_update as exp1135


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _source_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "exp1118": _write_json(
            tmp_path / "exp1118.json",
            {
                "improvement_over_baseline": 0.04,
                "thinkprm_v2_auroc": 0.9946,
            },
        ),
        "exp1120": _write_json(
            tmp_path / "exp1120.json",
            {
                "retrained_auroc_val": 0.977419,
                "energy_inversion_fixed": True,
                "n_raw_corpus": 7329,
            },
        ),
        "exp1121": _write_json(
            tmp_path / "exp1121.json",
            {
                "k5_and_compose_production_deployed": True,
                "benchmark_k5_auroc": 0.5547,
            },
        ),
        "exp1129": _write_json(
            tmp_path / "exp1129.json",
            {
                "improvement_over_baseline": 0.0851,
                "thinkprm_v2_auroc": 0.9946,
            },
        ),
        "exp1130": _write_json(
            tmp_path / "exp1130.json",
            {
                "alpha_t_post_retrain": 0.52,
                "alpha_t_prior": 0.38,
            },
        ),
    }


def _updated_tex() -> str:
    return r"""
\begin{abstract}
GRPO with ThinkPRM v2 improves held-out GSM8K by +4 pp to +8.51 pp.
\end{abstract}
\section{Empirical Realities \& Anomalies}
Energy Verifier Calibration (Milestone .87): Training the SOS-KAN verifier on a
7,329-pair corpus fixed the energy inversion. Post-retrain AUROC=0.9774.
correct energy ordering restored.
GRPO with Energy Reward (Milestone .87-.88): ThinkPRM v2 (AUROC=0.9946)
achieved +4 pp to +8.51 pp on held-out GSM8K.
k=5 AND-compose production deployment (exp1121): deployed.
Zenil alpha_t self-distillation grounding (Milestone .88): alpha_t measured at
0.52 with retrained verifier vs prior 0.38.
\section{Related Work}
HIVE~\cite{hive2026} detects hallucinations; Carnot additionally provides
energy-guided repair and local-first deployment.
\section{Conclusion \& Roadmap}
The retrained verifier fixes the observed energy ordering; remaining work is
generalization.
"""


def test_collect_findings_derives_req_publish_004_metrics(tmp_path):
    findings = exp1135.collect_findings(_source_paths(tmp_path))

    assert findings.energy_auroc == 0.9774
    assert findings.energy_inversion_fixed is True
    assert findings.sota_corpus_pairs == 7329
    assert findings.grpo_v1_pp == 4.0
    assert findings.grpo_v2_pp == 8.51
    assert findings.thinkprm_v2_auroc == 0.9946
    assert findings.k5_deployed is True
    assert findings.k5_auroc == 0.5547
    assert findings.alpha_t_post_retrain == 0.52
    assert findings.alpha_t_prior == 0.38


def test_detect_integrations_for_scenario_publish_004():
    flags = exp1135.detect_integrations(
        tex_text=_updated_tex(),
        bib_text="@article{hive2026, title={HIVE}, journal={arXiv:2604.26139}}",
    )

    assert flags["grpo_result_integrated"] is True
    assert flags["energy_inversion_result_integrated"] is True
    assert flags["zenil_alpha_t_result_integrated"] is True
    assert flags["hive_related_work_added"] is True
    assert flags["abstract_updated"] is True
    assert flags["results_updated"] is True
    assert flags["related_work_updated"] is True
    assert flags["conclusion_updated"] is True


def test_classify_verdict_closed_set_req_publish_004():
    fully_updated = {
        "grpo_result_integrated": True,
        "energy_inversion_result_integrated": True,
        "zenil_alpha_t_result_integrated": True,
        "hive_related_work_added": True,
    }
    grpo_pending = {**fully_updated, "grpo_result_integrated": False}
    minor_only = {**fully_updated, "energy_inversion_result_integrated": False}

    assert exp1135.classify_verdict(fully_updated) == "fully_updated"
    assert exp1135.classify_verdict(grpo_pending) == "partially_updated_grpo_pending"
    assert exp1135.classify_verdict(minor_only) == "minor_edits_only"


def test_build_artifact_schema_req_publish_004(tmp_path):
    findings = exp1135.collect_findings(_source_paths(tmp_path))
    artifact = exp1135.build_artifact(
        findings=findings,
        tex_text=_updated_tex(),
        bib_text="@article{hive2026, title={HIVE}, journal={arXiv:2604.26139}}",
    )

    required = {
        "sections_modified",
        "new_citations_added",
        "grpo_result_integrated",
        "energy_inversion_result_integrated",
        "zenil_alpha_t_result_integrated",
        "hive_related_work_added",
        "position_paper_findings_updated",
        "honest_verdict",
    }
    assert required.issubset(artifact)
    assert artifact["sections_modified"] == [
        "Abstract",
        "Results",
        "Related Work",
        "Conclusion",
    ]
    assert artifact["new_citations_added"] == ["HIVE 2604.26139"]
    assert artifact["position_paper_findings_updated"] is True
    assert artifact["honest_verdict"] == "fully_updated"
    assert artifact["source_metrics"]["grpo_v2_improvement_pp"] == 8.51


def test_main_writes_exp1135_deliverable_for_scenario_publish_004(tmp_path):
    paths = _source_paths(tmp_path)
    tex_path = tmp_path / "main.tex"
    bib_path = tmp_path / "carnot.bib"
    out_path = tmp_path / "experiment_1135.json"
    tex_path.write_text(_updated_tex(), encoding="utf-8")
    bib_path.write_text(
        "@article{hive2026, title={HIVE}, journal={arXiv:2604.26139}}",
        encoding="utf-8",
    )

    code = exp1135.main(
        [
            "--main-tex",
            str(tex_path),
            "--bib",
            str(bib_path),
            "--exp1118",
            str(paths["exp1118"]),
            "--exp1120",
            str(paths["exp1120"]),
            "--exp1121",
            str(paths["exp1121"]),
            "--exp1129",
            str(paths["exp1129"]),
            "--exp1130",
            str(paths["exp1130"]),
            "--out",
            str(out_path),
        ]
    )

    artifact = json.loads(out_path.read_text(encoding="utf-8"))
    assert code == 0
    assert artifact["honest_verdict"] == "fully_updated"
    assert artifact["grpo_result_integrated"] is True
