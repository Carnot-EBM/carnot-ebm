"""REQ-REPORT-9056-AUTHORITY: release structure cannot replace original human votes.

Read only prompt/response and source metadata. Original workbooks remain sealed
as evidence; aggregate judgments never become training targets or planning data.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.human_source_authority_9001 import normalized
from carnot.reporting.natural_label_eligibility_8975 import ref as ref

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_9056_v773_human_release_authority"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_mdh_human_release_9056.py"
OWNED = [
    "python/carnot/reporting/mdh_human_release_9056.py",
    "python/carnot/reporting/mdh_human_release_runner_9056.py",
    CLI,
]
REVISION = "7a1ea413106baa9943bb87a6db38dcf262c5a9ae"
DEFAULT_INPUT = ROOT / "results/raw" / NAME / "release"
MODEL_SPECS: list[Json] = []
NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
TAG = "{" + NS + "}"
ALLOWLIST = {
    "Category": "category",
    "Prompt ID": "prompt_id",
    "Prompts": "prompt",
    "Model Responses": "response",
}
PINS = {
    "metadata.json": "6d679e05d7a332fc662f911f8e21d8d00c8635af01d6c9eb2a690f61dc766c90",
    "ChatGPT Prompts and Responses.xlsx": "c9a637b0e404930a66b1ff3ea7c3469ccbf93961fec6c118e230f79964915803",
    "Claude Prompts and Responses.xlsx": "f90cc623d29b02e8c23b8fe6ab1e0df74f1ed295e943ae15afae58da370f5552",
    "DeepSeek Prompts and Responses.xlsx": "38477d561d532feaf0539960733d0209b14e0dd9dbee766523dae88d3253a9cb",
    "Gemini Prompts and Responses.xlsx": "85fe29c72919293419bf2b1d73c71ae4b5c3832164fbff24246d66fe91c781a2",
    "README.md": "9fc7075fc8109fff237ebaae42d5957278728f94f78e240379ee61c8cb55c9af",
    "configclaude.txt": "3f3cbcc5f7686e2dfed48c006e2b33dbe80d4c468a31643a0703eb90d1aaf60e",
    "configdeepseek.txt": "b8c5bc55a963536a949e9b19fa87da117b7228df66a16ed743a2b0a604343bb5",
    "configgemini.txt": "2a72572fc96544805cf1a22d58765ead3aecaef7cc5915d66854d953cd0c06c5",
    "configgpt.txt": "73032c525609708ab010aff894f7e60becdb7ee5037f3c7df06b06ff2826d323",
    "acquisition.json": "853ed7c69d7a0140d5887b3939b8d37f3b994fa271e01c10259406b3c2cce2da",
}
HISTORY = {
    "results/experiment_9043_v772_identity_obstruction_audit.json": "644e3db049e45b9fb9d80984a7083bf4afd32b53a0cb7eb4b9214227a758e324",
    "results/experiment_9044_v772_annotation_feasibility.json": "288ad02b0b47ff820a9afef66269a640a42a6465d389af002368fe6e9cfeec3a",
    "results/experiment_9053_v772_capstone.json": "1cf9e996fe6bb325db1df53c2fb8ffc9c31f373e2b74f1bba6bf1b3cf9c812ac",
    "openspec/change-proposals/research-roadmap-v771-preserved-20261112.md": "645327851ea953d7cfd0961838dcc36eb7e10dd8dbabd32a81c52adadb76d3fd",
    "openspec/change-proposals/research-roadmap-v771-queue-preserved-20261112.yaml": "2f18f190413146ad123228723a2e9d0780981a5c69a471473b5d06b757041d42",
}
INVENTORY = (
    "results/raw/experiment_9015_v770_human_disagreement_source/release/exposure_inventory.json"
)
INVENTORY_PIN = "ab488648f73f763d0c2ddaf5d1f6388dbb24572d0ab5691598ec49691bbe8c0e"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report completed units so blocked inputs cannot look like silent computation."""
    print(f"[exp9056] phase={phase} completed={completed} pending={pending}", flush=True)


def gate(check: str, expected: Any, observed: Any, reference: Json) -> Json:
    """Keep absence distinct from zero and name the exact authenticated operand."""
    return dict(
        check=check,
        upstream="author_MDH_release",
        **reference,
        artifact_field=check,
        op="==",
        expected=expected,
        observed=observed,
        passed=expected == observed,
    )


def cell_text(cell: ET.Element, shared: list[ET.Element]) -> str:
    """Decode only explicitly selected cells; excluded target strings stay unprojected."""
    if cell.get("t") == "inlineStr":
        return "".join(cell.itertext())
    value = cell.find(TAG + "v")
    text = "" if value is None else value.text or ""
    if cell.get("t") == "s":
        return "".join(shared[int(text)].itertext())
    return text


def workbook(path: Path) -> Json:
    """Inspect ZIP/XML without loading macros, formulas, external links or model code."""
    records: list[Json] = []
    sheets: list[Json] = []
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if sum(m.file_size for m in members) > 100_000_000 or any(
            "vbaproject" in m.filename.lower() or "externallinks/" in m.filename.lower()
            for m in members
        ):
            raise ValueError("unsafe_workbook")
        member_rows = [
            dict(
                member=m.filename,
                bytes=m.file_size,
                sha256="sha256:" + hashlib.sha256(archive.read(m)).hexdigest(),
            )
            for m in members
        ]
        shared = (
            ET.fromstring(archive.read("xl/sharedStrings.xml")).findall(TAG + "si")
            if "xl/sharedStrings.xml" in archive.namelist()
            else []
        )
        sheet_members = sorted(
            m.filename for m in members if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", m.filename)
        )
        for member in sheet_members:
            rows = ET.fromstring(archive.read(member)).findall(TAG + "sheetData/" + TAG + "row")
            headers = (
                {
                    re.sub(r"\d", "", c.get("r", "")): " ".join(cell_text(c, shared).split())
                    for c in rows[0]
                }
                if rows
                else {}
            )
            selected = {
                column: ALLOWLIST[header]
                for column, header in headers.items()
                if header in ALLOWLIST
            }
            response_sheet = "response" in selected.values() and "prompt" in selected.values()
            count = 0
            category = ""
            if response_sheet:
                for row in rows[1:]:
                    cells = {
                        selected[col]: c
                        for c in row
                        if (col := re.sub(r"\d", "", c.get("r", ""))) in selected
                    }
                    ident = cell_text(cells["prompt_id"], shared) if "prompt_id" in cells else ""
                    prompt = cells.get("prompt")
                    if (
                        not ident.isdecimal()
                        or prompt is None
                        or prompt.get("t") not in {"s", "inlineStr"}
                        or prompt.find(TAG + "f") is not None
                    ):
                        continue
                    public = {key: cell_text(c, shared) for key, c in cells.items()}
                    if not public.get("prompt"):
                        continue
                    category = public.get("category") or category
                    public["category"] = category
                    records.append(
                        dict(
                            unit_id=path.stem + ":" + member + ":" + row.get("r", ""),
                            workbook=path.name,
                            member=member,
                            row=row.get("r"),
                            **public,
                            original_judgment_authority="unknown",
                        )
                    )
                    count += 1
            sheets.append(
                dict(
                    workbook=path.name,
                    member=member,
                    headers=headers,
                    allowed_columns=selected,
                    protected_columns={c: h for c, h in headers.items() if c not in selected},
                    xml_row_count=len(rows),
                    response_slot_count=count,
                    original_judgment_fields_authenticated=False,
                    run_authority="sheet position only; generation/model version and human participation not authenticated",
                )
            )
            progress("sheet_inventory", len(sheets), len(sheet_members) - len(sheets))
    return dict(records=records, sheets=sheets, members=member_rows)


def groups(records: list[Json]) -> list[Json]:
    """Merge metadata and similar prompts transitively before assigning any outcomes.

    Text similarity finds candidate families only. Missing source-family authority
    remains unknown; this heuristic cannot certify independence or nonexposure.
    """
    items = sorted(records, key=lambda r: r["unit_id"])
    parents = list(range(len(items)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen: dict[str, int] = {}
    unique: dict[str, int] = {}
    for i, row in enumerate(items):
        text = normalized(row.get("prompt", ""))
        keys = ["prompt:" + canonical_hash(text)] if text else []
        if row.get("prompt_id"):
            keys.append(
                "id:" + normalized(row.get("category", "")) + ":" + normalized(row["prompt_id"])
            )
        for key in keys:
            if key in seen:
                parents[find(i)] = find(seen[key])
            seen[key] = i
        if text:
            unique[text] = i
    texts = [(text, i, set(re.findall(r"\w+", text))) for text, i in sorted(unique.items())]
    for n, (_, i, tokens) in enumerate(texts):
        for _, j, other in texts[:n]:
            if tokens and len(tokens & other) / len(tokens | other) >= 0.85:
                parents[find(i)] = find(j)
        if n % 100 == 0:
            progress("prompt_similarity", n + 1, len(texts) - n - 1)
    components: dict[int, list[Json]] = defaultdict(list)
    for i, row in enumerate(items):
        components[find(i)].append(row)
    result = []
    for rows in components.values():
        keys = sorted({"prompt:" + canonical_hash(normalized(r.get("prompt", ""))) for r in rows})
        ident = canonical_hash(keys)
        result.append(
            dict(
                group_id=ident,
                prompt_keys=keys,
                unit_ids=sorted(r["unit_id"] for r in rows),
                source_family_authority="unknown",
                role="UNASSIGNED",
                rank_sha256=hashlib.sha256(("v773-mdh-v1:" + ident).encode()).hexdigest(),
            )
        )
    return sorted(result, key=lambda g: g["rank_sha256"])


def inspect(directory: Path) -> Json:
    """Authenticate all source bytes before projection; absent authority stops admission."""
    gates: list[Json] = []
    references: list[Json] = []
    records: list[Json] = []
    sheets: list[Json] = []
    members: list[Json] = []
    clusters: list[Json] = []
    joins: list[Json] = []
    metadata: Json = {}
    history_rows: list[Json] = []
    try:
        for name, pin in PINS.items():
            reference = ref(directory / name)
            operand = gate(
                "release_bytes:" + name,
                "sha256:" + pin,
                reference["sha256"] or "missing_file",
                reference,
            )
            gates.append(operand)
            if not operand["passed"]:
                raise ValueError("release_authority")
            references.append(reference)
        metadata = json.loads((directory / "metadata.json").read_bytes())
        for field, expected in [("sha", REVISION), ("cardData.license", "mit")]:
            observed = (
                metadata.get("sha", "missing_field")
                if field == "sha"
                else metadata.get("cardData", {}).get("license", "missing_field")
            )
            gates.append(gate(field, expected, observed, ref(directory / "metadata.json")))
        for name, pin in {**HISTORY, INVENTORY: INVENTORY_PIN}.items():
            reference = ref(ROOT / name)
            operand = gate(
                "historical_bytes:" + name,
                "sha256:" + pin,
                reference["sha256"] or "missing_file",
                reference,
            )
            gates.append(operand)
            if not operand["passed"]:
                raise ValueError("historical_authority")
            references.append(reference)
            if name.startswith("results/experiment_"):
                old = json.loads((ROOT / name).read_bytes())
                history_rows.append(
                    dict(
                        reference=reference,
                        honest_verdict=old["honest_verdict"],
                        verdict_class=old["verdict_class"],
                        source_identity_ready_score=old.get(
                            "source_identity_ready_score", "missing_field"
                        ),
                        annotation_structure_ready_score=old.get(
                            "annotation_structure_ready_score", "missing_field"
                        ),
                    )
                )
        for name in PINS:
            if name.endswith(".xlsx"):
                progress("before_workbook")
                audit = workbook(directory / name)
                records.extend(audit["records"])
                sheets.extend(audit["sheets"])
                members.extend(dict(workbook=name, **m) for m in audit["members"])
                progress("after_workbook", len(records), 0)
        clusters = groups(records)
        inventory = json.loads((ROOT / INVENTORY).read_bytes())
        exposed = set(inventory["exposed_keys"])
        integrity = True
        for index, reference in enumerate(inventory["rows"]):
            actual = ref(Path(reference["path"]))
            integrity = integrity and actual["sha256"] == reference["sha256"]
            references.append(dict(path=reference["path"], sha256=reference["sha256"]))
            if index % 50 == 0:
                progress("exposure_ledger_hashes", index + 1, len(inventory["rows"]) - index - 1)
        for cluster in clusters:
            overlap = sorted(set(cluster["prompt_keys"]) & exposed)
            joins.append(
                dict(
                    group_id=cluster["group_id"],
                    inventory_reference=ref(ROOT / INVENTORY),
                    exact_prompt_matches=overlap,
                    eligible=False,
                    status="exposed" if overlap else "unknown_source_family_overlap",
                    corpora=["LPHB", "FaithBench", "HalluTruthQA", "Mu-SHROOM"],
                    unresolved_historical_identity_is_distinct=False,
                )
            )
        gates += [
            gate(
                "original_independent_human_judgments_per_response",
                "at_least_3_attributable_completed_original_judgments",
                "missing_field:original_individual_judgments",
                ref(directory / "metadata.json"),
            ),
            gate(
                "original_factual_error_criterion",
                "author_instructions_and_response_level_mapping",
                "missing_field:original_annotation_instructions",
                ref(directory / "README.md"),
            ),
            gate("exposure_ledger_integrity", True, integrity, ref(ROOT / INVENTORY)),
            gate("source_family_and_exposure_closure", True, False, ref(ROOT / INVENTORY)),
            gate(
                "independently_authenticated_unexposed_prompt_groups",
                400,
                "unknown",
                ref(ROOT / INVENTORY),
            ),
        ]
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, ET.ParseError) as error:
        if not gates or gates[-1]["passed"]:
            gates.append(
                gate("release_structure", "readable_author_bytes", str(error), ref(directory))
            )
    by_unit = {unit: group["group_id"] for group in clusters for unit in group["unit_ids"]}
    authority = [
        dict(
            unit_id=r["unit_id"],
            group_id=by_unit.get(r["unit_id"], "unknown"),
            workbook=r["workbook"],
            member=r["member"],
            row=r["row"],
            original_annotator_ids="missing_field",
            completed_individual_judgments="unknown",
            disagreement="unknown",
            missingness="individual_judgments_not_released_as_attributable_fields",
            criterion="unqualified",
            original_response=bool(r.get("response")),
            original_prompt=bool(r.get("prompt")),
            generation_authority="unverified_original_release_only",
            status="blocked",
        )
        for r in records
    ]
    units = [
        dict(
            unit_id=r["unit_id"],
            group_id=r["group_id"],
            arm="released_original",
            condition=r["workbook"] + ":" + r["member"],
            status="completed",
            metric="human_authority_audit",
            value=0,
            numerator=0,
            denominator=1,
            exclusion_reason="unknown_original_human_authority",
        )
        for r in authority
    ]
    return dict(
        gate_check_summary=gates,
        source_artifact_hashes=references,
        records=records,
        workbook_schema_rows=sheets,
        workbook_member_rows=members,
        annotation_authority_rows=authority,
        group_rows=clusters,
        exposure_join_rows=joins,
        historical_verdicts=history_rows,
        rows=units,
        intended_count=len(units),
        completed_count=len(units),
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        human_source_ready_score=0,
        release_revision=REVISION,
        release_hashes=[r for r in references if Path(r["path"]).parent == directory.absolute()],
        rights_reference=dict(
            reference=ref(directory / "README.md"),
            declaration=metadata.get("cardData", {}).get("license", "unknown"),
            scope="MIT dataset card declaration; no separate license text or attribution of individual judgments; article rights not imported",
        ),
        candidate_prompt_group_count=len(clusters),
        source_family_closure=False,
        protected_outcome_values_projected=False,
    )
