from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_loong_jsonl(project_root: Optional[Path] = None) -> Path:
    root = project_root or default_project_root()
    processed = root / "Loong" / "data" / "loong_process.jsonl"
    raw = root / "Loong" / "data" / "loong.jsonl"
    if processed.exists():
        return processed
    if raw.exists():
        return raw
    raise FileNotFoundError(f"Loong dataset not found: {processed} or {raw}")


def load_loong_records(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    jsonl_path = path or default_loong_jsonl()
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass
class ManifestItem:
    selected_index: int
    record_id: str
    set_id: int
    record_type: str
    level: int
    language: str
    question: str
    sample_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionEvaluation:
    prediction: Any
    gold_answer: Any
    exact_match: Optional[bool]
    pair_precision: Optional[float]
    pair_recall: Optional[float]
    pair_f1: Optional[float]
    missing: List[Any]
    extra: List[Any]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_domain_level_manifest(
    records: Sequence[Dict[str, Any]],
    max_per_combo: int = 1,
    require_question: bool = True,
) -> List[ManifestItem]:
    counts: Dict[Tuple[str, int], int] = {}
    manifest: List[ManifestItem] = []
    for index, record in enumerate(records):
        question = str(record.get("question", "")).strip()
        if require_question and not question:
            continue
        record_type = str(record.get("type", "unknown"))
        level = int(record.get("level", 0) or 0)
        key = (record_type, level)
        current = counts.get(key, 0)
        if current >= max_per_combo:
            continue
        counts[key] = current + 1
        manifest.append(
            ManifestItem(
                selected_index=index,
                record_id=str(record.get("id", "unknown")),
                set_id=int(record.get("set", 0) or 0),
                record_type=record_type,
                level=level,
                language=str(record.get("language", "unknown")),
                question=question,
                sample_id=f"{record_type}_level{level}_{current + 1}",
            )
        )
    return manifest


def save_manifest(items: Sequence[ManifestItem], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([item.to_dict() for item in items], ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _normalize_scalar(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    compact = re.sub(r"\s+", "", text)
    numeric_like = bool(re.fullmatch(r"[()\-+0-9,.$¥€£%元美元股]+", compact))
    if not numeric_like:
        return re.sub(r"\s+", " ", text)
    parenthesized = bool(re.search(r"\(\s*[$¥€£]?\s*\d[\d,]*(?:\.\d+)?\s*(?:元|%|美元|股)?\s*\)", text))
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not match:
        return re.sub(r"\s+", " ", text)
    number = match.group(0).replace(",", "")
    if parenthesized and not number.startswith("-"):
        number = f"-{number}"
    if "$" in text:
        return f"${number}" if not number.startswith("-") else f"-${number[1:]}"
    if "元" in text:
        return f"{number}元"
    if "%" in text:
        return f"{number}%"
    return number


def _normalize_mapping_value(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^\#+\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_mapping_values(value: Any) -> List[str]:
    raw_values = value if isinstance(value, list) else [value]
    values = []
    for item in raw_values:
        text = _normalize_mapping_value(item)
        if text and text.casefold() not in {"none", "null"}:
            values.append(text)
    return sorted(set(values), key=_label_sort_key)


def _label_sort_key(label: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)", label)
    if match:
        return int(match.group(1)), label
    return 10**9, label


def _coerce_gold_answer(value: Any) -> Any:
    if isinstance(value, (dict, list, int, float)):
        return value
    text = str(value or "").strip()
    if not text:
        return text
    try:
        return json.loads(text)
    except Exception:
        return text


def project_answer_from_checker(checker_output: Dict[str, Any], planner_output: Optional[Dict[str, Any]] = None) -> Any:
    if not checker_output.get("sufficiency_verdict", False):
        return None
    projected = checker_output.get("projected_answer_state", []) or []
    task_model = planner_output.get("task_model", []) if planner_output else []
    answer_schema = planner_output.get("answer_schema", []) if planner_output else []
    task_model_primary = task_model[0] if task_model else {}
    answer_topology = str(task_model_primary.get("answer_topology", "freeform")).strip().lower()
    expected_value_by_key = {
        str(item.get("key", "")).strip(): str(item.get("expected_value", "")).strip().lower()
        for item in answer_schema
        if str(item.get("key", "")).strip()
    }

    if answer_topology in {"mapping", "grouping"}:
        grouped: Dict[str, List[str]] = {}
        for item in projected:
            key = str(item.get("answer_key", "")).strip()
            value = str(item.get("value", "")).strip()
            if not key:
                continue
            if value == "[]":
                grouped.setdefault(key, [])
                continue
            if value.startswith("[") and value.endswith("]"):
                try:
                    maybe_list = json.loads(value.replace("'", '"'))
                    if isinstance(maybe_list, list):
                        for entry in maybe_list:
                            entry_text = str(entry).strip()
                            if entry_text:
                                grouped.setdefault(key, []).append(entry_text)
                        continue
                except Exception:
                    pass
            if value:
                grouped.setdefault(key, []).append(value)

        for key in list(grouped.keys()):
            grouped[key] = sorted(set(grouped[key]), key=_label_sort_key)

        if answer_topology == "grouping":
            return grouped

        result: Dict[str, Any] = {}
        for key, values in grouped.items():
            expected = expected_value_by_key.get(key, "")
            expects_list = "list" in expected or expected in {"title_list", "doc_list", "grouped_titles"}
            if expects_list:
                result[key] = values
            elif len(values) == 1:
                result[key] = values[0]
            else:
                result[key] = values
        return result

    values = [str(item.get("value", "")).strip() for item in projected if str(item.get("value", "")).strip()]
    if answer_topology in {"scalar", "ranking"} and values:
        return values[0]
    if values:
        return values
    return None


def evaluate_prediction_against_gold(prediction: Any, gold_answer: Any) -> Dict[str, Any]:
    if isinstance(gold_answer, dict) and isinstance(prediction, dict):
        normalized_prediction = {key: _normalize_mapping_values(values) for key, values in prediction.items()}
        normalized_gold = {key: _normalize_mapping_values(values) for key, values in gold_answer.items()}
        gold_pairs = {(key, value) for key, values in normalized_gold.items() for value in values}
        pred_pairs = {(key, value) for key, values in normalized_prediction.items() for value in values}
        true_positive = len(gold_pairs & pred_pairs)
        precision = true_positive / len(pred_pairs) if pred_pairs else 1.0
        recall = true_positive / len(gold_pairs) if gold_pairs else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {
            "exact_match": normalized_prediction == normalized_gold,
            "pair_precision": precision,
            "pair_recall": recall,
            "pair_f1": f1,
            "missing": sorted(gold_pairs - pred_pairs),
            "extra": sorted(pred_pairs - gold_pairs),
        }

    exact_match = _normalize_scalar(prediction) == _normalize_scalar(gold_answer)
    return {
        "exact_match": exact_match,
        "pair_precision": None,
        "pair_recall": None,
        "pair_f1": None,
        "missing": [] if exact_match else [gold_answer],
        "extra": [] if exact_match else [prediction],
    }


def evaluate_prediction(prediction: Any, gold_answer: Any) -> PredictionEvaluation:
    gold = _coerce_gold_answer(gold_answer)
    metrics = evaluate_prediction_against_gold(prediction, gold)
    return PredictionEvaluation(
        prediction=prediction,
        gold_answer=gold,
        exact_match=metrics["exact_match"],
        pair_precision=metrics["pair_precision"],
        pair_recall=metrics["pair_recall"],
        pair_f1=metrics["pair_f1"],
        missing=metrics["missing"],
        extra=metrics["extra"],
        notes="Mapping evaluation." if isinstance(gold, dict) else "Scalar or single-value evaluation.",
    )

