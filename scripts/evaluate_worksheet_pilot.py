"""Compare paired manual and assisted worksheet sessions against a reference CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from statistics import median


def read_csv(path: Path, fields: set[str], *, optional_fields: Iterable[str] = ()) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        if not fields <= columns <= fields.union(optional_fields):
            raise ValueError(f"{path} needs these columns: {', '.join(sorted(fields))}.")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path} has an incomplete row on line {reader.line_num}.")
            yield row


def answer_key(row: dict[str, str]) -> tuple[str, str]:
    page, question = row["page_id"], row["question"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", page) or question not in {"1", "2", "3", "4"}:
        raise ValueError(f"Invalid page or question: {page!r}, {question!r}.")
    return page, question


def normalize(text: str, kind: str) -> str:
    value = text.strip().upper()
    pattern = {"digits": r"[0-9]*", "word": r"[A-Z0-9]*"}[kind]
    if not re.fullmatch(pattern, value):
        raise ValueError(f"Invalid {kind} transcription: {text!r}.")
    return value


def evaluate(reference_path: Path, exports: Iterable[Path]) -> dict:
    reference = {}
    for row in read_csv(reference_path, {"page_id", "question", "kind", "answer"}):
        key, kind = answer_key(row), row["kind"]
        if kind not in {"digits", "word"} or key in reference:
            raise ValueError(f"Invalid kind or duplicate answer for {key}.")
        reference[key] = (kind, normalize(row["answer"], kind))
    if not reference:
        raise ValueError("The reference has no answers.")
    pages = {key[0] for key in reference}
    if len(reference) != 4 * len(pages):
        raise ValueError("Include all four questions for each reference page, including blank answers.")

    methods = {"assisted": {}, "manual": {}}
    times = {"assisted": {}, "manual": {}}
    recognition_failures = set()
    unverified_empty_suggestions = set()
    fields = {"page_id", "question", "kind", "method", "suggestion", "transcription", "seconds"}
    for path in exports:
        for row in read_csv(path, fields, optional_fields=("suggestion_status",)):
            key, method = answer_key(row), row["method"]
            if key not in reference or method not in methods or key in methods[method]:
                raise ValueError(f"Unexpected or duplicate session answer for {key} in {path}.")
            kind, target = reference[key]
            if row["kind"] != kind:
                raise ValueError(f"Unexpected answer kind for {key} in {path}.")
            status = row.get("suggestion_status")
            suggestion = normalize(row["suggestion"], kind)
            if method == "manual":
                if row["suggestion"] or status not in (None, ""):
                    raise ValueError(f"Unexpected manual suggestion or status for {key} in {path}.")
            elif status is None:
                if not suggestion:
                    unverified_empty_suggestions.add(key)
            elif status == "failed" and not suggestion:
                recognition_failures.add(key)
            elif status != "ok":
                raise ValueError(f"Invalid suggestion status for {key} in {path}.")
            seconds = float(row["seconds"])
            if not math.isfinite(seconds) or seconds <= 0 or times[method].setdefault(key[0], seconds) != seconds:
                raise ValueError(f"Inconsistent or invalid elapsed time for {key[0]} in {path}.")
            methods[method][key] = (
                (status == "ok" or bool(suggestion)) and suggestion == target,
                normalize(row["transcription"], kind) == target,
            )
    for method, answers in methods.items():
        if missing := reference.keys() - answers.keys():
            raise ValueError(f"Missing {method} results for {len(missing)} answers (first: {min(missing)}).")

    def scores(keys: list[tuple[str, str]]) -> dict[str, int | float | None]:
        total = len(keys)
        suggestion_errors = sum(not methods["assisted"][key][0] for key in keys)
        return {
            "answers": total,
            "assisted_suggestion_errors": suggestion_errors,
            "assisted_suggestion_accuracy_pct": 100 * (total - suggestion_errors) / total if total else None,
            "assisted_recognition_failures": sum(key in recognition_failures for key in keys),
            "assisted_unverified_empty_suggestions": sum(key in unverified_empty_suggestions for key in keys),
            "assisted_final_errors": sum(not methods["assisted"][key][1] for key in keys),
            "manual_final_errors": sum(not methods["manual"][key][1] for key in keys),
        }

    return {
        "paired_pages": len(pages),
        "all": scores(list(reference)),
        "digits": scores([key for key, (kind, _) in reference.items() if kind == "digits"]),
        "words": scores([key for key, (kind, _) in reference.items() if kind == "word"]),
        "median_seconds_per_page": {method: median(values.values()) for method, values in times.items()},
        "median_paired_seconds_saved": median(times["manual"][page] - times["assisted"][page] for page in pages),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("exports", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(args.reference, args.exports), indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
