"""Language augmentation utilities for canonical robotics tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union


DEFAULT_PARAPHRASE_TEMPLATES = {
    "put": ["place {object} {relation} {target}", "move {object} so it is {relation} {target}"],
    "pick": ["grasp {object}", "lift {object}"],
    "open": ["pull open {object}", "make {object} open"],
    "close": ["shut {object}", "make {object} closed"],
    "turn": ["switch {object}", "rotate {object}"],
    "generic": ["perform the task: {instruction}", "complete this instruction: {instruction}"],
}


def paraphrase_instruction(instruction: str, *, task_id: str = "unknown") -> List[Dict[str, str]]:
    """Generate deterministic template paraphrases for one instruction."""

    normalized = " ".join(instruction.strip().split())
    if not normalized:
        raise ValueError("instruction cannot be empty")

    slots = _infer_slots(normalized)
    templates = _templates_for_instruction(normalized)
    paraphrases = [normalized]
    for template in templates:
        text = template.format(**slots)
        text = " ".join(text.strip().split())
        if text and text not in paraphrases:
            paraphrases.append(text)

    return [
        {
            "task_id": task_id,
            "intent": task_id,
            "original": normalized,
            "paraphrase": paraphrase,
        }
        for paraphrase in paraphrases
    ]


def build_paraphrase_set(
    instructions: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen = set()
    for item in instructions:
        for row in paraphrase_instruction(item["instruction"], task_id=item.get("task_id", "unknown")):
            key = (row["task_id"], row["paraphrase"])
            if key not in seen:
                rows.append(row)
                seen.add(key)
    return rows


def validate_paraphrase_rows(rows: Sequence[Dict[str, str]]) -> None:
    for idx, row in enumerate(rows):
        for key in ("task_id", "intent", "original", "paraphrase"):
            if not row.get(key):
                raise ValueError(f"paraphrase row {idx} is missing {key}")
        if row["intent"] != row["task_id"]:
            raise ValueError(f"paraphrase row {idx} does not preserve intent")


def write_paraphrases(rows: Sequence[Dict[str, str]], out: Union[str, Path]) -> Path:
    validate_paraphrase_rows(rows)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(rows), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _templates_for_instruction(instruction: str) -> List[str]:
    first = instruction.split()[0].lower()
    return DEFAULT_PARAPHRASE_TEMPLATES.get(first, DEFAULT_PARAPHRASE_TEMPLATES["generic"])


def _infer_slots(instruction: str) -> Dict[str, str]:
    words = instruction.split()
    first = words[0].lower()
    if first in {"put", "place", "move"} and " on " in f" {instruction} ":
        before, after = instruction.split(" on ", maxsplit=1)
        obj = before.replace(first, "", 1).strip() or "the object"
        return {
            "instruction": instruction,
            "object": obj,
            "relation": "on",
            "target": after.strip() or "the target",
        }
    if len(words) > 1:
        obj = " ".join(words[1:])
    else:
        obj = "the object"
    return {
        "instruction": instruction,
        "object": obj,
        "relation": "near",
        "target": "the goal",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instructions", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    instructions = [
        {"task_id": f"manual_{idx:03d}", "instruction": instruction}
        for idx, instruction in enumerate(args.instructions)
    ]
    if not instructions:
        instructions = [
            {"task_id": "put_mug_on_plate", "instruction": "put the mug on the plate"},
            {"task_id": "open_drawer", "instruction": "open the drawer"},
        ]
    path = write_paraphrases(build_paraphrase_set(instructions), args.out)
    print(f"Wrote paraphrases to {path}")


if __name__ == "__main__":
    main()

