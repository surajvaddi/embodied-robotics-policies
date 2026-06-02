import json

import pytest

from ewpl.data.augmentations import (
    build_paraphrase_set,
    paraphrase_instruction,
    validate_paraphrase_rows,
    write_paraphrases,
)


def test_paraphrase_instruction_preserves_intent() -> None:
    rows = paraphrase_instruction("put the mug on the plate", task_id="mug_on_plate")

    assert len(rows) >= 3
    assert {row["intent"] for row in rows} == {"mug_on_plate"}
    assert "place the mug on the plate" in {row["paraphrase"] for row in rows}


def test_paraphrase_validation_rejects_missing_intent() -> None:
    with pytest.raises(ValueError, match="missing intent"):
        validate_paraphrase_rows(
            [{"task_id": "task", "intent": "", "original": "open drawer", "paraphrase": "open drawer"}]
        )


def test_write_paraphrase_set(tmp_path) -> None:
    rows = build_paraphrase_set(
        [
            {"task_id": "open_drawer", "instruction": "open the drawer"},
            {"task_id": "close_fridge", "instruction": "close the fridge"},
        ]
    )
    path = write_paraphrases(rows, tmp_path / "paraphrases.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload) >= 4
    assert all(row["task_id"] == row["intent"] for row in payload)

