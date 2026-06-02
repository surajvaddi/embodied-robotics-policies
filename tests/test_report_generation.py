from scripts.generate_report import generate_dataset_audit
from tests.test_storage_roundtrip import make_episode
from ewpl.data.storage import CanonicalEpisodeStorage


def test_generate_dataset_audit_report(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    storage = CanonicalEpisodeStorage(dataset_root)
    storage.save_episode(make_episode("ep0"))
    config = tmp_path / "dataset_audit.yaml"
    config.write_text(
        "\n".join(
            [
                "report_type: dataset_audit",
                "title: Test Dataset Audit",
                f"dataset: {dataset_root}",
                f"statistics_out: {tmp_path / 'stats.json'}",
                f"figures_dir: {tmp_path / 'figures'}",
                f"tables_dir: {tmp_path / 'tables'}",
                f"out: {tmp_path / 'dataset_audit.tex'}",
                "known_limitations:",
                "  - tiny test dataset",
            ]
        ),
        encoding="utf-8",
    )

    path = generate_dataset_audit(config)

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Test Dataset Audit" in text
    assert "tiny test dataset" in text
    assert (tmp_path / "figures" / "dataset_task_distribution.png").exists()
    assert (tmp_path / "tables" / "dataset_summary.csv").exists()

