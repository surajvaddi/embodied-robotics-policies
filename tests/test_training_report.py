import csv

from ewpl.reports.training import generate_bc_smoke_report, plot_training_loss, read_training_metrics


def write_metrics(path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss"])
        writer.writeheader()
        writer.writerow({"step": 1, "loss": "0.5"})
        writer.writerow({"step": 2, "loss": "0.25"})
        writer.writerow({"step": 3, "loss": "0.125"})


def test_read_training_metrics_computes_loss_stats(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    write_metrics(metrics_path)

    metrics = read_training_metrics(metrics_path)

    assert metrics["steps"] == 3
    assert metrics["initial_loss"] == 0.5
    assert metrics["final_loss"] == 0.125
    assert metrics["loss_delta"] == -0.375


def test_plot_training_loss_writes_png(tmp_path) -> None:
    path = plot_training_loss({"losses": [0.5, 0.25, 0.125]}, tmp_path / "loss.png")

    assert path.exists()
    assert path.suffix == ".png"


def test_generate_bc_smoke_report_writes_tex_and_plot(tmp_path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    checkpoint_path = tmp_path / "latest.pt"
    write_metrics(metrics_path)
    checkpoint_path.write_bytes(b"checkpoint")

    report = generate_bc_smoke_report(
        metrics_csv=metrics_path,
        checkpoint_path=checkpoint_path,
        out=tmp_path / "report.tex",
        figures_dir=tmp_path / "figures",
    )

    assert report.exists()
    assert (tmp_path / "figures" / "bc_training_loss.png").exists()
    assert "Behavior Cloning Smoke Report" in report.read_text(encoding="utf-8")
