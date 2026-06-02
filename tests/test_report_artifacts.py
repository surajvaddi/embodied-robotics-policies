from ewpl.reports.plots import generate_dataset_plots, plot_bar_counts
from ewpl.reports.tables import dataset_summary_rows, generate_dataset_tables


def sample_stats():
    return {
        "episodes": 2,
        "total_steps": 5,
        "sources": {"synthetic": 2},
        "tasks": {"pick_cube": 2},
        "action_dims": {"2": 5},
        "steps_per_episode_summary": {"mean": 2.5},
        "action_l2_summary": {"mean": 0.25},
        "language_length_summary": {"mean": 3.0},
    }


def test_plot_bar_counts_writes_png(tmp_path) -> None:
    path = plot_bar_counts({"a": 2, "b": 1}, tmp_path / "plot.png", title="Counts")

    assert path.exists()
    assert path.suffix == ".png"


def test_generate_dataset_tables(tmp_path) -> None:
    rows = dataset_summary_rows(sample_stats())
    paths = generate_dataset_tables(sample_stats(), tmp_path)

    assert rows[0] == {"metric": "episodes", "value": "2"}
    assert (tmp_path / "dataset_summary.csv").exists()
    assert (tmp_path / "dataset_summary.tex").exists()
    assert set(paths) == {"summary_csv", "summary_tex"}


def test_generate_dataset_plots(tmp_path) -> None:
    paths = generate_dataset_plots(sample_stats(), tmp_path)

    assert set(paths) == {
        "task_distribution",
        "source_distribution",
        "action_dim_distribution",
    }
    assert (tmp_path / "dataset_task_distribution.png").exists()

