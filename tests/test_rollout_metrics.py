import csv
import json

from ewpl.eval.rollout_metrics import compute_rollout_metrics, write_rollout_metrics


def test_compute_rollout_metrics_from_artifacts(tmp_path) -> None:
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    with (rollout_dir / "rollout_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode",
                "steps",
                "success",
                "termination_reason",
            ],
        )
        writer.writeheader()
        writer.writerow({"episode": 0, "steps": 2, "success": True, "termination_reason": "success"})
        writer.writerow({"episode": 1, "steps": 2, "success": False, "termination_reason": "max_steps"})

    with (rollout_dir / "per_step_logs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["episode", "step", "action_vector", "action_norm", "policy_latency_ms"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "episode": 0,
                "step": 0,
                "action_vector": json.dumps([0.0, 0.0]),
                "action_norm": 0.0,
                "policy_latency_ms": 1.0,
            }
        )
        writer.writerow(
            {
                "episode": 0,
                "step": 1,
                "action_vector": json.dumps([3.0, 4.0]),
                "action_norm": 5.0,
                "policy_latency_ms": 3.0,
            }
        )

    metrics = compute_rollout_metrics(rollout_dir)
    out = write_rollout_metrics(metrics, tmp_path / "metrics.csv")

    assert metrics["episodes"] == 2
    assert metrics["success_rate"] == 0.5
    assert metrics["average_episode_length"] == 2.0
    assert metrics["action_smoothness"] == 5.0
    assert metrics["failure_modes"] == {"max_steps": 1}
    assert out.exists()

