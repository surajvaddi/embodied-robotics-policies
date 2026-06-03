"""Training utilities."""

__all__ = ["BCTrainingConfig", "TrainingResult", "train_bc"]


def __getattr__(name: str):
    if name in __all__:
        from ewpl.training.bc import BCTrainingConfig, TrainingResult, train_bc

        exports = {
            "BCTrainingConfig": BCTrainingConfig,
            "TrainingResult": TrainingResult,
            "train_bc": train_bc,
        }
        return exports[name]
    raise AttributeError(name)
