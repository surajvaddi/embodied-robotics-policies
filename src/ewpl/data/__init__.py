"""Canonical robotics dataset utilities."""

__all__ = ["Action", "Episode", "Observation", "Step"]


def __getattr__(name):
    if name in __all__:
        from ewpl.data.episode import Action, Episode, Observation, Step

        return {
            "Action": Action,
            "Episode": Episode,
            "Observation": Observation,
            "Step": Step,
        }[name]
    raise AttributeError(name)
