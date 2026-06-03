import numpy as np

from ewpl.data.schemas import Action, Observation, Step
from ewpl.data.tensorization import normalize_image, tensorize_step, tokenize_language


def test_normalize_image_returns_chw_float() -> None:
    rgb = np.full((8, 10, 3), 255, dtype=np.uint8)
    image = normalize_image(rgb, image_size=4)

    assert image.shape == (3, 4, 4)
    assert image.dtype == np.float32
    assert float(image.max()) == 1.0


def test_tokenize_language_is_deterministic_and_padded() -> None:
    a = tokenize_language("pick up cube", length=5, vocab_size=128)
    b = tokenize_language("pick up cube", length=5, vocab_size=128)

    assert np.array_equal(a, b)
    assert a.shape == (5,)
    assert a[-1] == 0


def test_tensorize_step_shapes() -> None:
    step = Step(
        t=0,
        observation=Observation(
            rgb=np.zeros((6, 6, 3), dtype=np.uint8),
            depth=None,
            proprio=np.ones(7, dtype=np.float32),
            language="move the cube",
        ),
        action=Action(vector=np.zeros(4, dtype=np.float32), convention="test"),
        reward=None,
        done=True,
        info={},
    )

    sample = tensorize_step(step, image_size=8, language_length=6)

    assert sample.image.shape == (3, 8, 8)
    assert sample.proprio.shape == (7,)
    assert sample.language_ids.shape == (6,)
    assert sample.action.shape == (4,)

