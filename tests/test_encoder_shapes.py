import pytest

torch = pytest.importorskip("torch")

from ewpl.models.encoders.fusion import FusionEncoder
from ewpl.models.encoders.language import LanguageEncoder
from ewpl.models.encoders.proprio import ProprioEncoder
from ewpl.models.encoders.vision import VisionEncoder


def test_encoder_shapes() -> None:
    vision = VisionEncoder(output_dim=16)
    language = LanguageEncoder(vocab_size=128, embed_dim=8, output_dim=16)
    proprio = ProprioEncoder(input_dim=7, output_dim=16)
    fusion = FusionEncoder(input_dim=48, output_dim=32)

    image_emb = vision(torch.zeros(2, 3, 32, 32))
    lang_emb = language(torch.ones(2, 6, dtype=torch.long))
    prop_emb = proprio(torch.zeros(2, 7))
    fused = fusion(image_emb, lang_emb, prop_emb)

    assert image_emb.shape == (2, 16)
    assert lang_emb.shape == (2, 16)
    assert prop_emb.shape == (2, 16)
    assert fused.shape == (2, 32)

