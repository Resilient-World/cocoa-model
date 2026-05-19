import os

import pytest
import torch

pytestmark = pytest.mark.integration  # requires weight download (~340 MB for base)


@pytest.fixture(scope="module")
def nano_extractor():
    from models.galileo_features import GalileoFeatureConfig, GalileoFeatureExtractor

    if not torch.cuda.is_available() and os.getenv("CI") == "true":
        pytest.skip("Galileo integration test skipped on CPU CI")
    return GalileoFeatureExtractor(GalileoFeatureConfig(size="nano", device="cpu"))


def test_loader_returns_frozen_encoder():
    from models.galileo_loader import load_galileo

    m = load_galileo("nano", device="cpu")
    assert all(not p.requires_grad for p in m.parameters())


def test_embed_returns_expected_shape(nano_extractor):
    t, h, w = 2, 8, 8
    from models.vendor.galileo_data_utils import S2_BANDS

    s2 = torch.randn(t, h, w, len(S2_BANDS))
    emb = nano_extractor.embed(s2=s2, months=torch.tensor([5, 6]))
    assert emb.ndim == 2
    assert emb.shape[-1] > 0  # embedding dim
    assert emb.shape[0] > 0  # token count


def test_embed_is_deterministic(nano_extractor):
    torch.manual_seed(0)
    from models.vendor.galileo_data_utils import S2_BANDS

    s2 = torch.randn(2, 8, 8, len(S2_BANDS))
    a = nano_extractor.embed(s2=s2, months=torch.tensor([5, 6]))
    b = nano_extractor.embed(s2=s2, months=torch.tensor([5, 6]))
    assert torch.allclose(a, b, atol=1e-5)
