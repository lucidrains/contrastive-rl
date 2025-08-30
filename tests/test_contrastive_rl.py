import pytest

def test_contrast_rl():
    import torch
    from contrastive_rl.contrastive_rl import contrastive_loss

    embeds1 = torch.randn(10, 512)
    embeds2 = torch.randn(10, 512)

    loss = contrastive_loss(embeds1, embeds2)
    assert loss.numel() == 1
