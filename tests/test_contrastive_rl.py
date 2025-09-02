import pytest

import torch

def test_contrast_loss():
    from contrastive_rl_pytorch.contrastive_rl import contrastive_loss

    embeds1 = torch.randn(10, 512)
    embeds2 = torch.randn(10, 512)

    loss = contrastive_loss(embeds1, embeds2)
    assert loss.numel() == 1

def test_contrast_wrapper():
    from contrastive_rl_pytorch.contrastive_rl import ContrastiveWrapper

    from x_mlps_pytorch import MLP
    encoder = MLP(16, 256, 128)

    past_obs = torch.randn(10, 16)
    future_obs = torch.randn(10, 16)

    wrapper = ContrastiveWrapper(encoder)

    loss = wrapper(past_obs, future_obs)
    assert loss.numel() == 1

def test_contrast_trainer():
    from contrastive_rl_pytorch.contrastive_rl import ContrastiveRLTrainer
    from x_mlps_pytorch import MLP

    encoder = MLP(16, 256, 128)

    trainer = ContrastiveRLTrainer(
        encoder
    )

    trajectories = torch.randn(256, 512, 16)

    trainer(trajectories, 2)
