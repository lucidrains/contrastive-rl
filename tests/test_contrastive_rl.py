import pytest
param = pytest.mark.parametrize

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

@param('var_traj_len', (False, True))
@param('repetition_factor', (1, 2))
def test_contrast_trainer(
    var_traj_len,
    repetition_factor
):
    from contrastive_rl_pytorch.contrastive_rl import ContrastiveRLTrainer
    from x_mlps_pytorch import MLP

    encoder = MLP(16, 256, 128)

    trainer = ContrastiveRLTrainer(
        encoder,
        cpu = True,
        repetition_factor = repetition_factor
    )

    trajectories = torch.randn(256, 512, 16)

    trainer(trajectories, 2, lens = torch.randint(256, 512, (256,)) if var_traj_len else None)

def test_traditional_crl():
    import torch.nn.functional as F
    from contrastive_rl_pytorch import ContrastiveRLTrainer
    from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP

    encoder = ResidualNormedMLP(dim = 10, dim_in = 16 + 4, dim_out = 128)
    goal_encoder = ResidualNormedMLP(dim = 10, dim_in = 16, dim_out = 128)

    trainer = ContrastiveRLTrainer(encoder, goal_encoder, cpu = True)

    trajectories = torch.randn(256, 512, 16)
    actions = F.one_hot(torch.randint(0, 4, (256, 512)), 4)

    trainer(trajectories, 100, lens = torch.randint(384, 512, (256,)), actions = actions)

    torch.save(encoder.state_dict(), './trained.pt')

def test_train_policy():
    import torch.nn.functional as F
    from contrastive_rl_pytorch import ContrastiveRLTrainer, ActorTrainer

    from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP

    actor = ResidualNormedMLP(dim = 10, dim_in = 16, dim_out = 4)
    encoder = ResidualNormedMLP(dim = 10, dim_in = 16 + 4, dim_out = 128)
    goal_encoder = ResidualNormedMLP(dim = 10, dim_in = 16, dim_out = 128)

    actor_trainer = ActorTrainer(actor, encoder, goal_encoder, cpu = True)

    trajectories = torch.randn(256, 512, 16)

    lens = torch.randint(384, 512, (256,))
    goal = torch.randn(16)

    actor_trainer(trajectories, goal, 16, lens = lens)

    torch.save(actor.state_dict(), './trained-actor.pt')
