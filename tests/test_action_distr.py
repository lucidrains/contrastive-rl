from __future__ import annotations

import pytest
import torch

from torch.distributions import Normal

param = pytest.mark.parametrize

# action adapters under test

def make_adapter(kind, action_repr):
    from contrastive_rl_pytorch import ActionAdapter, CategoricalActionAdapter, MeanConcBetaActionAdapter

    if kind == 'categorical':
        return CategoricalActionAdapter(4, action_repr = action_repr)

    if kind == 'mean_conc_beta':
        pytest.importorskip('mean_conc_beta')

        from mean_conc_beta import Beta
        return MeanConcBetaActionAdapter(
            Beta(pos_fn = 'softplus', max_unimodal_floor = 50., squash_fn = 'leaky_tanh'),
            dim_action = 2,
            action_repr = action_repr
        )

    def normal_distr(params):
        return Normal(params[..., 0], params[..., 1].exp())

    return ActionAdapter(
        dim = 2,
        distr_fn = normal_distr,
        to_critic_fn = lambda params: torch.stack((params[..., 0], params[..., 1].exp()), dim = -1),
        entropy_fn = lambda params: normal_distr(params).entropy()
    )

CASES = (
    ('categorical', 'raw_logits'),
    ('categorical', 'softmax_probs'),
    ('categorical', 'sampled'),
    ('mean_conc_beta', 'sampled'),
    ('mean_conc_beta', 'raw_params'),
    ('mean_conc_beta', 'transformed'),
    ('custom_normal', 'transformed'),
)

@param('kind,action_repr', CASES)
def test_action_adapter_actor_trainer(kind, action_repr):
    from contrastive_rl_pytorch import ActorTrainer, ContrastiveLearning
    from x_mlps_pytorch import MLP

    adapter = make_adapter(kind, action_repr)

    dim_state = dim_goal = 16

    actor = MLP(dim_state + dim_goal, 32, adapter.actor_dim)
    encoder = MLP(dim_state + adapter.dim, 32, 32)
    goal_encoder = MLP(dim_goal, 32, 32)

    trainer = ActorTrainer(
        actor,
        encoder,
        goal_encoder,
        cpu = True,
        contrastive_learn = ContrastiveLearning(),
        action_adapter = adapter,
        action_entropy_loss_weight = 0.005
    )

    trajectories = torch.randn(32, 32, dim_state)
    loss = trainer(trajectories, 2, target_goals = torch.randn(dim_goal), pbar = False)

    assert isinstance(loss, float) and not torch.tensor(loss).isnan()

@param('kind,action_repr', CASES)
def test_action_adapter_to_critic(kind, action_repr):
    adapter = make_adapter(kind, action_repr)

    params = torch.randn(4, adapter.actor_dim, requires_grad = True)
    critic_action = adapter.to_critic(params)

    assert critic_action.shape == (4, adapter.dim)

    critic_action.sum().backward()
    assert torch.isfinite(params.grad).all()

def test_action_adapter_differentiable_sample_requires_rsample():
    from contrastive_rl_pytorch import ActionAdapter
    from torch.distributions import Bernoulli

    adapter = ActionAdapter(dim = 1, distr_fn = lambda params: Bernoulli(logits = params))
    params = torch.randn(2)

    assert adapter.sample(params).shape == (2,)

    with pytest.raises(AssertionError):
        adapter.sample(params, differentiable = True)
