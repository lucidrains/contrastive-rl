import pytest
param = pytest.mark.parametrize

import torch

def test_contrast_loss():
    from contrastive_rl_pytorch.contrastive_rl import ContrastiveLearning
    embeds1 = torch.randn(10, 512)
    embeds2 = torch.randn(10, 512)
    contrastive_learn = ContrastiveLearning()
    loss = contrastive_learn(embeds1, embeds2)
    assert loss.numel() == 1

def test_contrast_wrapper():
    from contrastive_rl_pytorch.contrastive_rl import ContrastiveWrapper, ContrastiveLearning

    from x_mlps_pytorch import MLP
    encoder = MLP(16, 256, 128)

    past_obs = torch.randn(10, 16)
    future_obs = torch.randn(10, 16)

    wrapper = ContrastiveWrapper(encoder, ContrastiveLearning())

    loss = wrapper(past_obs, future_obs)
    assert loss.numel() == 1

@param('var_traj_len', (False, True))
@param('repetition_factor', (1, 2))
@param('use_sigmoid', (False, True))
def test_contrast_trainer(
    var_traj_len,
    repetition_factor,
    use_sigmoid
):
    from contrastive_rl_pytorch.contrastive_rl import ContrastiveRLTrainer, ContrastiveLearning, SigmoidContrastiveLearning
    from x_mlps_pytorch import MLP

    encoder = MLP(16, 256, 128)

    if use_sigmoid:
        contrastive_learn = SigmoidContrastiveLearning()
    else:
        contrastive_learn = ContrastiveLearning()

    trainer = ContrastiveRLTrainer(
        encoder,
        cpu = True,
        repetition_factor = repetition_factor,
        contrastive_learn = contrastive_learn
    )

    trajectories = torch.randn(256, 512, 16)

    trainer(trajectories, 2, lens = torch.randint(256, 512, (256,)) if var_traj_len else None)

@param('use_sigmoid', (False, True))
def test_traditional_crl(use_sigmoid):
    import torch.nn.functional as F
    from contrastive_rl_pytorch import ContrastiveRLTrainer, ContrastiveLearning, SigmoidContrastiveLearning
    from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP

    encoder = ResidualNormedMLP(dim = 10, dim_in = 16 + 4, dim_out = 128, keel_post_ln = True)
    goal_encoder = ResidualNormedMLP(dim = 10, dim_in = 16, dim_out = 128, keel_post_ln = True)

    if use_sigmoid:
        contrastive_learn = SigmoidContrastiveLearning()
    else:
        contrastive_learn = ContrastiveLearning()

    trainer = ContrastiveRLTrainer(encoder, goal_encoder, cpu = True, contrastive_learn = contrastive_learn)

    trajectories = torch.randn(256, 512, 16)
    actions = F.one_hot(torch.randint(0, 4, (256, 512)), 4)

    trainer(trajectories, 100, lens = torch.randint(384, 512, (256,)), actions = actions)

    torch.save(encoder.state_dict(), './trained.pt')

@param('use_sigmoid', (False, True))
def test_train_policy(use_sigmoid):
    import torch.nn.functional as F
    from contrastive_rl_pytorch import ContrastiveRLTrainer, ActorTrainer, ContrastiveLearning, SigmoidContrastiveLearning

    from x_mlps_pytorch.residual_normed_mlp import ResidualNormedMLP

    actor = ResidualNormedMLP(dim = 10, dim_in = 16 * 2, dim_out = 4, keel_post_ln = True)
    encoder = ResidualNormedMLP(dim = 10, dim_in = 16 + 4, dim_out = 128, keel_post_ln = True)
    goal_encoder = ResidualNormedMLP(dim = 10, dim_in = 16, dim_out = 128, keel_post_ln = True)

    if use_sigmoid:
        contrastive_learn = SigmoidContrastiveLearning()
    else:
        contrastive_learn = ContrastiveLearning()

    actor_trainer = ActorTrainer(actor, encoder, goal_encoder, cpu = True, contrastive_learn = contrastive_learn)

    trajectories = torch.randn(256, 512, 16)

    lens = torch.randint(384, 512, (256,))

    actor_trainer(trajectories, 16, lens = lens)

    torch.save(actor.state_dict(), './trained-actor.pt')

def test_readme():
    import torch
    from contrastive_rl_pytorch import ContrastiveRLTrainer
    from x_mlps_pytorch import ResidualNormedMLP

    encoder = ResidualNormedMLP(dim = 256, dim_in = 16, dim_out = 128, keel_post_ln = True)

    trainer = ContrastiveRLTrainer(encoder)

    trajectories = torch.randn(256, 512, 16)

    trainer(trajectories, 1)

@param('num_discrete_actions', (2, 4))
@param('normalize_q_values', (False, True))
@param('use_target_goals', ('none', '1d', '2d'))
@param('use_sigmoid', (False, True))
def test_discrete_actor_trainer(
    num_discrete_actions,
    normalize_q_values,
    use_target_goals,
    use_sigmoid
):
    from contrastive_rl_pytorch import ActorTrainer, ContrastiveLearning, SigmoidContrastiveLearning
    from x_mlps_pytorch import MLP

    dim_state = 16
    dim_goal = 16
    dim_action = num_discrete_actions
    dim_contrastive_embed = 64

    actor = MLP(dim_state + dim_goal, 64, 64, dim_action)
    encoder = MLP(dim_state + dim_action, 64, 64, dim_contrastive_embed)
    goal_encoder = MLP(dim_goal, 64, 64, dim_contrastive_embed)

    contrastive_learn = SigmoidContrastiveLearning() if use_sigmoid else ContrastiveLearning()

    actor_trainer = ActorTrainer(
        actor,
        encoder,
        goal_encoder,
        num_discrete_actions = num_discrete_actions,
        normalize_q_values = normalize_q_values,
        cpu = True,
        contrastive_learn = contrastive_learn
    )

    trajectories = torch.randn(32, 64, dim_state)
    lens = torch.randint(32, 64, (32,))

    target_goals = None
    if use_target_goals == '1d':
        target_goals = torch.randn(dim_goal)
    elif use_target_goals == '2d':
        target_goals = torch.randn(8, dim_goal)

    loss = actor_trainer(trajectories, 2, lens = lens, target_goals = target_goals, pbar = False)
    assert isinstance(loss, float) and not torch.tensor(loss).isnan()

@param('use_target_goals', (False, True))
@param('use_sigmoid', (False, True))
def test_continuous_actor_trainer(use_target_goals, use_sigmoid):
    from contrastive_rl_pytorch import ActorTrainer, ContrastiveLearning, SigmoidContrastiveLearning
    from x_mlps_pytorch import MLP

    dim_state = 16
    dim_goal = 16
    dim_action = 2
    dim_contrastive_embed = 64

    actor = MLP(dim_state + dim_goal, 64, 64, dim_action)
    encoder = MLP(dim_state + dim_action, 64, 64, dim_contrastive_embed)
    goal_encoder = MLP(dim_goal, 64, 64, dim_contrastive_embed)

    contrastive_learn = SigmoidContrastiveLearning() if use_sigmoid else ContrastiveLearning()

    actor_trainer = ActorTrainer(
        actor,
        encoder,
        goal_encoder,
        cpu = True,
        contrastive_learn = contrastive_learn
    )

    trajectories = torch.randn(32, 64, dim_state)
    target_goals = torch.randn(dim_goal) if use_target_goals else None

    loss = actor_trainer(
        trajectories,
        2,
        target_goals = target_goals,
        sample_fn = lambda logits: torch.tanh(logits),
        pbar = False
    )
    assert isinstance(loss, float) and not torch.tensor(loss).isnan()

@param('use_sigmoid', (False, True))
@param('l2norm_embed', (False, True))
def test_euclidean_contrast_learning(use_sigmoid, l2norm_embed):
    from contrastive_rl_pytorch import ContrastiveLearning, SigmoidContrastiveLearning
    cls = SigmoidContrastiveLearning if use_sigmoid else ContrastiveLearning
    cl = cls(use_euclidean = True, l2norm_embed = l2norm_embed)

    embeds1 = torch.randn(10, 32)
    embeds2 = torch.randn(10, 32)

    loss = cl(embeds1, embeds2)
    assert loss.numel() == 1 and not loss.isnan()

    scores = cl(embeds1, embeds2, return_contrastive_score = True)
    assert scores.shape == (10,) and not scores.isnan().any()

    # batched score (e.g. discrete actions)
    e1 = torch.randn(10, 4, 32)
    e2 = torch.randn(10, 4, 32)
    scores_batched = cl(e1, e2, return_contrastive_score = True)
    assert scores_batched.shape == (10, 4) and not scores_batched.isnan().any()

@param('use_sigmoid', (False, True))
def test_euclidean_discrete_actor_trainer(use_sigmoid):
    from contrastive_rl_pytorch import ActorTrainer, ContrastiveLearning, SigmoidContrastiveLearning
    from x_mlps_pytorch import MLP

    dim_state = 8
    dim_goal = 8
    dim_action = 4
    dim_embed = 32

    actor = MLP(dim_state + dim_goal, 32, dim_action)
    encoder = MLP(dim_state + dim_action, 32, dim_embed)
    goal_encoder = MLP(dim_goal, 32, dim_embed)

    cls = SigmoidContrastiveLearning if use_sigmoid else ContrastiveLearning
    cl = cls(use_euclidean = True)

    actor_trainer = ActorTrainer(
        actor,
        encoder,
        goal_encoder,
        num_discrete_actions = dim_action,
        cpu = True,
        contrastive_learn = cl
    )

    trajectories = torch.randn(16, 32, dim_state)
    target_goals = torch.randn(dim_goal)

    loss = actor_trainer(trajectories, 2, target_goals = target_goals, pbar = False)
    assert isinstance(loss, float) and not torch.tensor(loss).isnan()

@param('discount', (0.9, 0.99, 1.0))
@param('as_tensor', (False, True))
def test_sample_truncated_geometric(discount, as_tensor):
    from contrastive_rl_pytorch import sample_truncated_geometric, sample_truncated_geometric_time
    assert sample_truncated_geometric is sample_truncated_geometric_time

    max_steps = torch.tensor([5, 10, 50, 100]) if as_tensor else 20
    delta = sample_truncated_geometric(max_steps, discount)

    if as_tensor:
        assert (delta >= 1).all()
        assert (delta <= max_steps).all()
    else:
        assert 1 <= delta.item() <= max_steps

    # test boundary behavior with rand_uniform=0 and rand_uniform=1

    rem = torch.tensor([5, 10, 25, 50])
    rand_0 = torch.zeros_like(rem, dtype = torch.float32)
    rand_1 = torch.ones_like(rem, dtype = torch.float32) * (1. - 1e-7)

    assert (sample_truncated_geometric(rem, discount, rand_uniform = rand_0) == 1).all()
    assert (sample_truncated_geometric(rem, discount, rand_uniform = rand_1) == rem).all()

    # test tensor / batched discount

    tensor_discount = torch.full_like(rem, discount, dtype = torch.float32)
    delta_tensor = sample_truncated_geometric(rem, tensor_discount)
    assert (delta_tensor >= 1).all() and (delta_tensor <= rem).all()
