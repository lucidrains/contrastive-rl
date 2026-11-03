import pytest
param = pytest.mark.parametrize

import torch

def exists(v):
    return v is not None

def test_contrast_loss():
    from contrastive_rl_pytorch.contrastive_rl import ContrastiveLearning
    embeds1 = torch.randn(10, 512)
    embeds2 = torch.randn(10, 512)
    contrastive_learn = ContrastiveLearning()
    loss = contrastive_learn(embeds1, embeds2)
    assert loss.numel() == 1

def test_contrast_loss_lse_reg():
    from contrastive_rl_pytorch.contrastive_rl import ContrastiveLearning

    with pytest.raises(AssertionError):
        ContrastiveLearning(symmetric = True, l2norm_embed = False, lse_reg_weight = 0.01)

    with pytest.raises(AssertionError):
        ContrastiveLearning(symmetric = False, l2norm_embed = True, lse_reg_weight = 0.01)

    learn = ContrastiveLearning(symmetric = False, l2norm_embed = False, lse_reg_weight = 0.01)
    loss = learn(torch.randn(10, 128), torch.randn(10, 128))

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
@param('orthogonal_transition', (False, True))
def test_contrast_trainer(
    var_traj_len,
    repetition_factor,
    use_sigmoid,
    orthogonal_transition
):
    from contrastive_rl_pytorch.contrastive_rl import ContrastiveRLTrainer, ContrastiveLearning, SigmoidContrastiveLearning
    from x_mlps_pytorch import MLP

    encoder = MLP(16, 256, 128)
    goal_encoder = MLP(16, 256, 128)

    if orthogonal_transition:
        contrastive_learn = ContrastiveLearning(rotation_dim = 128)
    elif use_sigmoid:
        contrastive_learn = SigmoidContrastiveLearning()
    else:
        contrastive_learn = ContrastiveLearning()

    trainer = ContrastiveRLTrainer(
        encoder,
        goal_encoder,
        cpu = True,
        repetition_factor = repetition_factor,
        contrastive_learn = contrastive_learn
    )

    trajectories = torch.randn(256, 512, 16)

    trainer(trajectories, 2, lens = torch.randint(256, 512, (256,)) if var_traj_len else None)

@param('var_traj_len', (False, True))
@param('repetition_factor', (1, 2))
@param('goal_discount', (None, 0.95))
def test_temporal_distance_critic(var_traj_len, repetition_factor, goal_discount):
    from contrastive_rl_pytorch import ActorTrainer, ContrastiveRLTrainer, TemporalDistanceLearning
    from x_mlps_pytorch import MLP

    dim_state, dim_goal, dim_action, dim_embed = 16, 16, 2, 64

    encoder = MLP(dim_state + dim_action, 64, dim_embed)
    goal_encoder = MLP(dim_goal, 64, dim_embed)
    actor = MLP(dim_state + dim_goal, 64, dim_action)

    contrastive_learn = TemporalDistanceLearning()

    critic_trainer = ContrastiveRLTrainer(
        encoder,
        goal_encoder,
        batch_size = 32,
        repetition_factor = repetition_factor,
        discount = 0.999,
        goal_discount = goal_discount,
        cpu = True,
        contrastive_learn = contrastive_learn
    )

    trajectories = torch.randn(32, 64, dim_state)
    actions = torch.nn.functional.one_hot(torch.randint(0, dim_action, (32, 64)), dim_action)
    lens = torch.randint(32, 64, (32,)) if var_traj_len else None

    critic_loss = critic_trainer(trajectories, 2, lens = lens, actions = actions, pbar = False)
    assert torch.isfinite(torch.tensor(critic_loss))

    actor_trainer = ActorTrainer(
        actor,
        encoder,
        goal_encoder,
        batch_size = 16,
        num_discrete_actions = dim_action,
        cpu = True,
        contrastive_learn = contrastive_learn
    )

    actor_loss = actor_trainer(trajectories, 2, lens = lens, target_goals = torch.zeros(dim_goal), pbar = False)
    assert torch.isfinite(torch.tensor(actor_loss))

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
    from contrastive_rl_pytorch import ActorTrainer, ContrastiveLearning, SigmoidContrastiveLearning

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
    from x_mlps_pytorch import AttnResidualNormedMLP

    encoder = AttnResidualNormedMLP(dim = 256, dim_in = 16, dim_out = 128, depth = 4, use_rmsnorm = True)

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

def test_sample_random_state_ignores_padding():
    from contrastive_rl_pytorch.contrastive_rl import sample_random_state

    class FakeBuffer:
        num_episodes = 2

        def get_all_data(self, fields = None, meta_fields = None):
            return dict(
                state = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [0., 0.]]]),
                episode_lens = torch.tensor([2, 1])
            )

    for _ in range(64):
        state = sample_random_state(FakeBuffer(), None, 1.)
        assert state.shape == (2,)
        assert not torch.allclose(state, torch.zeros(2))

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

def test_discount_conditioned_e2e():
    from contrastive_rl_pytorch import (
        ContrastiveRLTrainer,
        ActorTrainer,
        default_discount_transform
    )
    from x_mlps_pytorch import MLP

    dim_state = 8
    dim_goal = 8
    dim_action = 4
    dim_discount = 3

    critic = MLP(dim_state + dim_action + dim_discount, 64, 32)
    goal_encoder = MLP(dim_goal, 64, 32)
    actor = MLP(dim_state + dim_goal + dim_discount, 64, dim_action)

    critic_trainer = ContrastiveRLTrainer(
        critic,
        goal_encoder,
        discount = (0.85, 0.999),
        discount_condition = True,
        discount_transform = default_discount_transform,
        cpu = True
    )

    actor_trainer = ActorTrainer(
        actor,
        critic,
        goal_encoder,
        discount = (0.85, 0.999),
        discount_condition = True,
        discount_transform = default_discount_transform,
        num_discrete_actions = dim_action,
        cpu = True
    )

    trajectories = torch.randn(16, 32, dim_state)
    actions = torch.randn(16, 32, dim_action)

    c_loss = critic_trainer(trajectories, 2, actions = actions, pbar = False)
    a_loss = actor_trainer(trajectories, 2, target_goals = torch.randn(dim_goal), pbar = False)

    assert isinstance(c_loss, float) and not torch.tensor(c_loss).isnan()
    assert isinstance(a_loss, float) and not torch.tensor(a_loss).isnan()

    state = torch.randn(1, dim_state)
    goal = torch.randn(1, dim_goal)
    discount = default_discount_transform(0.999)

    action = actor(torch.cat((state, goal, discount), dim = -1)).argmax(dim = -1)
    assert action.shape == (1,)

def test_hyperspherical_contrastive():
    from torch_einops_utils.shape import shape
    from x_mlps_pytorch import MLP

    from contrastive_rl_pytorch import ContrastiveLearning, ContrastiveWrapper

    dim, batch = 16, 10

    contrastive_learn = ContrastiveLearning(rotation_dim = dim)
    wrapper = ContrastiveWrapper(
        MLP(dim + 2, 64, dim),
        contrastive_learn,
        future_encoder = MLP(dim, 64, dim)
    )

    loss = wrapper(torch.randn(batch, dim), torch.randn(batch, dim), torch.randn(batch, 2))
    loss.backward()

    rotation = contrastive_learn.rotation.detach()  # d d
    d, = shape(rotation, '[d] _')

    assert torch.allclose(rotation.T @ rotation, torch.eye(d), atol = 1e-5)
    assert exists(contrastive_learn.rotation_param.grad)

def test_orthogonal_transition_matches_c_slerp():
    import torch.nn.functional as F
    from contrastive_rl_pytorch import ContrastiveLearning

    dim, batch = 8, 7

    learn = ContrastiveLearning(rotation_dim = dim, scale = 1.)

    with torch.no_grad():
        learn.rotation_param.copy_(torch.randn(dim, dim))

    embeds1 = torch.randn(batch, dim)
    embeds2 = torch.randn(batch, dim)

    e1 = F.normalize(embeds1, dim = -1)
    e2 = F.normalize(embeds2, dim = -1)
    R = learn.rotation

    # rotated pairwise logits
    logits = e1 @ R @ e2.T

    labels = torch.arange(batch)
    expected = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) * 0.5

    assert torch.allclose(learn(embeds1, embeds2), expected, atol = 1e-6)
    assert torch.allclose(learn(embeds1, embeds2, return_contrastive_score = True), logits.diagonal(), atol = 1e-6)
    assert torch.allclose(R.T @ R, torch.eye(dim), atol = 1e-5)

def test_orthogonal_transition_critic():
    from contrastive_rl_pytorch import ContrastiveLearning, ContrastiveRLTrainer
    from x_mlps_pytorch import MLP

    dim_state, dim_goal, dim_action, dim_embed = 8, 2, 2, 16

    encoder = MLP(dim_state + dim_action, 64, dim_embed)
    goal_encoder = MLP(dim_goal, 64, dim_embed)

    trainer = ContrastiveRLTrainer(
        encoder,
        goal_encoder,
        batch_size = 8,
        repetition_factor = 1,
        state_to_goal_fn = lambda t: t[..., :dim_goal],
        contrastive_learn = ContrastiveLearning(rotation_dim = dim_embed),
        cpu = True
    )

    trajectories = torch.randn(16, 32, dim_state)
    actions = torch.randn(16, 32, dim_action)

    loss = trainer(trajectories, 2, actions = actions, pbar = False)
    assert torch.isfinite(torch.tensor(loss))

def test_temporal_distance_valid_combinations():
    from contrastive_rl_pytorch import SigmoidContrastiveLearning, TemporalDistanceLearning

    assert issubclass(TemporalDistanceLearning, SigmoidContrastiveLearning)

    with pytest.raises(AssertionError):
        # squared-chord temporal distance needs normalized embeddings
        SigmoidContrastiveLearning(
            use_euclidean = False,
            l2norm_embed = False,
            time_weight = 1.
        )

    batch, dim = 8, 16
    embeds1 = torch.randn(batch, dim)
    embeds2 = torch.randn(batch, dim)
    delta_times = torch.randint(1, 8, (batch, 1))
    raw_discount = torch.full((batch, 1), 0.9)

    for use_euclidean, l2norm_embed in ((True, False), (True, True), (False, True)):
        learn = SigmoidContrastiveLearning(
            use_euclidean = use_euclidean,
            l2norm_embed = l2norm_embed,
            time_weight = 1.,
            sep_weight = 1.
        )

        loss = learn(
            embeds1,
            embeds2,
            delta_times = delta_times,
            raw_discount = raw_discount,
            goal_states = embeds2
        )

        assert torch.isfinite(loss)

def test_temporal_distance_changes_loss():
    from contrastive_rl_pytorch import SigmoidContrastiveLearning

    batch, dim = 8, 16
    embeds1 = torch.randn(batch, dim)
    embeds2 = torch.randn(batch, dim)
    delta_times = torch.randint(1, 8, (batch, 1))
    raw_discount = torch.full((batch, 1), 0.9)

    base = SigmoidContrastiveLearning(use_euclidean = True, l2norm_embed = False)
    temporal = SigmoidContrastiveLearning(
        use_euclidean = True,
        l2norm_embed = False,
        time_weight = 1.,
        sep_weight = 1.
    )

    base_loss = base(embeds1, embeds2)
    temporal_loss = temporal(
        embeds1,
        embeds2,
        delta_times = delta_times,
        raw_discount = raw_discount,
        goal_states = embeds2
    )

    assert not torch.allclose(base_loss, temporal_loss)

def test_orthogonal_transition_with_temporal():
    from contrastive_rl_pytorch import ContrastiveLearning

    dim, batch = 8, 7
    learn = ContrastiveLearning(rotation_dim = dim, time_weight = 1., sep_weight = 1.)

    embeds1 = torch.randn(batch, dim)
    embeds2 = torch.randn(batch, dim)
    delta_times = torch.randint(1, 8, (batch, 1))
    raw_discount = torch.full((batch, 1), 0.9)

    loss = learn(
        embeds1,
        embeds2,
        delta_times = delta_times,
        raw_discount = raw_discount,
        goal_states = embeds2
    )

    assert torch.isfinite(loss)

def test_sigmoid_orthogonal_transition_with_temporal():
    from contrastive_rl_pytorch import SigmoidContrastiveLearning

    dim, batch = 8, 7
    learn = SigmoidContrastiveLearning(
        l2norm_embed = True,
        rotation_dim = dim,
        time_weight = 1.,
        sep_weight = 1.
    )

    embeds1 = torch.randn(batch, dim)
    embeds2 = torch.randn(batch, dim)
    delta_times = torch.randint(1, 8, (batch, 1))
    raw_discount = torch.full((batch, 1), 0.9)

    loss = learn(
        embeds1,
        embeds2,
        delta_times = delta_times,
        raw_discount = raw_discount,
        goal_states = embeds2
    )

    assert torch.isfinite(loss)
    assert torch.allclose(learn.rotation.T @ learn.rotation, torch.eye(dim), atol = 1e-5)

def test_temporal_distance_types():
    from contrastive_rl_pytorch import SigmoidContrastiveLearning

    batch, dim = 8, 16
    embeds1 = torch.randn(batch, dim)
    embeds2 = torch.randn(batch, dim)
    delta_times = torch.randint(1, 8, (batch, 1))
    raw_discount = torch.full((batch, 1), 0.9)

    for distance_type in ('euclidean', 'chord', 'squared_chord'):
        learn = SigmoidContrastiveLearning(
            l2norm_embed = True,
            time_weight = 1.,
            sep_weight = 1.,
            temporal_distance_type = distance_type
        )

        loss = learn(
            embeds1,
            embeds2,
            delta_times = delta_times,
            raw_discount = raw_discount,
            goal_states = embeds2
        )

        assert torch.isfinite(loss)

def test_chord_and_squared_chord_distance():
    from contrastive_rl_pytorch.contrastive_rl import calc_distance

    a = torch.tensor([1., 0., 0., 0.])
    b = torch.tensor([0., 1., 0., 0.])
    c = -a

    assert torch.allclose(calc_distance(a, a), torch.tensor(0.), atol = 1e-6)
    assert torch.allclose(calc_distance(a, b), torch.tensor(2.), atol = 1e-6)
    assert torch.allclose(calc_distance(a, c), torch.tensor(4.), atol = 1e-6)

    assert torch.allclose(calc_distance(a, a, distance_type = 'chord'), torch.tensor(0.), atol = 1e-6)
    assert torch.allclose(calc_distance(a, b, distance_type = 'chord'), torch.tensor(2.).sqrt(), atol = 1e-6)
    assert torch.allclose(calc_distance(a, c, distance_type = 'chord'), torch.tensor(2.), atol = 1e-6)

def test_c_slerp_plan_e2e():
    import torch.nn.functional as F
    from contrastive_rl_pytorch import CSlerpPlan, c_slerp_plan

    dim = 8
    state = F.normalize(torch.randn(dim), dim = -1)
    goal = F.normalize(torch.randn(dim), dim = -1)

    # orthogonal state encoder / decoder

    rotation = torch.linalg.qr(torch.randn(dim, dim)).Q
    encode = lambda x: rotation @ x
    decode_state = lambda embed: rotation.T @ embed

    state_embed, goal_embed = encode(state), encode(goal)

    # deterministic plan, identity rotation

    plan = c_slerp_plan(state_embed, goal_embed, num_waypoints = 2, alpha = 0.)
    assert isinstance(plan, CSlerpPlan)

    # decode plan fields

    waypoints, num_waypoints, selected_index, subgoal = plan
    assert num_waypoints == 2
    assert waypoints.shape == (3, dim)
    assert torch.allclose(waypoints.norm(dim = -1), torch.ones(3), atol = 1e-5)
    assert selected_index is None and subgoal is None

    # waypoints progress toward goal, last one is goal

    cos = waypoints @ goal_embed
    assert torch.all(cos[1:] > cos[:-1])
    assert torch.allclose(waypoints[-1], goal_embed, atol = 1e-5)

    # decode state from last waypoint

    assert torch.allclose(decode_state(waypoints[-1]), goal, atol = 1e-5)

    # furthest reachable waypoint becomes subgoal

    score_fn = lambda state, waypoints: waypoints @ state
    plan = c_slerp_plan(state_embed, goal_embed, num_waypoints = 2, alpha = 0., threshold = -1., score_fn = score_fn)

    waypoints, num_waypoints, selected_index, subgoal = plan
    assert selected_index == 2
    assert torch.allclose(subgoal, waypoints[2], atol = 1e-5)

    # decode state from subgoal

    assert torch.allclose(decode_state(subgoal), goal, atol = 1e-5)


def test_latent_world_model_rollout():
    from contrastive_rl_pytorch import LatentWorldModel

    dim, action_dim, batch, horizon = 8, 4, 5, 3
    model = LatentWorldModel(dim, action_dim, dim_hidden = 32, depth = 2)

    latent = torch.randn(batch, dim)
    latent = latent / latent.norm(dim = -1, keepdim = True)
    actions = torch.randn(batch, horizon, action_dim)

    next_latent = model(latent, actions[:, 0])
    assert next_latent.shape == (batch, dim)
    assert torch.allclose(next_latent.norm(dim = -1), torch.ones(batch), atol = 1e-5)

    latents = model.rollout(latent, actions)
    assert latents.shape == (batch, horizon + 1, dim)
    assert torch.allclose(latents.norm(dim = -1), torch.ones(batch, horizon + 1), atol = 1e-5)

    latents.sum().backward()


def test_contrastive_world_model_losses():
    from contrastive_rl_pytorch import ContrastiveLearning, ContrastiveWorldModel

    dim_state, action_dim, dim_embed, batch, horizon = 16, 4, 8, 6, 3

    state_encoder = torch.nn.Sequential(
        torch.nn.Linear(dim_state, 32),
        torch.nn.SiLU(),
        torch.nn.Linear(32, dim_embed)
    )

    model = ContrastiveWorldModel(
        state_encoder,
        action_dim = action_dim,
        dim_embed = dim_embed,
        contrastive_learn = ContrastiveLearning(l2norm_embed = True),
        discrete_action = True,
        dim_hidden = 32
    )

    state = torch.randn(batch, dim_state)
    next_state = torch.randn(batch, dim_state)
    goal = torch.randn(batch, dim_state)
    action = torch.randint(0, action_dim, (batch,))

    state_embed = model.encode_state(state)
    goal_embed = model.encode_goal(goal)

    assert torch.allclose(state_embed.norm(dim = -1), torch.ones(batch), atol = 1e-5)
    assert torch.allclose(goal_embed.norm(dim = -1), torch.ones(batch), atol = 1e-5)

    losses = model.losses(state, action, next_state, goal)

    for name, loss in losses.items():
        assert torch.isfinite(loss), name

    losses['loss'].backward()

    states = torch.randn(batch, horizon + 1, dim_state)
    actions = torch.randint(0, action_dim, (batch, horizon))

    sequence_loss = model.sequence_world_model_loss(states, actions)
    assert torch.isfinite(sequence_loss)

    sequence_q_loss = model.sequence_q_contrastive_loss(states, actions, goal)
    assert torch.isfinite(sequence_q_loss)

    # q(s, a) = v(s') where s' is the predicted next latent

    with torch.no_grad():
        q_value = model.q_value(state, action, goal)
        next_embed = model.predict_next(state, action)
        value = model.value(next_embed, model.encode_goal(goal))

    assert torch.allclose(q_value, value, atol = 1e-6)


def test_world_model_planner_cem_and_c_slerp():
    from contrastive_rl_pytorch import (
        ContrastiveLearning,
        ContrastiveWorldModel,
        WorldModelPlanner
    )

    dim_state, action_dim, dim_embed = 16, 4, 8

    state_encoder = torch.nn.Sequential(
        torch.nn.Linear(dim_state, 32),
        torch.nn.SiLU(),
        torch.nn.Linear(32, dim_embed)
    )

    model = ContrastiveWorldModel(
        state_encoder,
        action_dim = action_dim,
        dim_embed = dim_embed,
        contrastive_learn = ContrastiveLearning(l2norm_embed = True),
        discrete_action = True,
        dim_hidden = 32
    )

    planner = WorldModelPlanner(
        model,
        horizon = 2,
        num_samples = 16,
        num_elites = 4,
        num_iters = 2,
        discrete_action = True,
        num_actions = action_dim,
        c_slerp_kwargs = dict(num_waypoints = 2, alpha = 0., threshold = 0.)
    )

    state = torch.randn(dim_state)
    goal = torch.randn(dim_state)

    plan = planner.plan(state, goal)
    assert plan.actions.shape == (2,)
    assert plan.latents.shape == (3, dim_embed)
    assert torch.isfinite(plan.score)

    slerp_plan = planner.plan(
        state,
        goal,
        use_c_slerp = True
    )

    assert slerp_plan.actions.shape == (2,)
    assert slerp_plan.latents.shape == (3, dim_embed)
    assert slerp_plan.waypoints.shape == (3, dim_embed)
    assert slerp_plan.subgoal.shape == (dim_embed,)

    # continuous branch also returns a valid action sequence

    continuous_planner = WorldModelPlanner(
        model,
        horizon = 2,
        num_samples = 16,
        num_elites = 4,
        num_iters = 2,
        discrete_action = False,
        action_low = -1.,
        action_high = 1.
    )

    continuous_plan = continuous_planner.plan(state, goal)
    assert continuous_plan.actions.shape == (2, action_dim)
    assert continuous_plan.latents.shape == (3, dim_embed)
import copy
import pytest
import torch
from torch import nn

param = pytest.mark.parametrize


def make_encoders(dim_state = 8, dim_action = 4, dim_embed = 32, extra = 0, seed = 0):
    torch.manual_seed(seed)

    state_encoder = nn.Sequential(
        nn.Linear(dim_state + extra, 32),
        nn.SiLU(),
        nn.Linear(32, dim_embed)
    )

    action_encoder = nn.Sequential(
        nn.Linear(dim_action, 32),
        nn.SiLU(),
        nn.Linear(32, dim_embed)
    )

    return state_encoder, action_encoder


@param('block_size', (1, 2, 4, 8, 32))
def test_block_tucker_shapes(block_size):
    from contrastive_rl_pytorch import BlockDiagonalTuckerActionEncoder

    dim_state, dim_action, dim_embed = 8, 4, 32
    state_encoder, action_encoder = make_encoders(dim_state, dim_action, dim_embed)

    encoder = BlockDiagonalTuckerActionEncoder(
        state_encoder,
        action_encoder,
        dim_state = dim_state,
        dim_action = dim_action,
        block_size = block_size,
        dim_embed = dim_embed
    )

    x = torch.randn(5, dim_state + dim_action)
    out = encoder(x)

    assert out.shape == (5, dim_embed * block_size)
    assert encoder.output_dim == dim_embed * block_size


def test_block_tucker_cp_equivalence():
    from contrastive_rl_pytorch import FactorizedActionEncoder, BlockDiagonalTuckerActionEncoder

    dim_state, dim_action, dim_embed = 8, 4, 32
    state_encoder, action_encoder = make_encoders(dim_state, dim_action, dim_embed)

    cp = FactorizedActionEncoder(
        copy.deepcopy(state_encoder),
        copy.deepcopy(action_encoder),
        dim_state = dim_state,
        dim_action = dim_action
    )

    block = BlockDiagonalTuckerActionEncoder(
        copy.deepcopy(state_encoder),
        copy.deepcopy(action_encoder),
        dim_state = dim_state,
        dim_action = dim_action,
        block_size = 1,
        dim_embed = dim_embed
    )

    x = torch.randn(5, dim_state + dim_action)

    assert torch.allclose(cp(x), block(x), atol = 1e-6)
    assert block(x).shape == (5, dim_embed)


def test_block_tucker_full_equivalence():
    from contrastive_rl_pytorch import TuckerActionEncoder, BlockDiagonalTuckerActionEncoder

    dim_state, dim_action, dim_embed = 8, 4, 16
    state_encoder, action_encoder = make_encoders(dim_state, dim_action, dim_embed)

    full = TuckerActionEncoder(
        copy.deepcopy(state_encoder),
        copy.deepcopy(action_encoder),
        dim_state = dim_state,
        dim_action = dim_action
    )

    block = BlockDiagonalTuckerActionEncoder(
        copy.deepcopy(state_encoder),
        copy.deepcopy(action_encoder),
        dim_state = dim_state,
        dim_action = dim_action,
        block_size = dim_embed,
        dim_embed = dim_embed
    )

    x = torch.randn(5, dim_state + dim_action)

    assert torch.allclose(full(x), block(x), atol = 1e-6)
    assert block(x).shape == (5, dim_embed * dim_embed)


def test_block_tucker_matches_manual_blocks():
    from contrastive_rl_pytorch import BlockDiagonalTuckerActionEncoder

    dim_state, dim_action, dim_embed, block_size = 8, 4, 16, 4
    state_encoder, action_encoder = make_encoders(dim_state, dim_action, dim_embed)

    encoder = BlockDiagonalTuckerActionEncoder(
        state_encoder,
        action_encoder,
        dim_state = dim_state,
        dim_action = dim_action,
        block_size = block_size,
        dim_embed = dim_embed
    )

    x = torch.randn(5, dim_state + dim_action)
    state, action = x[:, :dim_state], x[:, dim_state:]

    with torch.no_grad():
        phi_s = state_encoder(state).reshape(5, -1, block_size)
        phi_a = action_encoder(action).reshape(5, -1, block_size)
        expected = torch.einsum('b g i, b g j -> b g i j', phi_s, phi_a).reshape(5, -1)

    assert torch.allclose(encoder(x), expected, atol = 1e-6)


def test_block_tucker_extra_features():
    from contrastive_rl_pytorch import BlockDiagonalTuckerActionEncoder

    dim_state, dim_action, dim_embed, extra, block_size = 8, 4, 16, 3, 4
    state_encoder, action_encoder = make_encoders(dim_state, dim_action, dim_embed, extra = extra)

    encoder = BlockDiagonalTuckerActionEncoder(
        state_encoder,
        action_encoder,
        dim_state = dim_state,
        dim_action = dim_action,
        block_size = block_size,
        dim_embed = dim_embed
    )

    x = torch.randn(5, dim_state + dim_action + extra)
    out = encoder(x)

    assert out.shape == (5, dim_embed * block_size)


def test_block_tucker_invalid_block():
    from contrastive_rl_pytorch import BlockDiagonalTuckerActionEncoder

    state_encoder, action_encoder = make_encoders(dim_embed = 10)

    with pytest.raises(AssertionError):
        BlockDiagonalTuckerActionEncoder(
            state_encoder,
            action_encoder,
            dim_state = 8,
            dim_action = 4,
            block_size = 4,
            dim_embed = 10
        )


def test_block_tucker_trainer_step():
    from contrastive_rl_pytorch import (
        BlockDiagonalTuckerActionEncoder,
        ContrastiveRLTrainer,
        ContrastiveLearning
    )
    from x_mlps_pytorch import MLP

    dim_state, dim_action, dim_embed, block_size = 8, 4, 16, 4
    state_encoder, action_encoder = make_encoders(dim_state, dim_action, dim_embed)

    encoder = BlockDiagonalTuckerActionEncoder(
        state_encoder,
        action_encoder,
        dim_state = dim_state,
        dim_action = dim_action,
        block_size = block_size,
        dim_embed = dim_embed
    )

    goal_encoder = MLP(dim_state, 32, dim_embed * block_size)

    trainer = ContrastiveRLTrainer(
        encoder,
        goal_encoder,
        batch_size = 8,
        repetition_factor = 1,
        cpu = True,
        contrastive_learn = ContrastiveLearning()
    )

    trajectories = torch.randn(16, 20, dim_state)
    actions = torch.randn(16, 20, dim_action)
    loss = trainer(
        trajectories,
        1,
        lens = torch.full((16,), 20),
        actions = actions,
        pbar = False
    )

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))
