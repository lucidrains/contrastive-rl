import torch
from torch.nn import Module
import torch.nn.functional as F

from einops import einsum

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

def arange_from_tensor_dim(t, dim):
    device = t.device
    return torch.arange(t.shape[dim], device = device)

# tensor functions

def contrastive_loss(
    embeds1,          # (b d)
    embeds2,          # (b d)
    norm = False,     # not needed as original paper had a very nice negative results section at the end, but we'll allow for it
    temperature = 1.
):
    assert embeds1.shape == embeds2.shape

    if norm:
        embeds1, embeds2 = map(l2norm, (embeds1, embeds2))

    sim = einsum(embeds1, embeds2, 'i d, j d -> i j')

    if temperature != 1.:
        sim = sim * temperature

    labels = arange_from_tensor_dim(embeds1, dim = 0)

    contrastive_loss = F.cross_entropy(sim, labels)

    return contrastive_loss
