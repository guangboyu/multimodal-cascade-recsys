"""Pure-logic tests for residual quantization (shapes, determinism, collapse countermeasures)."""

from __future__ import annotations

import torch

from vlmrec.sid.rqvae import RQVAE, VectorQuantizerEMA, collision_rate, kmeans_init


def _model(levels=2, n_codes=4):
    torch.manual_seed(0)
    return RQVAE(in_dim=8, hidden=(16,), latent_dim=4, levels=levels, n_codes=n_codes)


def test_encode_codes_shape_and_range():
    m = _model(levels=3, n_codes=5)
    x = torch.randn(32, 8)
    codes = m.encode_codes(x)
    assert codes.shape == (32, 3)
    assert int(codes.min()) >= 0 and int(codes.max()) < 5


def test_encode_codes_deterministic_in_eval():
    m = _model()
    x = torch.randn(16, 8)
    a, b = m.encode_codes(x), m.encode_codes(x)
    assert torch.equal(a, b)


def test_forward_loss_finite_and_codes_stack():
    m = _model()
    m.train()
    loss, recon, codes = m(torch.randn(64, 8))
    assert torch.isfinite(loss) and torch.isfinite(recon)
    assert codes.shape == (64, 2)


def test_dead_code_reseeding_triggers():
    torch.manual_seed(0)
    q = VectorQuantizerEMA(n_codes=4, dim=2, dead_steps=1)
    # plant one codeword on the data and three far away — the far ones go dead immediately
    q.codebook.copy_(torch.tensor([[0.0, 0.0], [50.0, 50.0], [60.0, 60.0], [70.0, 70.0]]))
    far_before = q.codebook[1:].clone()
    q.train()
    x = torch.randn(128, 2) * 0.1  # all near the origin -> only code 0 ever wins
    q(x)  # step 1: codes 1-3 unused -> steps_unused = 1 == dead_steps
    q(x)  # step 2: dead codes re-seeded from live batch residuals
    assert not torch.allclose(q.codebook[1:], far_before)  # they moved to the data
    assert float(q.codebook[1:].abs().max()) < 5.0  # re-seeded near the origin cluster


def test_kmeans_init_finds_separated_clusters():
    torch.manual_seed(0)
    a = torch.randn(50, 2) * 0.05 + torch.tensor([5.0, 5.0])
    b = torch.randn(50, 2) * 0.05 - torch.tensor([5.0, 5.0])
    cents = kmeans_init(torch.cat([a, b]), n_codes=2, iters=5)
    dist_a = torch.cdist(torch.tensor([[5.0, 5.0]]), cents).min()
    dist_b = torch.cdist(torch.tensor([[-5.0, -5.0]]), cents).min()
    assert float(dist_a) < 1.0 and float(dist_b) < 1.0


def test_collision_rate_known_values():
    codes = torch.tensor([[0, 0], [0, 0], [1, 2]])
    assert collision_rate(codes) == round(2 / 3, 4)
    unique = torch.tensor([[0, 0], [0, 1], [1, 0]])
    assert collision_rate(unique) == 0.0
