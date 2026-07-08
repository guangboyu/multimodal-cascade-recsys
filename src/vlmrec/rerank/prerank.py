"""Pre-ranking: a lightweight scorer distilled from the heavy ranker.

The pre-ranker (a small Ranker: no DIN, no cross, small d_model) is trained to mimic the ranker's
click logit on the retrieval candidate sets (MSE distillation). It cheaply cuts ~200 candidates to
~50 while preserving the ranker's top picks — the consistency the cascade needs. We report how much
of the ranker's top-10 survives in the pre-ranker's top-50.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..ranking.model import Ranker
from ..utils import get_logger

log = get_logger("vlmrec.rerank.prerank")


def distill_prerank(
    teacher, rd, tensors, cand_topk, device, epochs=2, d_model=64, batch_users=512, lr=1e-3, seed=42
):
    content, cat, seq_t, ret_ui = tensors
    use_score = bool(getattr(teacher, "use_retrieval_score", False))
    d = rd.base
    n_users, k = cand_topk.shape
    student = Ranker(
        d.content_dim,
        d.cat_cardinalities,
        d_model=d_model,
        n_tasks=1,
        use_multimodal=True,
        use_din=False,
        use_cross=False,
        use_mmoe=False,
        use_retrieval_score=use_score,
    ).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    cand_t = torch.tensor(cand_topk, device=device)
    teacher.eval()
    rng = np.random.default_rng(seed)
    log.info(
        "distill pre-ranker: params=%s (teacher mimic, score_feature=%s)",
        f"{sum(p.numel() for p in student.parameters()):,}",
        use_score,
    )

    for ep in range(1, epochs + 1):
        student.train()
        perm = rng.permutation(n_users)
        total, nb = 0.0, 0
        for s in range(0, n_users, batch_users):
            ub = perm[s : s + batch_users]
            users_t = torch.as_tensor(np.repeat(ub, k), device=device)
            cand_flat = cand_t[torch.as_tensor(ub, device=device)].reshape(-1)
            seq_b = seq_t[users_t]
            ret = (ret_ui[0][users_t] * ret_ui[1][cand_flat]).sum(-1) if use_score else None
            with torch.no_grad():
                t_logit = teacher(cand_flat, seq_b, content, cat, ret_score=ret)[:, 0]
            s_logit = student(cand_flat, seq_b, content, cat, ret_score=ret)[:, 0]
            loss = F.mse_loss(s_logit, t_logit)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        log.info("prerank ep%02d | distill MSE %.4f", ep, total / nb)

    return student, _consistency(teacher, student, rd, tensors, cand_topk, device)


@torch.no_grad()
def _consistency(
    teacher, student, rd, tensors, cand_topk, device, sample=3000, top_t=10, top_s=50, seed=0
):
    content, cat, seq_t, ret_ui = tensors
    use_score = bool(getattr(teacher, "use_retrieval_score", False))
    n_users, k = cand_topk.shape
    rng = np.random.default_rng(seed)
    users = rng.choice(n_users, size=min(sample, n_users), replace=False)
    cand_t = torch.tensor(cand_topk, device=device)
    recovered = []
    for s in range(0, len(users), 512):
        ub = users[s : s + 512]
        users_t = torch.as_tensor(np.repeat(ub, k), device=device)
        cand_flat = cand_t[torch.as_tensor(ub, device=device)].reshape(-1)
        seq_b = seq_t[users_t]
        ret = (ret_ui[0][users_t] * ret_ui[1][cand_flat]).sum(-1) if use_score else None
        ts = (
            teacher(cand_flat, seq_b, content, cat, ret_score=ret)[:, 0]
            .reshape(len(ub), k)
            .cpu()
            .numpy()
        )
        ss = (
            student(cand_flat, seq_b, content, cat, ret_score=ret)[:, 0]
            .reshape(len(ub), k)
            .cpu()
            .numpy()
        )
        for r in range(len(ub)):
            t_top = set(np.argpartition(-ts[r], top_t)[:top_t])
            s_top = set(np.argpartition(-ss[r], top_s)[:top_s])
            recovered.append(len(t_top & s_top) / top_t)
    key = f"prerank_top{top_s}_recall_of_ranker_top{top_t}"
    return {key: round(float(np.mean(recovered)), 4)}
