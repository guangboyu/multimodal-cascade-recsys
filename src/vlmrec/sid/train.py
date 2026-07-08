"""Train the RQ-VAE over fused item content and materialize semantic-ID codes.

Artifacts (``data/sid/``): ``rqvae.pt``, ``sid_codes.npy`` (N, levels int64), ``metrics.json``
(recon MSE curve + per-level utilization/perplexity + collision rate). Health gates: utilization
should exceed ~60% per level — below that the codebook collapsed and the SIDs are meaningless.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from ..paths import Paths
from ..retrieval.data import cfg_sources, load_retrieval_data
from ..utils import get_logger, pick_device, set_seed
from .rqvae import RQVAE, collision_rate

log = get_logger("vlmrec.sid.train")


def sid_codes_path(paths: Paths):
    return paths.data / "sid" / "sid_codes.npy"


def run(cfg, paths: Paths) -> dict:
    s = cfg.sid
    set_seed(int(cfg.seed))
    device = pick_device(str(cfg.device))
    out_dir = paths.data / "sid"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = load_retrieval_data(paths, sources=cfg_sources(cfg))
    x = torch.tensor(d.content, device=device)
    x = torch.nn.functional.normalize(x, dim=-1)  # unit rows: recon scale comparable across dims
    n = x.shape[0]

    model = RQVAE(
        in_dim=int(x.shape[1]),
        hidden=tuple(int(h) for h in s.hidden),
        latent_dim=int(s.latent_dim),
        levels=int(s.levels),
        n_codes=int(s.codebook_size),
        beta=float(s.beta),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(s.lr))
    log.info(
        "RQ-VAE: %d items x %d dims -> %d levels x %d codes (latent %d)",
        n,
        x.shape[1],
        int(s.levels),
        int(s.codebook_size),
        int(s.latent_dim),
    )

    # k-means init per level on the actual residual distribution (anti-collapse measure #1)
    with torch.no_grad():
        z = model.encoder(x)
        residual = z
        for q in model.quantizers:
            q.init_from(residual)
            quant, _ = q(residual)
            residual = residual - quant

    batch = int(s.batch_size)
    curve = []
    t0 = time.time()
    for ep in range(1, int(s.epochs) + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss, total_recon, nb = 0.0, 0.0, 0
        for i in range(0, n, batch):
            loss, recon, _ = model(x[perm[i : i + batch]])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss)
            total_recon += float(recon)
            nb += 1
        if ep % 25 == 0 or ep == int(s.epochs):
            model.eval()
            with torch.no_grad():
                z = model.encoder(x)
                residual = z
                level_stats = []
                for q in model.quantizers:
                    level_stats.append(q.stats(residual))
                    quant, _ = q(residual)
                    residual = residual - quant
            curve.append({"epoch": ep, "recon_mse": round(total_recon / nb, 6)})
            log.info(
                "ep%03d | loss %.5f recon %.5f | %s",
                ep,
                total_loss / nb,
                total_recon / nb,
                level_stats,
            )

    model.eval()
    codes = model.encode_codes(x).cpu()
    np.save(sid_codes_path(paths), codes.numpy().astype(np.int64))
    torch.save(model.state_dict(), out_dir / "rqvae.pt")

    with torch.no_grad():
        z = model.encoder(x)
        residual = z
        final_stats = []
        for q in model.quantizers:
            final_stats.append(q.stats(residual))
            quant, _ = q(residual)
            residual = residual - quant
    metrics = {
        "n_items": int(n),
        "in_dim": int(x.shape[1]),
        "levels": int(s.levels),
        "codebook_size": int(s.codebook_size),
        "sources": list(cfg_sources(cfg)),
        "levels_stats": final_stats,
        "collision_rate": collision_rate(codes),
        "recon_curve": curve,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    ok = all(ls["utilization"] >= 0.6 for ls in final_stats)
    log.info(
        "sid codes -> %s | collisions=%.3f | utilization gate %s",
        sid_codes_path(paths),
        metrics["collision_rate"],
        "PASS" if ok else "FAIL (codebook collapse — see learn/03)",
    )
    return metrics
