from __future__ import annotations

# IG = Information Geometry (informaciona geometrija)

"""
Refinement calculus na loto CSV — v1

Na koraku t: S_t = draws[t]  (tačan next iz CSV — postoji).
P0 (apstraktno): uniforman 7-izbor iz {1…39}\\last, seed=39+t
P1 (rafinisano): top7 po Hebbian masi od draws[:t]

Merilo μ = prosečan |pred ∩ S_t|  (pogodaka)
Empirijski ⊑:  μ(P1) ≥ μ(P0)  na walk-forwardu.

CSV: loto7_4652_k57.csv, seed=39.
Ime: ig_refine_v1_walkspec.py
"""

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

SEED = 39
FRONT_N = 39
FRONT_SELECT = 7
LAMBDA_TEMP = 0.35
WARMUP = 500
STEP = 50
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "loto7_4652_k57.csv"

np.random.seed(SEED)


def load_draws(csv_path: Path = CSV_PATH) -> np.ndarray:
    draws = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if len(row) < FRONT_SELECT:
                continue
            try:
                draw = sorted(int(x.strip()) for x in row[:FRONT_SELECT])
            except ValueError:
                continue
            if len(draw) == FRONT_SELECT and all(1 <= x <= FRONT_N for x in draw):
                if len(set(draw)) == FRONT_SELECT:
                    draws.append(draw)
    if not draws:
        raise ValueError(f"Nema validnih kola u {csv_path}")
    return np.array(draws, dtype=int)


def hebbian_weights(draws, lam=LAMBDA_TEMP):
    W = np.zeros((FRONT_N, FRONT_N), dtype=float)
    for d in draws:
        idx = [int(x) - 1 for x in d.tolist()]
        for a, b in combinations(idx, 2):
            W[a, b] += 1.0
            W[b, a] += 1.0
    for t in range(len(draws) - 1):
        a_idx = [int(x) - 1 for x in draws[t].tolist()]
        b_idx = [int(x) - 1 for x in draws[t + 1].tolist()]
        for a in a_idx:
            for b in b_idx:
                if a == b:
                    continue
                W[a, b] += lam
                W[b, a] += lam
    np.fill_diagonal(W, 0.0)
    return W


def hebbian_add_draw(W, prev, cur, lam=LAMBDA_TEMP):
    idx = [int(x) - 1 for x in cur.tolist()]
    for a, b in combinations(idx, 2):
        W[a, b] += 1.0
        W[b, a] += 1.0
    a_idx = [int(x) - 1 for x in prev.tolist()]
    for a in a_idx:
        for b in idx:
            if a == b:
                continue
            W[a, b] += lam
            W[b, a] += lam
    np.fill_diagonal(W, 0.0)


def energy_distribution(W):
    D = W.copy()
    row = D.sum(axis=1, keepdims=True)
    row = np.where(row < 1e-18, 1.0, row)
    return D / row


def hebbian_mass(D, last):
    idx = [int(x) - 1 for x in last.tolist()]
    return D[idx].mean(axis=0)


def hits(pred, actual) -> int:
    return len(set(pred) & set(int(x) for x in actual.tolist()))


def predict_P0(last, t_index: int) -> list[int]:
    """Apstraktno: uniforman izbor van last."""
    ban = set(int(x) for x in last.tolist())
    pool = [n for n in range(1, FRONT_N + 1) if n not in ban]
    rng = np.random.default_rng(SEED + t_index)
    pick = rng.choice(pool, size=FRONT_SELECT, replace=False)
    return sorted(int(x) for x in pick)


def predict_P1(W, last) -> list[int]:
    """Rafinisano: top7 po Hebbian masi."""
    ban = set(int(x) for x in last.tolist())
    D = energy_distribution(W)
    mass = hebbian_mass(D, last)
    ranked = sorted(
        (i + 1 for i in range(FRONT_N) if (i + 1) not in ban),
        key=lambda n: (-float(mass[n - 1]), n),
    )
    return sorted(ranked[:FRONT_SELECT])


def walk_refine(draws: np.ndarray) -> dict:
    """Walk-forward: S_t = draws[t]; meri μ(P0), μ(P1); empirijski ⊑."""
    T = len(draws)
    W = hebbian_weights(draws[:WARMUP])
    h0, h1 = [], []
    t = WARMUP
    while t < T:
        if (t - WARMUP) % STEP == 0:
            last = draws[t - 1]
            S = draws[t]  # specifikacija: tačan next
            p0 = predict_P0(last, t)
            p1 = predict_P1(W, last)
            h0.append(hits(p0, S))
            h1.append(hits(p1, S))
        if t < T - 1:
            hebbian_add_draw(W, draws[t - 1], draws[t])
        t += 1

    mu0 = float(np.mean(h0)) if h0 else 0.0
    mu1 = float(np.mean(h1)) if h1 else 0.0
    return {
        "n_eval": len(h0),
        "mu_P0": mu0,
        "mu_P1": mu1,
        "refines_empirically": mu1 >= mu0,
        "delta": mu1 - mu0,
    }


def run_v1(csv_path: Path = CSV_PATH) -> None:
    draws = load_draws(csv_path)
    stats = walk_refine(draws)

    last = draws[-1]
    W_full = hebbian_weights(draws)
    next_p1 = predict_P1(W_full, last)

    print(f"CSV: {csv_path.name}")
    print(
        f"Kola: {len(draws)} | seed={SEED} | WARMUP={WARMUP} STEP={STEP} | ig_refine_v1"
    )
    print(f"last: {last.tolist()}")
    print()
    print("=== spec ===")
    print("S_t = draws[t]  (tačan next iz CSV na koraku t)")
    print("P0 → P1  |  μ = mean |pred ∩ S_t|")
    print()
    print("=== empirijski refinement ===")
    print(
        {
            "n_eval": stats["n_eval"],
            "mu_P0": round(stats["mu_P0"], 4),
            "mu_P1": round(stats["mu_P1"], 4),
            "delta": round(stats["delta"], 4),
            "P0_sqsubseteq_P1": stats["refines_empirically"],
        }
    )
    print()
    print("=== next (P1 na celom CSV) ===")
    print("next:", next_p1)


if __name__ == "__main__":
    run_v1()



"""
CSV: loto7_4652_k57.csv
Kola: 4652 | seed=39 | WARMUP=500 STEP=50 | ig_refine_v1
last: [7, 8, 14, 15, 17, 23, 32]

=== spec ===
S_t = draws[t]  (tačan next iz CSV na koraku t)
P0 → P1  |  μ = mean |pred ∩ S_t|

=== empirijski refinement ===
{'n_eval': 84, 'mu_P0': 1.1429, 'mu_P1': 1.3929, 'delta': 0.25, 'P0_sqsubseteq_P1': True}

=== next (P1 na celom CSV) ===
next: [11, x, 26, y, 34, z, 39]
"""



"""
v1 — jedna spec + jedan korak + merilo
S_t = draws[t]; P0 uniform; P1 Hebbian top7; μ = mean hits; ⊑ ako μ(P1)≥μ(P0).

jedna spec S_t = draws[t], jedan korak P0 ⊑ P1, merilo walk-forward na CSV.




ig_refine_v1_walkspec.py

S_t = draws[t]

P0 uniform, P1 Hebbian top7

μ = mean pogodaka; ⊑ ako μ(P1) ≥ μ(P0)

na kraju next od P1
"""



"""
Rezultat: P0 ⊑ P1 drži (μ 1.14 → 1.39).

next: [11, x, 26, y, 34, z, 39] kaže samo ovo:

Na celom CSV-u, posle last, P1 (Hebbian top7 van last) bira tu komb — to je izlaz rafinisane procedure, ne S za kolo 4653 (tog reda još nema).

Walk-forward je merio P1 protiv pravog S_t; ovaj next je ista P1 logika, bez budućeg S.
"""
 
