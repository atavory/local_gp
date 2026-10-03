"""wanger check: disagreement subspace, stable vs EoS. Diagonal H -> exact per-coord sim."""
import random

def run(eta, d=200, n_out=12, N=8, K=8, reps=120, seed=0):
    rng = random.Random(seed)
    lam = [rng.uniform(8, 12) for _ in range(n_out)] + [rng.uniform(0.1, 1.0) for _ in range(d - n_out)]
    sig = 1.0
    ov_s, ov_f, pr = [], [], []
    for r in range(reps):
        gap_var, des_var = [], []
        for j in range(d):
            a = 1.0 - eta * lam[j]
            dev = [0.0] * N
            pts = []
            for k in range(K):
                xi = [rng.gauss(0, sig) for _ in range(N)]
                m = sum(xi) / N
                dev = [a * dd + eta * (x - m) for dd, x in zip(dev, xi)]
                pts.extend(dev)
            m = sum(dev) / N
            gap_var.append(sum((x - m) ** 2 for x in dev) / (N - 1))
            pm = sum(pts) / len(pts)
            des_var.append(sum((x - pm) ** 2 for x in pts) / len(pts))
        top = sorted(range(d), key=lambda j: gap_var[j], reverse=True)[:n_out]
        ov_s.append(sum(1 for j in top if j < n_out) / n_out)
        ov_f.append(sum(1 for j in top if j >= d - 50) / n_out)
        s = sum(des_var); pr.append(s * s / sum(v * v for v in des_var))
    return sum(ov_s)/reps, sum(ov_f)/reps, sum(pr)/reps

for eta, tag in [(0.05, 'stable eta*lamax~0.6'), (0.15, 'EoS eta*lamax~1.8')]:
    s, f, p = run(eta)
    print(f"{tag}: top12-gap overlap sharp={s:.3f} flat50={f:.3f} design-PR={p:.1f} (nominal NK=64)", flush=True)
