import sys
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def episodes(path):
    d = pd.read_csv(path, usecols=["step", "env_id", "ep_step", "ep_had_success"]).sort_values(["env_id", "step"])
    out = []
    for _, g in d.groupby("env_id"):
        es = g["ep_step"].to_numpy(); s = g["ep_had_success"].to_numpy().astype(bool)
        ends = np.where(np.diff(es) < 0)[0]
        out += list(s[ends])
    return np.array(out)


def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


res = [(p, episodes(p)) for p in sys.argv[1:]]
k0, n0 = res[0][1].sum(), len(res[0][1])
for p, e in res:
    k, n = e.sum(), len(e); lo, hi = wilson(k, n)
    pv = fisher_exact([[k, n - k], [k0, n0 - k0]])[1] if p != res[0][0] else float("nan")
    print(f"{p.split('/')[-1]:40s} success {k/n:.3f} ({k}/{n})  95%CI [{lo:.3f}, {hi:.3f}]  Fisher p vs first {pv:.3g}")
