"""PCA on accumulated inputs, per function.

For each function F1-F8, fits a PCA with all components (one per input dimension)
and ranks every PC by explained variance. Writes a markdown summary.
"""
import os
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

base_dir = Path(__file__).resolve().parents[1] / 'data' / 'accumulated'

lines = ['# PCA – All PCs Ranked per Function', '',
         'Inputs from `data/accumulated/function_i/initial_inputs.npy` (all points to date). '
         'One PCA per function, all components kept, ranked by explained variance. '
         '"Top dim" = dimension with the largest absolute loading on that PC.', '']

overview = ['## Overview', '', '| Function | Dims | N points | PC1 % | PCs for 90% var | Top dim on PC1 |',
            '|---|---|---|---|---|---|']
details = []

for i in range(1, 9):
    path = base_dir / f'function_{i}' / 'initial_inputs.npy'
    if not path.exists():
        continue
    X = np.load(path)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape
    pca = PCA(n_components=min(n, d)).fit(X)
    var = pca.explained_variance_ratio_
    cum = np.cumsum(var)
    comps = pca.components_
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    overview.append(f'| F{i} | {d} | {n} | {var[0]*100:.1f}% | {n90} | d{np.argmax(np.abs(comps[0]))+1} |')

    details += [f'## F{i} ({d}D, {n} points)', '']
    header = '| Rank | Var % | Cum % | Top dim | ' + ' | '.join(f'd{k+1}' for k in range(d)) + ' |'
    details += [header, '|' + '---|' * (4 + d)]
    for r, (v, c, comp) in enumerate(zip(var, cum, comps), start=1):
        top = np.argmax(np.abs(comp)) + 1
        loads = ' | '.join(f'{w:+.2f}' for w in comp)
        details.append(f'| PC{r} | {v*100:.1f}% | {c*100:.1f}% | d{top} | {loads} |')
    details.append('')

out = Path(os.getenv('AGY_ARTIFACT_DIR', Path.cwd())) / 'pca_summary.md'
out.write_text('\n'.join(lines + overview + [''] + details), encoding='utf-8')
print(f'PCA summary written to {out}')
