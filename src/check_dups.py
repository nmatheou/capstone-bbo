import numpy as np
import re

lines = open('results_week4.txt').readlines()
coords = {}
func_idx = 0
for line in lines:
    if line.startswith('--- Function '):
        func_idx = int(re.search(r'\d+', line).group())
    elif line.startswith('Coordinate String:') and not line.startswith('Coordinate String (GP suggestion):'):
        coords[func_idx] = np.array([float(x) for x in line.split(': ')[1].strip().split('-')])

for f in range(1, 9):
    if f not in coords:
        continue
    X = np.load(f'next_week_data/function_{f}/initial_inputs.npy')
    x_new = coords[f]
    dists = np.linalg.norm(X - x_new, axis=1)
    min_dist = np.min(dists)
    match_idx = np.argmin(dists)
    if min_dist < 1e-4:
        print(f'F{f}: DANGER! Duplicate of index {match_idx} (dist {min_dist}) -> {x_new}')
    else:
        print(f'F{f}: OK. Closest is index {match_idx} (dist {min_dist:.6f})')
