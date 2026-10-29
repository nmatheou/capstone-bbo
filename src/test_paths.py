"""Sanity checks for repository-relative data loading. Run from repo root:

    python src/test_paths.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from paths import (  # noqa: E402
    ACCUMULATED_DIR,
    INITIAL_N,
    REPO_ROOT,
    load_accumulated,
    load_history_before_week,
    load_initial,
)
from check_dups import parse_coordinate_strings, parse_plain_submissions  # noqa: E402


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


def main():
    assert_true(os.path.isdir(os.path.join(REPO_ROOT, "data", "accumulated")),
                f"missing accumulated dir under {REPO_ROOT}")
    assert_true(os.path.isdir(ACCUMULATED_DIR), "ACCUMULATED_DIR does not exist")

    expected_final = {1: 23, 2: 23, 3: 28, 4: 43, 5: 29, 6: 33, 7: 43, 8: 53}
    for fi, n0 in INITIAL_N.items():
        X0, y0 = load_initial(fi)
        X, y = load_accumulated(fi)
        assert_true(len(y0) == n0, f"F{fi} initial length {len(y0)} != {n0}")
        assert_true(len(y) == expected_final[fi], f"F{fi} accumulated {len(y)}")
        assert_true(np.allclose(X[:n0], X0), f"F{fi} initial slice mismatch")

        X_pre, y_pre = load_history_before_week(fi, 13)
        assert_true(len(y_pre) == n0 + 12, f"F{fi} pre-W13 length {len(y_pre)}")

    f1_init_best = float(np.max(load_initial(1)[1]))
    assert_true(abs(f1_init_best) < 1e-12, f"F1 init best should be 0, got {f1_init_best}")

    sample = (
        "--- Function 1 ---\n"
        "Coordinate String: 0.625571-0.623369\n"
        "Final Submitted Coordinate String: 0.628542-0.628542\n"
    )
    parsed = parse_coordinate_strings(sample)
    assert_true(np.allclose(parsed[1], [0.628542, 0.628542]), parsed)

    plain = parse_plain_submissions("0.628542-0.628542\n0.688100-0.387000\n")
    assert_true(len(plain) == 2, plain)

    # Runtime path and validation tests
    from paths import get_output_dir, validate_output_dir, validate_project_structure
    out_dir = get_output_dir(cli_arg=REPO_ROOT)
    assert_true(out_dir == REPO_ROOT, f"Expected {REPO_ROOT}, got {out_dir}")

    # Verify structure check passes for Week 13
    warnings = validate_project_structure(REPO_ROOT, week=13)
    assert_true(isinstance(warnings, list), "Expected list of warnings")

    # Verify non-existent path check raises FileNotFoundError
    try:
        validate_output_dir("non_existent_folder_xyz_12345")
        raise AssertionError("Expected FileNotFoundError for non-existent path")
    except FileNotFoundError:
        pass

    print("test_paths: all checks passed (including first-run & validation logic)")


if __name__ == "__main__":
    main()
