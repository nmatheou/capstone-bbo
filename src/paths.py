"""Repository-relative paths and runtime configuration for BBO optimization tools.

Features:
- Auto-detects the repository root.
- Interactive first-run prompt to configure base/output directory (cached in configs/runtime_config.json).
- On subsequent runs, verifies that the configured directory exists before executing.
- Validates that required project directories (data/accumulated, configs, etc.) exist.
- Supports override via CLI flag (--output-dir) or environment variable (BBO_OUTPUT_DIR).
"""

from __future__ import annotations

import ast
from datetime import datetime
import json
import os
import re
import sys

import numpy as np

# Absolute path to repository root (parent directory of this src/ folder)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNTIME_CONFIG_FILE = "runtime_config.json"
ACCUMULATED_DIR = os.path.join(REPO_ROOT, "data", "accumulated")
CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "hyperparameters.json")

DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}

# Week-0 sample sizes from the oracle (README dataset table)
INITIAL_N = {1: 10, 2: 10, 3: 15, 4: 30, 5: 16, 6: 20, 7: 30, 8: 40}


def get_runtime_config_path(repo_root: str = REPO_ROOT) -> str:
    """Return the absolute path to configs/runtime_config.json."""
    return os.path.join(repo_root, "configs", RUNTIME_CONFIG_FILE)


def validate_output_dir(path: str) -> None:
    """Verify that the given path exists and is a directory."""
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[BBO Error] Configured output directory does not exist:\n"
            f"  --> '{path}'\n"
            f"Remediation steps:\n"
            f"  1. Verify the folder path exists on this machine.\n"
            f"  2. Delete or edit '{get_runtime_config_path()}' to reset configuration.\n"
            f"  3. Or specify a valid path using the --output-dir <path> argument.\n"
        )
    if not os.path.isdir(path):
        raise NotADirectoryError(f"[BBO Error] Path is not a directory: '{path}'")


def get_output_dir(cli_arg: str | None = None, prompt_if_first_run: bool = True) -> str:
    """Resolve and validate the base/output directory.

    Precedence:
    1. CLI argument `--output-dir` (if passed)
    2. Environment variable `BBO_OUTPUT_DIR` (if set)
    3. Saved config in `configs/runtime_config.json` (if exists and valid)
    4. First-run setup:
       - If running interactively, prompt user to select repository root or a custom path.
       - If non-interactive, default to REPO_ROOT.
       The choice is saved to `configs/runtime_config.json` for subsequent runs.
    """
    # 1. CLI argument override
    if cli_arg:
        abs_cli = os.path.abspath(cli_arg)
        validate_output_dir(abs_cli)
        return abs_cli

    # 2. Environment variable override
    env_dir = os.environ.get("BBO_OUTPUT_DIR")
    if env_dir:
        abs_env = os.path.abspath(env_dir)
        validate_output_dir(abs_env)
        return abs_env

    # 3. Saved runtime config (subsequent runs)
    cfg_path = get_runtime_config_path()
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            saved_dir = cfg.get("output_dir")
            if saved_dir:
                abs_saved = os.path.abspath(saved_dir)
                if os.path.isdir(abs_saved):
                    return abs_saved
                else:
                    print(
                        f"\n[BBO Warning] Previously configured output directory not found:\n"
                        f"  '{abs_saved}'"
                    )
                    if not (prompt_if_first_run and sys.stdin.isatty()):
                        validate_output_dir(abs_saved)
        except Exception as e:
            print(f"[BBO Warning] Could not read runtime config ({e}), triggering re-configuration.")

    # 4. First-run setup (or recovery if saved dir is invalid)
    chosen_dir = REPO_ROOT
    is_interactive = sys.stdin.isatty() and prompt_if_first_run

    if is_interactive:
        print("\n" + "=" * 72)
        print(" [BBO Setup - Initial Configuration]")
        print(" No valid output directory configuration was found.")
        print(" Detected repository root:")
        print(f"   {REPO_ROOT}")
        print("=" * 72)
        try:
            user_input = input(
                " Press [Enter] to use repository root, or enter custom path: "
            ).strip()
            if user_input:
                custom_path = os.path.abspath(user_input)
                if not os.path.exists(custom_path):
                    create = input(
                        f" Directory '{custom_path}' does not exist. Create it? [Y/n]: "
                    ).strip().lower()
                    if create in ("", "y", "yes"):
                        os.makedirs(custom_path, exist_ok=True)
                        chosen_dir = custom_path
                    else:
                        print(" Using default repository root instead.")
                        chosen_dir = REPO_ROOT
                else:
                    chosen_dir = custom_path
        except (KeyboardInterrupt, EOFError):
            print("\n Setup interrupted, defaulting to repository root.")
            chosen_dir = REPO_ROOT
    else:
        print(f"[BBO Setup] Initialized output directory to repository root: {REPO_ROOT}")

    # Validate the chosen directory
    validate_output_dir(chosen_dir)

    # Save to configs/runtime_config.json for subsequent runs
    try:
        os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "output_dir": chosen_dir,
                    "configured_at": datetime.now().isoformat(),
                    "repo_root": REPO_ROOT,
                },
                f,
                indent=2,
            )
        print(f"[BBO Setup] Saved configuration to {os.path.relpath(cfg_path, REPO_ROOT)}")
    except Exception as e:
        print(f"[BBO Warning] Could not save runtime config to '{cfg_path}': {e}")

    return chosen_dir


def validate_project_structure(output_dir: str, week: int | None = None) -> list[str]:
    """Verify that all required project directories and files exist.

    Checks:
    - data/accumulated/ with function_1 through function_8
    - configs/hyperparameters.json
    - (If week >= 2): data/week_<week-1>_returns/ with inputs.txt and outputs.txt

    Ensures existence of output directories:
    - results/ and results/week_<week>/
    - logs/
    - submissions/ and submissions/week_<week>/
    """
    warnings: list[str] = []

    # 1. Base accumulated dataset check
    acc_dir = os.path.join(output_dir, "data", "accumulated")
    if not os.path.isdir(acc_dir):
        # Fallback check under REPO_ROOT if output_dir is custom
        if os.path.isdir(os.path.join(REPO_ROOT, "data", "accumulated")):
            acc_dir = os.path.join(REPO_ROOT, "data", "accumulated")
        else:
            raise FileNotFoundError(
                f"[BBO Error] Accumulated data directory not found:\n"
                f"  Expected: '{acc_dir}'\n"
                f"Please ensure the repository was cloned completely with data/accumulated/ intact."
            )

    for fi in range(1, 9):
        f_dir = os.path.join(acc_dir, f"function_{fi}")
        if not os.path.isdir(f_dir):
            raise FileNotFoundError(
                f"[BBO Error] Missing data folder for Function {fi}: '{f_dir}'"
            )
        npy_in = os.path.join(f_dir, "initial_inputs.npy")
        npy_out = os.path.join(f_dir, "initial_outputs.npy")
        if not os.path.exists(npy_in) or not os.path.exists(npy_out):
            raise FileNotFoundError(
                f"[BBO Error] Function {fi} is missing initial_inputs.npy or initial_outputs.npy in '{f_dir}'"
            )

    # 2. Hyperparameters configuration check
    cfg_file = os.path.join(output_dir, "configs", "hyperparameters.json")
    if not os.path.exists(cfg_file):
        cfg_file_fallback = os.path.join(REPO_ROOT, "configs", "hyperparameters.json")
        if not os.path.exists(cfg_file_fallback):
            raise FileNotFoundError(
                f"[BBO Error] Hyperparameters configuration file not found:\n"
                f"  Expected: '{cfg_file}'"
            )

    # 3. Weekly returns check if week is specified
    if week is not None and week >= 2:
        prev_w = week - 1
        returns_dir = os.path.join(output_dir, "data", f"week_{prev_w}_returns")
        if not os.path.isdir(returns_dir):
            fallback_ret = os.path.join(REPO_ROOT, "data", f"week_{prev_w}_returns")
            if os.path.isdir(fallback_ret):
                returns_dir = fallback_ret
            else:
                warnings.append(
                    f"Warning: data/week_{prev_w}_returns/ not found in '{output_dir}'. "
                    f"Will proceed if accumulated dataset already includes Week {prev_w} observations."
                )

    # 4. Ensure output directories exist
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "submissions"), exist_ok=True)

    if week is not None:
        os.makedirs(os.path.join(output_dir, "results", f"week_{week}"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "results", f"week_{week}", "plots"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "submissions", f"week_{week}"), exist_ok=True)

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions for data loading
# ─────────────────────────────────────────────────────────────────────────────

def get_accumulated_dir(output_dir: str | None = None) -> str:
    base = output_dir or get_output_dir(prompt_if_first_run=False)
    target = os.path.join(base, "data", "accumulated")
    if not os.path.isdir(target) and os.path.isdir(os.path.join(REPO_ROOT, "data", "accumulated")):
        return os.path.join(REPO_ROOT, "data", "accumulated")
    return target


def get_config_path(output_dir: str | None = None) -> str:
    base = output_dir or get_output_dir(prompt_if_first_run=False)
    target = os.path.join(base, "configs", "hyperparameters.json")
    if not os.path.exists(target) and os.path.exists(os.path.join(REPO_ROOT, "configs", "hyperparameters.json")):
        return os.path.join(REPO_ROOT, "configs", "hyperparameters.json")
    return target


def function_dir(func_idx: int, output_dir: str | None = None) -> str:
    return os.path.join(get_accumulated_dir(output_dir), f"function_{func_idx}")


def week_returns_dir(week: int, output_dir: str | None = None) -> str:
    base = output_dir or get_output_dir(prompt_if_first_run=False)
    target = os.path.join(base, "data", f"week_{week}_returns")
    if not os.path.isdir(target) and os.path.isdir(os.path.join(REPO_ROOT, "data", f"week_{week}_returns")):
        return os.path.join(REPO_ROOT, "data", f"week_{week}_returns")
    return target


def load_accumulated(func_idx: int, output_dir: str | None = None):
    f_dir = function_dir(func_idx, output_dir)
    X = np.load(os.path.join(f_dir, "initial_inputs.npy"))
    y = np.load(os.path.join(f_dir, "initial_outputs.npy"))
    return X, y


def load_initial(func_idx: int, output_dir: str | None = None):
    """Return only the Week-0 oracle baseline (before weekly queries)."""
    X, y = load_accumulated(func_idx, output_dir)
    n0 = INITIAL_N[func_idx]
    if len(y) < n0:
        raise ValueError(
            f"F{func_idx} accumulated length {len(y)} is smaller than INITIAL_N={n0}"
        )
    return X[:n0].copy(), y[:n0].copy()


def load_week_query(week: int, output_dir: str | None = None):
    """Return (inputs_list, outputs_list) for a weekly oracle return, or (None, None)."""
    w_dir = week_returns_dir(week, output_dir)
    try:
        with open(os.path.join(w_dir, "inputs.txt"), "r") as f:
            raw = f.read().replace("array(", "").replace(")", "")
        raw = re.sub(r"\]\s*\n\s*\[", "],\n[", raw)
        inputs_list = ast.literal_eval(f"[{raw}]")[-1]

        with open(os.path.join(w_dir, "outputs.txt"), "r") as f:
            out_raw = f.read()
        out_raw = re.sub(r"\]\s*\n\s*\[", "],\n[", out_raw)
        outputs_list = ast.literal_eval(f"[{out_raw}]")[-1]
        return inputs_list, outputs_list
    except FileNotFoundError:
        return None, None


def load_history_before_week(func_idx: int, week: int, output_dir: str | None = None):
    """Points known when proposing week `week` (baseline + returns through week-1)."""
    X, y = load_initial(func_idx, output_dir)
    for w in range(1, week):
        inputs_list, outputs_list = load_week_query(w, output_dir)
        if inputs_list is None:
            break
        X = np.vstack([X, np.atleast_2d(np.array(inputs_list[func_idx - 1]))])
        y = np.append(y, outputs_list[func_idx - 1])
    return X, y
