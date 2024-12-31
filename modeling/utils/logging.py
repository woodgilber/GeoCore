import cProfile
import logging
import os
import pstats
import subprocess
import urllib.parse
from typing import Dict

import mlflow

logger = logging.getLogger("modeling")


def flatten_dict(cfg: Dict, sep: str = "/", parent_key: str = ""):
    """Flatten a nested dict"""
    out = {}
    for k, v in cfg.items():
        key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, sep, key))
        else:
            out[key] = v
    return out


def log_config(cfg: Dict) -> None:
    """Log the config to mlflow

    Args:
        cfg (Dict): config as dict
    """
    flat_cfg = flatten_dict(cfg)

    # mlflow only takes 500 char max params
    for key, value in flat_cfg.items():
        mlflow.log_param(key, str(value)[:500])


def Cprofile_and_log(function, description, snakeviz_port=None):
    """
    Profile a function, save outputs as .prof and .txt files, and log a Snakeviz URL in MLflow.

    Args:
        function: Function to profile (wrapped in a lambda if it takes arguments).
        description: A description of the profiling task (used for filenames).
        snakeviz_port: Port to use for the Snakeviz server (default: 8081).
    """

    snakeviz_port = snakeviz_port or 8080

    logger.info(f"Profiling {description}...")

    # Start profiling
    profiler = cProfile.Profile()
    profiler.enable()
    result = function()
    profiler.disable()

    # Create profiling output folder
    profile_folder = f"./runs/{description.replace(' ', '_')}"
    os.makedirs(profile_folder, exist_ok=True)

    # Save the .prof file
    profile_path = os.path.join(profile_folder, "profile.prof")
    profiler.dump_stats(profile_path)
    logger.info(f"Profiler stats saved to {profile_path}.")

    # Save the human-readable pstats summary as a .txt file
    pstats_path = os.path.join(profile_folder, "profile_summary.txt")
    with open(pstats_path, "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats()
    logger.info(f"Profiler summary saved to {pstats_path}.")

    # Generate Snakeviz link and log in MLflow
    try:
        snakeviz_url = f"http://127.0.0.1:{snakeviz_port}/snakeviz/{urllib.parse.quote(profile_path)}"
        logger.info(f"Snakeviz will be available at {snakeviz_url}")

        # Start the Snakeviz server in the background
        subprocess.Popen(
            ["snakeviz", "-H", "127.0.0.1", "-p", str(snakeviz_port), profile_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Use a unique parameter name for Snakeviz URL
        snakeviz_param_name = f"snakeviz_url_{description.replace(' ', '_')}"
        mlflow.log_artifact(profile_path, artifact_path=f"profile/{snakeviz_url}")

    except Exception as e:
        logger.error(f"Failed to start Snakeviz server: {e}")

    # Log the .prof file in MLflow
    mlflow.log_artifact(profile_path, artifact_path=f"profile/{description}")
    mlflow.log_artifact(pstats_path, artifact_path=f"profile/{description}")

    logger.info(f"Finished profiling {description}. Data saved to {profile_folder}.")
    return result
