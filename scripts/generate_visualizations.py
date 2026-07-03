#!/usr/bin/env python3
"""
generate_visualizations.py
==========================

Additive visualization utility for the
`rl-alignment-and-reward-hacking` research repository.

This script does NOT touch any of the original experiment code, reward
functions, hyper-parameters, environments or committed evaluation data.
It only *reads* the project's wrappers/training configuration and *adds*
new artifacts under ``logs/videos/``.

For every trained agent it:

  1. Discovers an existing model checkpoint (``*.zip``).
  2. If none is found, retrains the agent using the EXACT wrappers and
     hyper-parameters defined in the original training scripts, writing the
     new checkpoint to an isolated directory (``logs/videos/_models/``) so
     the committed ``evaluations.npz`` / ``monitor.csv`` files are never
     overwritten.
  3. Runs a single deterministic evaluation episode with
     ``render_mode="rgb_array"``.
  4. Prepends a short title card (environment, reward function, learned
     behavior) and writes an MP4 (and optional GIF) to ``logs/videos/``.
  5. Prints the checkpoint used, episode reward, episode length and whether
     the episode terminated or truncated.

Run from the repository root::

    python scripts/generate_visualizations.py

Optional flags::

    --no-train         Do not train; skip any agent without a checkpoint.
    --agents a,b       Only process the named agents (keys below).
    --no-gif           Skip GIF generation (MP4 only).
    --seed 42          Evaluation reset seed (deterministic rollout).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# --- Repository imports -----------------------------------------------------
# Make the repo root importable so we can reuse the *exact* wrappers the
# experiments were trained with (we never modify them).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import EvalCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from src.wrappers import (  # noqa: E402
    PureSparseRewardWrapper,
    HeavilyShapedEnergyWrapper,
    HoverExploitWrapper,
)

import imageio.v2 as imageio  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


# --- Output locations -------------------------------------------------------
VIDEO_DIR = os.path.join(REPO_ROOT, "logs", "videos")
MODEL_DIR = os.path.join(REPO_ROOT, "logs", "videos", "_models")


# ---------------------------------------------------------------------------
# Agent configuration.
#
# Each entry mirrors one experiment from the original training scripts:
#   * MountainCar  -> experiments/train_mountain_car.py  (default/sparse/shaped)
#   * LunarLander  -> experiments/train_lunar_lander.py  (hover_exploit)
#
# The training hyper-parameters below are copied verbatim from those scripts
# so that a fresh checkpoint reproduces the original experimental setup.
# ---------------------------------------------------------------------------
@dataclass
class AgentSpec:
    key: str                       # output filename stem + model subdir
    env_id: str
    reward_name: str               # human-readable reward-function label
    behavior: str                  # short description of the learned behavior
    wrapper: Optional[Callable] = None   # reward/step wrapper class (or None)
    # Training config (exact copy of the originals):
    total_timesteps: int = 500_000
    eval_freq: int = 10_000
    ppo_kwargs: dict = field(default_factory=dict)
    # Where the *original* experiment would have saved best_model.zip
    # (searched during checkpoint discovery, read-only):
    original_log_subdir: str = ""
    # Proxy-reward reporter: returns the reward the agent actually optimized,
    # for the reward-hacking commentary. Signature: (env, obs) -> float.
    proxy_reward: Optional[Callable] = None


def _mc_ppo():
    # From experiments/train_mountain_car.py
    return dict(verbose=0, ent_coef=0.01, learning_rate=0.001)


def _sparse_proxy(env, obs) -> float:
    # Mirrors PureSparseRewardWrapper.reward for *reporting only*.
    return 1.0 if env.unwrapped.state[0] >= 0.5 else 0.0


def _shaped_proxy(env, obs) -> float:
    # Mirrors HeavilyShapedEnergyWrapper.reward for *reporting only*.
    pos, vel = env.unwrapped.state[0], env.unwrapped.state[1]
    return (np.sin(3 * pos) + vel ** 2) * 10


def _hover_proxy(env, obs) -> float:
    # Mirrors HoverExploitWrapper.step for *reporting only*.
    if obs[6] == 1.0 or obs[7] == 1.0:
        return -100.0
    if obs[1] > 0:
        return 1.0
    return 0.0


AGENTS: dict[str, AgentSpec] = {
    "mountaincar_reward1": AgentSpec(
        key="mountaincar_reward1",
        env_id="MountainCar-v0",
        reward_name="Reward Function 1: Default (-1 per step)",
        behavior=(
            "Aligned. The time penalty drives the agent to build momentum "
            "with the back-and-forth swing and reach the flag efficiently "
            "(~90 steps)."
        ),
        wrapper=None,
        total_timesteps=500_000,
        eval_freq=10_000,
        ppo_kwargs=_mc_ppo(),
        original_log_subdir=os.path.join("logs", "mountain_car", "default"),
    ),
    "mountaincar_reward2": AgentSpec(
        key="mountaincar_reward2",
        env_id="MountainCar-v0",
        reward_name="Reward Function 2: Pure Sparse (+1 only at goal)",
        behavior=(
            "Sparse reward, still aligned here. With reward zero everywhere "
            "except the flag the gradient signal is scarce, but PPO "
            "bootstrapped from a chance success and learned the momentum "
            "swing, reaching the goal in ~103 steps (slightly less efficient "
            "than the dense-reward agent)."
        ),
        wrapper=PureSparseRewardWrapper,
        total_timesteps=500_000,
        eval_freq=10_000,
        ppo_kwargs=_mc_ppo(),
        original_log_subdir=os.path.join("logs", "mountain_car", "sparse"),
        proxy_reward=_sparse_proxy,
    ),
    "mountaincar_reward3": AgentSpec(
        key="mountaincar_reward3",
        env_id="MountainCar-v0",
        reward_name="Reward Function 3: Heavily Shaped Energy",
        behavior=(
            "Alignment tax / policy collapse. Rewarding mechanical energy "
            "(kinetic + potential) creates a local optimum: the agent settles "
            "into a small oscillation near the valley floor (position stays "
            "around -0.4), never builds enough momentum to reach the flag, "
            "and the episode times out at 200 steps."
        ),
        wrapper=HeavilyShapedEnergyWrapper,
        total_timesteps=500_000,
        eval_freq=10_000,
        ppo_kwargs=_mc_ppo(),
        original_log_subdir=os.path.join("logs", "mountain_car", "shaped"),
        proxy_reward=_shaped_proxy,
    ),
    "lunar_hover": AgentSpec(
        key="lunar_hover",
        env_id="LunarLander-v3",
        reward_name="Hover Exploit (+1 airborne, -100 on ground contact)",
        behavior=(
            "Specification gaming / reward hacking. Penalizing ground contact "
            "and rewarding airborne time teaches the lander to hover for the "
            "full episode, farming the proxy reward (+1000) while never "
            "landing; the true-task return stays deeply negative (~-284)."
        ),
        wrapper=HoverExploitWrapper,
        total_timesteps=1_500_000,
        eval_freq=50_000,
        ppo_kwargs=dict(verbose=0),  # From experiments/train_lunar_lander.py
        original_log_subdir=os.path.join("logs", "lunar_lander", "hover_exploit"),
        proxy_reward=_hover_proxy,
    ),
}


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def discover_checkpoint(spec: AgentSpec) -> Optional[str]:
    """Return the path to an existing checkpoint, or None if none exists.

    Search order (first hit wins):
      1. logs/videos/_models/<key>/final_model.zip   (this script's output)
      2. logs/videos/_models/<key>/best_model.zip
      3. the original experiment's best_model.zip (read-only), if committed
    """
    candidates = [
        os.path.join(MODEL_DIR, spec.key, "final_model.zip"),
        os.path.join(MODEL_DIR, spec.key, "best_model.zip"),
        os.path.join(REPO_ROOT, spec.original_log_subdir, "best_model.zip"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Training (only when no checkpoint exists) -- exact copy of original setup
# ---------------------------------------------------------------------------
def train_agent(spec: AgentSpec) -> str:
    """Train one agent with the original hyper-parameters into an isolated dir.

    Reproduces experiments/train_*.py exactly:
      env = Wrapper(Monitor(base_env, out_dir))   # Monitor logs TRUE reward
      PPO trains on the wrapped (proxy) reward
      EvalCallback saves best_model.zip on the unmodified env's true reward
    Additionally saves final_model.zip (the fully-trained end-state policy).
    """
    out_dir = os.path.join(MODEL_DIR, spec.key)
    os.makedirs(out_dir, exist_ok=True)

    print(f"    [train] {spec.key}: {spec.total_timesteps:,} timesteps "
          f"on {spec.env_id} ... (this can take a while)")
    t0 = time.time()

    base_env = gym.make(spec.env_id)
    env = Monitor(base_env, out_dir)
    if spec.wrapper is not None:
        env = spec.wrapper(env)

    eval_env = Monitor(gym.make(spec.env_id))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=out_dir,
        log_path=out_dir,
        eval_freq=spec.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = PPO("MlpPolicy", env, **spec.ppo_kwargs)
    model.learn(total_timesteps=spec.total_timesteps, callback=eval_callback)

    final_path = os.path.join(out_dir, "final_model.zip")
    model.save(final_path)

    env.close()
    eval_env.close()
    print(f"    [train] {spec.key}: done in {time.time() - t0:.0f}s "
          f"-> {final_path}")
    return final_path


# ---------------------------------------------------------------------------
# Title card
# ---------------------------------------------------------------------------
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]:
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _wrap(draw, text, font, max_width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def make_title_card(spec: AgentSpec, frame_shape, checkpoint: str) -> np.ndarray:
    """Render a title-card frame matching the rollout frame dimensions."""
    h, w = frame_shape[0], frame_shape[1]
    img = Image.new("RGB", (w, h), color=(15, 18, 28))
    draw = ImageDraw.Draw(img)
    margin = int(w * 0.06)
    max_w = w - 2 * margin

    title_font = _load_font(max(20, int(h * 0.075)))
    label_font = _load_font(max(15, int(h * 0.048)))
    body_font = _load_font(max(13, int(h * 0.040)))

    y = int(h * 0.10)
    env_name = spec.env_id
    draw.text((margin, y), env_name, font=title_font, fill=(120, 200, 255))
    y += int(h * 0.11)

    for line in _wrap(draw, spec.reward_name, label_font, max_w):
        draw.text((margin, y), line, font=label_font, fill=(255, 235, 120))
        y += int(h * 0.058)

    y += int(h * 0.03)
    draw.text((margin, y), "Learned behavior:", font=body_font,
              fill=(180, 180, 180))
    y += int(h * 0.052)
    for line in _wrap(draw, spec.behavior, body_font, max_w):
        draw.text((margin, y), line, font=body_font, fill=(230, 230, 230))
        y += int(h * 0.050)

    ckpt_rel = os.path.relpath(checkpoint, REPO_ROOT)
    draw.text((margin, h - int(h * 0.07)),
              f"checkpoint: {ckpt_rel}", font=body_font, fill=(130, 140, 150))

    return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Deterministic rollout
# ---------------------------------------------------------------------------
def run_rollout(spec: AgentSpec, checkpoint: str, seed: int):
    """Run one deterministic episode on the *unmodified* env, capturing frames.

    Reward is measured on the true (unwrapped) environment objective, which is
    the meaningful quantity for judging alignment vs. reward hacking. The proxy
    reward the agent actually optimized is reported alongside for context.
    """
    model = PPO.load(checkpoint, device="cpu")
    env = gym.make(spec.env_id, render_mode="rgb_array")

    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    true_reward = 0.0
    proxy_reward = 0.0
    length = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        true_reward += float(reward)
        if spec.proxy_reward is not None:
            proxy_reward += spec.proxy_reward(env, obs)
        length += 1
        frames.append(env.render())

    env.close()
    return {
        "frames": frames,
        "true_reward": true_reward,
        "proxy_reward": proxy_reward,
        "length": length,
        "terminated": terminated,
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def write_mp4(path, frames, fps=30):
    with imageio.get_writer(path, fps=fps, codec="libx264",
                            quality=8, macro_block_size=None) as writer:
        for f in frames:
            writer.append_data(f)


def write_gif(path, frames, fps=20, max_width=360, subsample=2):
    small = []
    for i, f in enumerate(frames):
        if i % subsample != 0:
            continue
        img = Image.fromarray(f)
        if img.width > max_width:
            ratio = max_width / img.width
            img = img.resize((max_width, int(img.height * ratio)),
                             Image.BILINEAR)
        small.append(np.asarray(img))
    imageio.mimsave(path, small, fps=fps // subsample if subsample else fps,
                    loop=0)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def process_agent(spec: AgentSpec, args) -> Optional[dict]:
    print(f"\n=== {spec.key} ({spec.env_id}) ===")
    checkpoint = discover_checkpoint(spec)

    if checkpoint is None:
        if args.no_train:
            print(f"    No checkpoint found and --no-train set; skipping "
                  f"{spec.key}.")
            return None
        print(f"    No checkpoint found for {spec.key}. Retraining is "
              f"REQUIRED (no committed *.zip models exist in the repo).")
        checkpoint = train_agent(spec)
    else:
        print(f"    Using existing checkpoint: "
              f"{os.path.relpath(checkpoint, REPO_ROOT)}")

    roll = run_rollout(spec, checkpoint, args.seed)

    end = "TERMINATED" if roll["terminated"] else "TRUNCATED"
    print(f"    ---- rollout report ----")
    print(f"    checkpoint        : {os.path.relpath(checkpoint, REPO_ROOT)}")
    print(f"    true env reward   : {roll['true_reward']:.2f}")
    if spec.proxy_reward is not None:
        print(f"    proxy reward      : {roll['proxy_reward']:.2f}  "
              f"(what the agent optimized)")
    print(f"    episode length    : {roll['length']} steps")
    print(f"    episode ended by  : {end}  "
          f"(terminated={roll['terminated']}, truncated={roll['truncated']})")

    # Build final frame list: title card (held) + rollout frames.
    title = make_title_card(spec, roll["frames"][0].shape, checkpoint)
    title_hold = int(args.fps * 2.5)  # ~2.5s
    all_frames = [title] * title_hold + roll["frames"]

    os.makedirs(VIDEO_DIR, exist_ok=True)
    mp4_path = os.path.join(VIDEO_DIR, f"{spec.key}.mp4")
    write_mp4(mp4_path, all_frames, fps=args.fps)
    print(f"    saved MP4         : {os.path.relpath(mp4_path, REPO_ROOT)}")

    gif_path = None
    if not args.no_gif:
        gif_path = os.path.join(VIDEO_DIR, f"{spec.key}.gif")
        write_gif(gif_path, all_frames, fps=args.fps)
        print(f"    saved GIF         : {os.path.relpath(gif_path, REPO_ROOT)}")

    return {
        "key": spec.key,
        "checkpoint": os.path.relpath(checkpoint, REPO_ROOT),
        "true_reward": roll["true_reward"],
        "proxy_reward": roll["proxy_reward"] if spec.proxy_reward else None,
        "length": roll["length"],
        "end": end,
        "mp4": os.path.relpath(mp4_path, REPO_ROOT),
        "gif": os.path.relpath(gif_path, REPO_ROOT) if gif_path else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-train", action="store_true",
                        help="Skip agents without an existing checkpoint.")
    parser.add_argument("--agents", type=str, default="",
                        help="Comma-separated subset of agent keys to process.")
    parser.add_argument("--no-gif", action="store_true",
                        help="Only write MP4 files.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Deterministic reset seed for the rollout.")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate.")
    args = parser.parse_args()

    keys = [k.strip() for k in args.agents.split(",") if k.strip()] \
        or list(AGENTS.keys())
    unknown = [k for k in keys if k not in AGENTS]
    if unknown:
        parser.error(f"Unknown agent keys: {unknown}. "
                     f"Valid keys: {list(AGENTS.keys())}")

    print("Output video directory :", os.path.relpath(VIDEO_DIR, REPO_ROOT))
    print("Isolated model directory:", os.path.relpath(MODEL_DIR, REPO_ROOT))
    print("Agents to process      :", keys)

    results = []
    for key in keys:
        try:
            r = process_agent(AGENTS[key], args)
            if r:
                results.append(r)
        except Exception as exc:  # keep going for remaining agents
            print(f"    !! {key} failed: {exc}")
            import traceback
            traceback.print_exc()

    print("\n================ SUMMARY ================")
    for r in results:
        proxy = "" if r["proxy_reward"] is None \
            else f", proxy={r['proxy_reward']:.1f}"
        print(f"  {r['key']:<20} true_reward={r['true_reward']:.1f}"
              f"{proxy}, len={r['length']}, {r['end']}")
        print(f"    -> {r['mp4']}"
              + (f" | {r['gif']}" if r['gif'] else ""))
    print("========================================")


if __name__ == "__main__":
    main()
