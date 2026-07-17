from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import RawValue

import numpy as np
import torch
import torch.multiprocessing as mp


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_dummy_env(obs_dim: int, action_dim: int, device: torch.device, clip_actions: float):
    """Minimal env interface required by rl-games."""

    class _DummyEnv:
        observation_space = type("S", (), {"shape": (obs_dim,)})()
        action_space = type("S", (), {
            "shape": (action_dim,),
            "high":  np.full(action_dim, clip_actions, dtype=np.float32),
            "low":   np.full(action_dim, -clip_actions, dtype=np.float32),
        })()
        num_envs = 1

        def reset(self):
            return torch.zeros(1, obs_dim, device=device)

        def step(self, _):
            return (torch.zeros(1, obs_dim, device=device),
                    torch.zeros(1), torch.zeros(1, dtype=torch.bool), {})

    return _DummyEnv()


# ──────────────────────────────────────────────────────────────────────────────
# Inference process entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_inference_process(
    obs_shm: torch.Tensor,
    action_shm: torch.Tensor,
    obs_flag: RawValue,
    action_flag: RawValue,
    stop_event: mp.Event,
    *,
    checkpoint: str,
    cfg_path: str,
    obs_dim: int,
    action_dim: int,
    device_str: str,
) -> None:
    """Inference-only subprocess: no ROS, no GIL contention.

    Waiting on obs_flag is a pure spin-wait (continue). Using time.sleep()
    would add unnecessary waiting on the order of Linux's timer resolution (1-4ms).
    """
    import yaml

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Backend dispatch: factory/forge use rl-games; pick_place uses rsl_rl.
    backend = str(cfg.get("backend") or "rl_games").lower()
    if backend == "rsl_rl":
        _run_rsl_rl_inference(
            obs_shm, action_shm, obs_flag, action_flag, stop_event,
            cfg=cfg, checkpoint=checkpoint,
            obs_dim=obs_dim, action_dim=action_dim, device=device,
        )
        return

    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    cfg["params"].update({"load_checkpoint": True, "load_path": checkpoint})
    cfg["params"]["config"].update({
        "device": str(device),
        "device_name": str(device),
        "num_actors": 1,
    })

    clip_actions = float(cfg["params"]["env"].get("clip_actions", 1.0))
    dummy_env = _build_dummy_env(obs_dim, action_dim, device, clip_actions)
    vecenv.register("IsaacRlgWrapper", lambda *_, **__: dummy_env)
    env_configurations.register("rlgpu", {
        "vecenv_type": "IsaacRlgWrapper",
        "env_creator": lambda **__: dummy_env,
    })

    runner = Runner()
    runner.load(cfg)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")

    agent = runner.create_player()
    agent.restore(checkpoint)
    agent.reset()
    _ = agent.get_batch_size(torch.zeros(1, obs_dim, device=device), 1)
    if agent.is_rnn:
        agent.init_rnn()

    model_device = next(agent.model.parameters()).device

    print("[INFER] Ready.")
    action_flag.value = 2  # warm-up done signal

    # ── Inference loop (spin-wait) ────────────────────────────────────────────
    while not stop_event.is_set():
        if obs_flag.value != 1:
            continue  # ← pure spin: no sleep allowed
        obs_flag.value = 0

        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs_shm.to(device)).to(model_device)
            act = agent.get_action(obs_t, is_deterministic=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        action_shm.copy_(act.cpu())
        action_flag.value = 1

    print("[INFER] Stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# rsl_rl backend (pick_place)
# ──────────────────────────────────────────────────────────────────────────────

def _build_actor_mlp(obs_dim: int, action_dim: int, hidden_dims, activation: str):
    """Rebuild rsl_rl ActorCritic's actor MLP (no rsl_rl runtime dependency).

    Matches rsl_rl's layer layout: Linear/act per hidden dim, then a final Linear.
    Linear layers therefore sit at Sequential indices 0, 2, 4, ... so the stripped
    state_dict keys ("0.weight", "2.weight", ...) line up with this module.
    """
    import torch.nn as nn

    act_map = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh,
               "selu": nn.SELU, "crelu": nn.ReLU, "lrelu": nn.LeakyReLU}
    act_cls = act_map[str(activation).lower()]

    layers = []
    in_dim = int(obs_dim)
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, int(h)))
        layers.append(act_cls())
        in_dim = int(h)
    layers.append(nn.Linear(in_dim, int(action_dim)))
    return nn.Sequential(*layers)


def _run_rsl_rl_inference(
    obs_shm: torch.Tensor,
    action_shm: torch.Tensor,
    obs_flag: RawValue,
    action_flag: RawValue,
    stop_event: mp.Event,
    *,
    cfg: dict,
    checkpoint: str,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
) -> None:
    """Deterministic inference for an rsl_rl PPO policy.

    Loads only the actor MLP from the checkpoint's `model_state_dict` and emits the
    actor mean. Assumes `actor_obs_normalization=False` (the training default for
    pick_place); if it is on, the running obs normalizer must also be loaded.
    """
    hidden_dims = cfg.get("hidden_dims", [256, 128, 64])
    activation = cfg.get("activation", "elu")
    obs_dim = int(cfg.get("obs_dim", obs_dim))
    action_dim = int(cfg.get("action_dim", action_dim))

    if cfg.get("actor_obs_normalization", False):
        print("[INFER][rsl_rl] WARNING: actor_obs_normalization=True is not handled; "
              "observations are fed to the actor without normalization.")

    actor = _build_actor_mlp(obs_dim, action_dim, hidden_dims, activation).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    actor_state = {k[len("actor."):]: v for k, v in state.items() if k.startswith("actor.")}
    if not actor_state:
        raise RuntimeError(
            "[INFER][rsl_rl] No 'actor.*' weights found in checkpoint; "
            f"available top-level keys: {list(state.keys())[:8]}..."
        )
    missing, unexpected = actor.load_state_dict(actor_state, strict=False)
    if missing or unexpected:
        print(f"[INFER][rsl_rl] WARNING: missing={missing} unexpected={unexpected}")
    actor.eval()

    print("[INFER] Ready.")
    action_flag.value = 2  # warm-up done signal

    # ── Inference loop (spin-wait) ────────────────────────────────────────────
    while not stop_event.is_set():
        if obs_flag.value != 1:
            continue  # ← pure spin: no sleep allowed
        obs_flag.value = 0

        with torch.inference_mode():
            obs_t = obs_shm.to(device)
            act = actor(obs_t)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        action_shm.copy_(act.cpu())
        action_flag.value = 1

    print("[INFER] Stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# Handle
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceProcessHandle:
    proc: mp.Process
    stop_event: mp.Event

    def wait_ready(self, action_flag: RawValue) -> None:
        print("[INFO] Waiting for inference warm-up ...")
        while action_flag.value != 2:
            time.sleep(0.05)
        action_flag.value = 0
        print("[INFO] Inference process ready.")

    def shutdown(self) -> None:
        self.stop_event.set()
        self.proc.join(timeout=3)


def start_inference_process(
    obs_shm: torch.Tensor,
    action_shm: torch.Tensor,
    obs_flag: RawValue,
    action_flag: RawValue,
    *,
    checkpoint: str,
    cfg_path: str,
    obs_dim: int,
    action_dim: int,
    device: str,
) -> InferenceProcessHandle:
    stop_event = mp.Event()
    proc = mp.Process(
        target=run_inference_process,
        args=(obs_shm, action_shm, obs_flag, action_flag, stop_event),
        kwargs=dict(
            checkpoint=checkpoint,
            cfg_path=cfg_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device_str=device,
        ),
        daemon=True,
    )
    proc.start()
    return InferenceProcessHandle(proc=proc, stop_event=stop_event)