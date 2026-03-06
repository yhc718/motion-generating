"""Unified wrist-conditioned motion diffusion testing script.

Compatible with ALL wrist conditioning parameterizations:
  - absolute       (14D): world-frame positions + orientations
  - rh_first       (21D): per-hand first-frame relative + hand comm
  - rh_first_global_z (21D): same as rh_first but z is global height
  - absrel         (35D): absolute + rh_first_base + hand comm
  - ours           (37D): relative deltas + global-z + hand comm + first-frame-relative base

Auto-detects wrist_cond_param from checkpoint when set to "auto".
Supports x0, epsilon, and v_prediction model types.

Usage:
  # Auto-detect everything from checkpoint:
  python test_diffusion_unified.py --checkpoint-dir ./experiments/AMASS/v13/checkpoints_5000/

  # Explicit wrist cond param:
  python test_diffusion_unified.py --checkpoint-dir ./experiments/AMASS/v6/checkpoints_40000/ \\
      --wrist-cond-param absrel

  # Use a different NPZ input:
  python test_diffusion_unified.py --input-npz ./my_data.npz \\
      --checkpoint-dir ./experiments/AMASS/v5/checkpoints_40000/

  # Disable visualization:
  python test_diffusion_unified.py --checkpoint-dir ./experiments/AMASS/v13/checkpoints_5000/ \\
      --no-visualize
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Make sure src/ is on path so egoallo can be imported
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from egoallo import fncsmpl, network
from egoallo.inference_utils import load_denoiser
from egoallo.sampling import (
    CosineNoiseScheduleConstants,
    run_sampling_with_stitching,
    quadratic_ts,
)
from egoallo.transforms import SE3, SO3


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Arguments
# ═══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class Args:
    """Unified testing arguments for wrist-conditioned motion diffusion."""

    # -- Input / model --
    input_npz: Path = Path("./flute_pass_1_stageii.npz")
    """StageII NPZ file with SMPL-H parameters (root_orient, trans, pose_body, betas)."""

    checkpoint_dir: Path = Path("./experiments/AMASS/v9/checkpoints_25000/")
    """Checkpoint directory containing model.safetensors / ema_model.safetensors."""

    use_ema: bool = True
    """Use EMA weights for inference."""

    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")
    """SMPL-H body model path."""

    # -- Output / sampling --
    traj_length: int = 0
    """Trajectory length (frames). 0 = use all available frames."""

    num_samples: int = 1
    """Number of samples to generate."""

    save_output: bool = True
    """Save NPZ + YAML outputs."""

    visualize: bool = True
    """Launch viser visualization."""

    viser_port: int = 8080
    """Port for viser visualization server."""

    output_dir: Path = Path("./egoallo_outputs")
    """Directory for saving results."""

    # -- Sampling schedule --
    sampling_schedule: Literal[
        "quadratic", "linear", "cosine_flow", "logit_normal", "power",
    ] = "quadratic"
    """Inference timestep schedule."""

    sampling_steps: int = 32
    """Number of denoising steps (ignored for quadratic schedule)."""

    schedule_power: float = 3.0
    """Power schedule exponent (only for --sampling-schedule=power)."""

    logit_normal_mu: float = 0.0
    """Logit-normal schedule mu."""

    logit_normal_sigma: float = 1.0
    """Logit-normal schedule sigma."""

    # -- Guidance --
    guidance_mode: Literal["off", "global", "local"] = "off"
    """Guidance optimization mode."""

    guidance_inner: bool = False
    """Run guidance between denoising steps."""

    guidance_post: bool = False
    """Run guidance after diffusion."""

    # -- Wrist conditioning --
    wrist_cond_param: Literal[
        "auto", "absolute", "rh_first", "rh_first_global_z", "absrel", "ours",
    ] = "auto"
    """Wrist conditioning parameterization. 'auto' infers from checkpoint."""

    use_rh_frame: bool = True
    """Transform root to right-hand frame (matching training convention)."""

    # -- Diagnostics --
    run_single_step_diagnostic: bool = True
    """Run single-step denoiser diagnostic at several noise levels."""

    diagnostic_noise_levels: Tuple[int, ...] = (1, 10, 50, 100, 500)
    """Noise levels for single-step diagnostic."""

    ddim_eta: float = 0.8
    """Stochasticity parameter for DDIM sampling (0 = deterministic)."""


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    return x[None, :] if x.ndim == 1 else x


def load_stageii_smplh_params(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load StageII SMPL-H parameters from NPZ.

    Returns (root_orient, transl, pose_body, betas).
    """
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)

    root_orient = (
        _ensure_2d(d["root_orient"])
        if "root_orient" in keys
        else _ensure_2d(d["poses"])[:, :3]
    )
    transl = _ensure_2d(d["trans"]) if "trans" in keys else None
    if transl is None:
        raise KeyError("Missing 'trans' in NPZ")
    pose_body = (
        _ensure_2d(d["pose_body"])
        if "pose_body" in keys
        else _ensure_2d(d["poses"])[:, 3:66]
    )
    betas = (
        _ensure_2d(d["betas"])
        if "betas" in keys
        else np.zeros((1, 16), dtype=np.float32)
    )
    return root_orient, transl, pose_body, betas


# ═══════════════════════════════════════════════════════════════════════════════
# Wrist conditioning builders (inference-time, non-batched)
#
# All builders take:
#   positions:    (T, 2, 3)   -- world-frame wrist positions [left, right]
#   orientations: (T, 2, 4)   -- world-frame wrist quaternions (wxyz) [left, right]
# ═══════════════════════════════════════════════════════════════════════════════

def _build_wrist_cond_absolute(
    positions: torch.Tensor,
    orientations: torch.Tensor,
) -> torch.Tensor:
    """Absolute world-frame: [left_pos(3), right_pos(3), left_quat(4), right_quat(4)] = 14D."""
    T = positions.shape[0]
    return torch.cat(
        [positions.reshape(T, 6), orientations.reshape(T, 8)], dim=-1
    )


def _build_wrist_cond_rh_first_base(
    positions: torch.Tensor,
    orientations: torch.Tensor,
    keep_global_z: bool,
) -> torch.Tensor:
    """Per-hand first-frame relative (without hand comm) = 14D.

    Each hand's trajectory is expressed relative to its pose at frame 0.
    """
    T = positions.shape[0]

    # First-frame anchors
    lh_pos0 = positions[0, 0, :]  # (3,)
    rh_pos0 = positions[0, 1, :]  # (3,)
    lh_q0 = orientations[0, 0, :]  # (4,)
    rh_q0 = orientations[0, 1, :]  # (4,)

    lh_inv = SO3(lh_q0).inverse()
    rh_inv = SO3(rh_q0).inverse()

    # Relative positions
    lh_pos_rel = lh_inv @ (positions[:, 0, :] - lh_pos0[None, :])  # (T, 3)
    rh_pos_rel = rh_inv @ (positions[:, 1, :] - rh_pos0[None, :])  # (T, 3)

    if keep_global_z:
        lh_pos_rel = torch.cat(
            [lh_pos_rel[:, :2], positions[:, 0, 2:3]], dim=-1
        )
        rh_pos_rel = torch.cat(
            [rh_pos_rel[:, :2], positions[:, 1, 2:3]], dim=-1
        )

    # Relative orientations
    lh_ori_rel = (lh_inv @ SO3(orientations[:, 0, :])).wxyz  # (T, 4)
    rh_ori_rel = (rh_inv @ SO3(orientations[:, 1, :])).wxyz  # (T, 4)

    return torch.cat([lh_pos_rel, rh_pos_rel, lh_ori_rel, rh_ori_rel], dim=-1)


def _build_wrist_cond_hand_comm(
    positions: torch.Tensor,
    orientations: torch.Tensor,
) -> torch.Tensor:
    """Hand communication: right wrist in left frame = 7D."""
    lr_pos_world = positions[:, 1, :] - positions[:, 0, :]  # (T, 3)
    lh_inv = SO3(orientations[:, 0, :]).inverse()
    lr_pos_left = lh_inv @ lr_pos_world  # (T, 3)
    lr_ori_left = (lh_inv @ SO3(orientations[:, 1, :])).wxyz  # (T, 4)
    return torch.cat([lr_pos_left, lr_ori_left], dim=-1)


def _build_wrist_cond_hand_comm_deltas(
    positions: torch.Tensor,
    orientations: torch.Tensor,
) -> torch.Tensor:
    """Hand communication deltas (frame-to-frame) = 7D."""
    T = positions.shape[0]
    out = torch.zeros((T, 7), device=positions.device, dtype=positions.dtype)

    # Compute LR relative first
    lr_pos_world = positions[:, 1, :] - positions[:, 0, :]
    lh_quat = orientations[:, 0, :]  # (T, 4)
    rh_quat = orientations[:, 1, :]  # (T, 4)
    lr_pos_left = SO3(lh_quat).inverse() @ lr_pos_world
    lr_ori = (SO3(lh_quat).inverse() @ SO3(rh_quat)).wxyz

    if T > 1:
        # Position deltas
        delta_pos = lr_pos_left[1:] - lr_pos_left[:-1]  # (T-1, 3)
        prev_lr_ori = SO3(lr_ori[:-1])
        out[1:, :3] = prev_lr_ori.inverse() @ delta_pos
        # Orientation deltas
        out[1:, 3:] = (prev_lr_ori.inverse() @ SO3(lr_ori[1:])).wxyz
    return out


def _build_wrist_cond_ours(
    positions: torch.Tensor,
    orientations: torch.Tensor,
) -> torch.Tensor:
    """'ours' parameterization = 37D.

    Combines local motion deltas with absolute spatial context, inspired by
    the original cond_param 'ours' which pairs frame-to-frame deltas with
    a canonicalized rotation and global height.

    Layout: [pos_deltas(6), ori_deltas(8), global_z(2), hand_comm(7), rh_first_base(14)]
    """
    T = positions.shape[0]

    pos_deltas = torch.zeros_like(positions)       # (T, 2, 3)
    ori_deltas = torch.zeros_like(orientations)    # (T, 2, 4)
    ori_deltas[..., 0] = 1.0  # identity quaternion at t=0

    if T > 1:
        for hand_idx in range(2):
            pos_delta_world = positions[1:, hand_idx, :] - positions[:-1, hand_idx, :]
            prev_ori = SO3(orientations[:-1, hand_idx, :])
            pos_deltas[1:, hand_idx, :] = prev_ori.inverse() @ pos_delta_world
            ori_deltas[1:, hand_idx, :] = (
                prev_ori.inverse() @ SO3(orientations[1:, hand_idx, :])
            ).wxyz

    global_z = positions[:, :, 2]  # (T, 2)

    # Absolute hand communication (inter-hand relationship each frame).
    hand_comm = _build_wrist_cond_hand_comm(positions, orientations)  # (T, 7)

    # First-frame-relative positions + orientations per hand
    # (absolute spatial context, analogous to canonicalized rotation).
    rh_first_base = _build_wrist_cond_rh_first_base(
        positions, orientations, keep_global_z=True
    )  # (T, 14)

    return torch.cat(
        [
            pos_deltas.reshape(T, 6),
            ori_deltas.reshape(T, 8),
            global_z,
            hand_comm,
            rh_first_base,
        ],
        dim=-1,
    )


def build_wrist_conditioning(
    positions: torch.Tensor,
    orientations: torch.Tensor,
    wrist_cond_param: str,
) -> torch.Tensor:
    """Dispatch to the correct wrist conditioning builder.

    Args:
        positions:    (T, 2, 3) world-frame wrist positions [left, right].
        orientations: (T, 2, 4) world-frame wrist quats wxyz [left, right].
        wrist_cond_param: one of {absolute, rh_first, rh_first_global_z, absrel, ours}.

    Returns:
        (T, D) wrist conditioning tensor.
    """
    if wrist_cond_param == "absolute":
        return _build_wrist_cond_absolute(positions, orientations)
    if wrist_cond_param == "rh_first":
        base = _build_wrist_cond_rh_first_base(positions, orientations, keep_global_z=False)
        hand_comm = _build_wrist_cond_hand_comm(positions, orientations)
        return torch.cat([base, hand_comm], dim=-1)
    if wrist_cond_param == "rh_first_global_z":
        base = _build_wrist_cond_rh_first_base(positions, orientations, keep_global_z=True)
        hand_comm = _build_wrist_cond_hand_comm(positions, orientations)
        return torch.cat([base, hand_comm], dim=-1)
    if wrist_cond_param == "absrel":
        absolute = _build_wrist_cond_absolute(positions, orientations)
        rh_first_base = _build_wrist_cond_rh_first_base(positions, orientations, keep_global_z=True)
        hand_comm = _build_wrist_cond_hand_comm(positions, orientations)
        return torch.cat([absolute, rh_first_base, hand_comm], dim=-1)
    if wrist_cond_param == "ours":
        return _build_wrist_cond_ours(positions, orientations)
    raise ValueError(f"Unknown wrist_cond_param='{wrist_cond_param}'")


# Mapping from wrist_cond_param to expected raw dimension (before Fourier encoding)
WRIST_COND_DIMS = {
    "absolute": 14,
    "rh_first": 21,
    "rh_first_global_z": 21,
    "absrel": 35,
    "ours": 37,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Wrist conditioning auto-detection
# ═══════════════════════════════════════════════════════════════════════════════

def _infer_wrist_cond_dim(denoiser: network.EgoDenoiser) -> int:
    """Read the raw wrist conditioning dim from checkpoint weights."""
    if not denoiser.config.use_wrist_conditioning:
        return 0
    factor = 1 + 2 * denoiser.config.fourier_enc_freqs
    weight = denoiser.latent_from_cond.weight
    if weight.shape[1] % factor != 0:
        raise ValueError(
            f"latent_from_cond.in={weight.shape[1]} is not divisible by Fourier factor {factor}"
        )
    return weight.shape[1] // factor


def _dim_to_param(dim: int) -> str:
    """Infer wrist_cond_param from raw dimension."""
    for param, d in WRIST_COND_DIMS.items():
        if d == dim:
            return param
    raise ValueError(
        f"Cannot infer wrist_cond_param from wrist dim={dim}. "
        f"Known dims: {WRIST_COND_DIMS}. Pass --wrist-cond-param explicitly."
    )


def resolve_wrist_cond_param(
    requested: str,
    denoiser: network.EgoDenoiser,
) -> str:
    """Resolve the wrist conditioning parameterization.

    If 'auto', tries config attribute first, then infers from weight dimensions.
    If explicit, validates against the checkpoint.
    """
    if not denoiser.config.use_wrist_conditioning:
        return "none"

    inferred_dim = _infer_wrist_cond_dim(denoiser)

    if requested == "auto":
        # 1. Try config attribute
        cfg_param = getattr(denoiser.config, "wrist_cond_param", None)
        if cfg_param is not None and cfg_param in WRIST_COND_DIMS:
            if WRIST_COND_DIMS[cfg_param] == inferred_dim:
                return cfg_param
        # 2. Infer from dimension
        # For ambiguous dims (e.g., rh_first vs rh_first_global_z both 21D),
        # we prefer the config's value if it matches the dim.
        candidates = [p for p, d in WRIST_COND_DIMS.items() if d == inferred_dim]
        if not candidates:
            raise ValueError(
                f"Cannot infer wrist_cond_param from checkpoint dim={inferred_dim}."
            )
        if cfg_param in candidates:
            return cfg_param
        return candidates[0]

    # Explicit: validate against checkpoint
    expected_dim = WRIST_COND_DIMS.get(requested)
    if expected_dim is None:
        raise ValueError(f"Unknown wrist_cond_param='{requested}'")
    if expected_dim != inferred_dim:
        raise ValueError(
            f"Requested wrist_cond_param='{requested}' needs dim={expected_dim}, "
            f"but checkpoint expects dim={inferred_dim}."
        )
    return requested


# ═══════════════════════════════════════════════════════════════════════════════
# Prediction type helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_prediction_type(model: network.EgoDenoiser) -> str:
    return getattr(model.config, "prediction_type", "x0")


def convert_model_output_to_x0(
    raw_output: torch.Tensor,
    x_t_packed: torch.Tensor,
    alpha_bar_t_val: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    """Convert raw network output to x_0 prediction."""
    if prediction_type == "x0":
        return raw_output
    if prediction_type == "epsilon":
        alpha_safe = alpha_bar_t_val.clamp(min=1e-3)
        return (
            x_t_packed - torch.sqrt(1.0 - alpha_bar_t_val) * raw_output
        ) / torch.sqrt(alpha_safe)
    if prediction_type == "v_prediction":
        return (
            torch.sqrt(alpha_bar_t_val) * x_t_packed
            - torch.sqrt(1.0 - alpha_bar_t_val) * raw_output
        )
    raise ValueError(f"Unknown prediction_type: {prediction_type}")


def ddim_reverse_step(
    x_t_packed: torch.Tensor,
    x_0_packed_pred: torch.Tensor,
    alpha_bar_t: torch.Tensor,
    alpha_bar_t_next: torch.Tensor,
    eta: float = 0.8,
) -> torch.Tensor:
    """Single DDIM reverse step: x_t -> x_{t_next}."""
    if alpha_bar_t_next > 0:
        sigma = eta * torch.sqrt(
            (1.0 - alpha_bar_t_next) / (1.0 - alpha_bar_t)
            * (1.0 - alpha_bar_t / alpha_bar_t_next)
        )
    else:
        sigma = torch.tensor(0.0, device=x_t_packed.device)

    return (
        torch.sqrt(alpha_bar_t_next) * x_0_packed_pred
        + (
            torch.sqrt(1 - alpha_bar_t_next - sigma ** 2)
            * (x_t_packed - torch.sqrt(alpha_bar_t) * x_0_packed_pred)
            / torch.sqrt(1 - alpha_bar_t + 1e-1)
        )
        + sigma * torch.randn_like(x_0_packed_pred)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Sampling loops
# ═══════════════════════════════════════════════════════════════════════════════

def _run_sampling_eps_or_vpred(
    denoiser: network.EgoDenoiser,
    body_model: fncsmpl.SmplhModel,
    wrist_cond: torch.Tensor | None,
    Ts_world_cpf: torch.Tensor,
    custom_ts: np.ndarray,
    num_samples: int,
    prediction_type: str,
    device: torch.device,
    eta: float = 0.8,
) -> network.EgoDenoiseTraj:
    """DDIM sampling with epsilon / v_prediction output -> x_0 conversion."""
    from tqdm import tqdm

    noise_constants = CosineNoiseScheduleConstants.compute(timesteps=1000).to(device)
    alpha_bar = noise_constants.alpha_bar_t

    T_cpf_tm1_cpf_t = (
        SE3(Ts_world_cpf[..., :-1, :]).inverse() @ SE3(Ts_world_cpf[..., 1:, :])
    ).wxyz_xyz

    seq_len = Ts_world_cpf.shape[0] - 1
    d_state = denoiser.get_d_state()
    x_t_packed = torch.randn((num_samples, seq_len, d_state), device=device)

    window_size = 128
    overlap_size = 32
    canonical_overlap_weights = (
        torch.from_numpy(
            np.minimum(
                overlap_size,
                np.minimum(np.arange(1, seq_len + 1), np.arange(1, seq_len + 1)[::-1]),
            ) / overlap_size
        ).to(device, torch.float32)
    )

    for i in tqdm(range(len(custom_ts) - 1), desc=f"Sampling ({prediction_type})"):
        t = custom_ts[i]
        t_next = custom_ts[i + 1]

        with torch.inference_mode():
            raw_output = torch.zeros_like(x_t_packed)
            overlap_weights = torch.zeros((1, seq_len, 1), device=device)

            for start_t in range(0, seq_len, window_size - overlap_size):
                end_t = min(start_t + window_size, seq_len)
                w = canonical_overlap_weights[None, : end_t - start_t, None]
                overlap_weights[:, start_t:end_t, :] += w

                out = denoiser.forward(
                    x_t_packed[:, start_t:end_t, :],
                    torch.tensor([t], device=device).expand(num_samples),
                    T_cpf_tm1_cpf_t=T_cpf_tm1_cpf_t[None, start_t:end_t, :].repeat(
                        num_samples, 1, 1
                    ),
                    T_world_cpf=Ts_world_cpf[None, start_t + 1 : end_t + 1, :].repeat(
                        num_samples, 1, 1
                    ),
                    project_output_rotmats=False,
                    hand_positions_wrt_cpf=None,
                    wrist_positions=(
                        wrist_cond[None, start_t:end_t, :].repeat(num_samples, 1, 1)
                        if wrist_cond is not None else None
                    ),
                    mask=None,
                )
                raw_output[:, start_t:end_t, :] += out * w

            raw_output /= overlap_weights

            # Convert to x_0
            x_0_pred = convert_model_output_to_x0(
                raw_output, x_t_packed, alpha_bar[t], prediction_type
            ).clamp(-10.0, 10.0)
            x_0_pred = network.EgoDenoiseTraj.unpack(
                x_0_pred,
                include_hands=denoiser.config.include_hands,
                project_rotmats=True,
            ).pack()

        if torch.any(torch.isnan(x_0_pred)):
            print(f"  [WARNING] NaN detected at step {i}")

        x_t_packed = ddim_reverse_step(
            x_t_packed, x_0_pred, alpha_bar[t], alpha_bar[t_next], eta=eta
        )

    return network.EgoDenoiseTraj.unpack(
        x_t_packed, include_hands=denoiser.config.include_hands
    )


def run_sampling(
    denoiser: network.EgoDenoiser,
    body_model: fncsmpl.SmplhModel,
    wrist_cond: torch.Tensor | None,
    Ts_world_cpf: torch.Tensor,
    custom_ts: np.ndarray,
    num_samples: int,
    device: torch.device,
    guidance_mode: str,
    guidance_inner: bool,
    guidance_post: bool,
    eta: float = 0.8,
) -> network.EgoDenoiseTraj:
    """Prediction-type-aware sampling dispatcher.

    For x0 models -> run_sampling_with_stitching (original pipeline).
    For epsilon / v_prediction -> custom DDIM loop.
    """
    prediction_type = _resolve_prediction_type(denoiser)

    if prediction_type == "x0":
        return run_sampling_with_stitching(
            denoiser,
            body_model=body_model,
            guidance_mode=guidance_mode,
            guidance_inner=guidance_inner,
            guidance_post=guidance_post,
            Ts_world_cpf=Ts_world_cpf,
            hamer_detections=None,
            aria_detections=None,
            num_samples=num_samples,
            device=device,
            floor_z=0.0,
            wrist_positions=wrist_cond,
        )

    print(f"  Using prediction-type-aware DDIM loop (type='{prediction_type}')")
    return _run_sampling_eps_or_vpred(
        denoiser=denoiser,
        body_model=body_model,
        wrist_cond=wrist_cond,
        Ts_world_cpf=Ts_world_cpf,
        custom_ts=custom_ts,
        num_samples=num_samples,
        prediction_type=prediction_type,
        device=device,
        eta=eta,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Timestep schedule builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_schedule(args: Args) -> np.ndarray:
    """Build the denoising timestep schedule."""
    if args.sampling_schedule == "quadratic":
        ts = quadratic_ts()
    elif args.sampling_schedule == "linear":
        ts = np.linspace(1000, 0, args.sampling_steps + 1).astype(int)
    elif args.sampling_schedule == "cosine_flow":
        # More steps at high noise
        u = np.linspace(0, 1, args.sampling_steps + 1)
        ts = (1000 * (1.0 - np.cos(u * np.pi / 2))).astype(int)
        ts = 1000 - ts
    elif args.sampling_schedule == "logit_normal":
        from scipy.special import expit  # type: ignore
        u = np.linspace(-4, 4, args.sampling_steps + 1)
        w = expit(args.logit_normal_mu + args.logit_normal_sigma * u)
        ts = (1000 * (1.0 - w)).astype(int)
    elif args.sampling_schedule == "power":
        u = np.linspace(0, 1, args.sampling_steps + 1)
        ts = (1000 * (1.0 - u ** args.schedule_power)).astype(int)
    else:
        raise ValueError(f"Unknown schedule: {args.sampling_schedule}")
    return np.unique(np.clip(ts, 0, 999))[::-1]


# ═══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_step_diagnostic(
    denoiser: network.EgoDenoiser,
    gt_x_0: network.EgoDenoiseTraj,
    noise_constants: CosineNoiseScheduleConstants,
    Ts_world_cpf: torch.Tensor,
    wrist_cond: torch.Tensor | None,
    device: torch.device,
    levels: Tuple[int, ...],
    gt_root_trans_ref: torch.Tensor,
    gt_body_rotmats_ref: torch.Tensor,
    gt_root_orient_ref: torch.Tensor,
) -> None:
    """Diagnostic: feed GT + noise to the model and measure reconstruction error."""
    pred_type = _resolve_prediction_type(denoiser)
    gt_packed = gt_x_0.pack().unsqueeze(0)
    T = gt_packed.shape[1]
    alpha_bar = noise_constants.alpha_bar_t
    rad2deg = 180.0 / np.pi

    print(f"\n{'='*70}")
    print(f"[DIAGNOSTIC] Single-step denoiser test (prediction_type={pred_type})")
    print(f"{'='*70}")
    print(f"  {'t':>5s}  {'SNR(dB)':>8s}  {'root_trans':>11s}  {'root_orient':>12s}  {'body_rotmat':>12s}")
    print(f"  {'-'*5}  {'-'*8}  {'-'*11}  {'-'*12}  {'-'*12}")

    for t_val in levels:
        t = torch.tensor([t_val], device=device)
        a_bar = alpha_bar[t_val]
        eps = torch.randn_like(gt_packed)
        x_t = torch.sqrt(a_bar) * gt_packed + torch.sqrt(1.0 - a_bar) * eps

        with torch.inference_mode():
            # Build identity SE3 for T_cpf_tm1_cpf_t: [w=1, x=0, y=0, z=0, tx=0, ty=0, tz=0]
            identity_se3 = torch.zeros(1, T, 7, device=device)
            identity_se3[:, :, 0] = 1.0  # w component of identity quaternion
            raw = denoiser(
                x_t,
                t,
                T_cpf_tm1_cpf_t=identity_se3,
                T_world_cpf=Ts_world_cpf[None, 1 : T + 1, :],
                project_output_rotmats=(pred_type == "x0"),
                hand_positions_wrt_cpf=None,
                wrist_positions=(
                    wrist_cond[None, :T, :] if wrist_cond is not None else None
                ),
                mask=None,
            )
            x0_pred = convert_model_output_to_x0(raw, x_t, a_bar, pred_type).clamp(-10.0, 10.0)

        x0_traj = network.EgoDenoiseTraj.unpack(
            x0_pred,
            include_hands=denoiser.config.include_hands,
            project_rotmats=True,
        )

        root_err = torch.norm(x0_traj.root_trans[0] - gt_root_trans_ref, dim=-1).mean().item()

        # Body rotation error
        R_err = torch.matmul(
            x0_traj.body_rotmats[0], gt_body_rotmats_ref.transpose(-1, -2)
        )
        trace = R_err[..., 0, 0] + R_err[..., 1, 1] + R_err[..., 2, 2]
        body_angle_deg = (
            torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)).mean() * rad2deg
        ).item()

        # Root orient error
        R_root_err = torch.matmul(
            x0_traj.root_orient[0], gt_root_orient_ref.transpose(-1, -2)
        )
        trace_root = R_root_err[..., 0, 0] + R_root_err[..., 1, 1] + R_root_err[..., 2, 2]
        root_orient_deg = (
            torch.acos(((trace_root - 1.0) / 2.0).clamp(-1.0, 1.0)).mean() * rad2deg
        ).item()

        snr_db = 10 * np.log10(a_bar.item() / (1 - a_bar.item() + 1e-8))
        print(
            f"  {t_val:5d}  {snr_db:+8.1f}  {root_err:11.4f}m  {root_orient_deg:11.2f}°  {body_angle_deg:11.2f}°"
        )

    print(f"{'='*70}")
    print("  t=1 error large -> model hasn't seen this data (or not converged)")
    print("  t=1 small, t=500 large -> normal (denoising from high noise is hard)")
    print("  all small but sampling bad -> issue in multi-step sampling")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

JOINT_NAMES = [
    "leftUpLeg", "rightUpLeg", "spine", "leftLeg", "rightLeg",
    "spine1", "leftFoot", "rightFoot", "spine2", "leftToeBase",
    "rightToeBase", "neck", "leftShoulder", "rightShoulder", "head",
    "leftArm", "rightArm", "leftForeArm", "rightForeArm",
    "leftHand", "rightHand",
]

WRIST_LEFT_IDX = 19
WRIST_RIGHT_IDX = 20


def evaluate_generation(
    traj: network.EgoDenoiseTraj,
    body_model: fncsmpl.SmplhModel,
    gt_posed: fncsmpl.SmplhShapedAndPosed,
    gt_root_position: torch.Tensor,
    verbose: bool = True,
) -> dict[str, float]:
    """Evaluate generated trajectory against ground truth.

    Returns dict with position and rotation errors.
    """
    rad2deg = 180.0 / np.pi

    # Apply FK
    pred_posed = traj.apply_to_body(body_model)

    # Wrist position errors
    pred_lw = pred_posed.Ts_world_joint[0, :, WRIST_LEFT_IDX, 4:7]
    pred_rw = pred_posed.Ts_world_joint[0, :, WRIST_RIGHT_IDX, 4:7]
    gt_lw = gt_posed.Ts_world_joint[:, WRIST_LEFT_IDX, 4:7]
    gt_rw = gt_posed.Ts_world_joint[:, WRIST_RIGHT_IDX, 4:7]

    left_err = torch.norm(pred_lw - gt_lw, dim=-1).mean().item()
    right_err = torch.norm(pred_rw - gt_rw, dim=-1).mean().item()

    # Root position error
    root_err = torch.norm(traj.root_trans[0] - gt_root_position, dim=-1).mean().item()

    results = {
        "left_wrist_pos_m": left_err,
        "right_wrist_pos_m": right_err,
        "root_pos_m": root_err,
    }

    if verbose:
        print(f"\n{'='*60}")
        print("Evaluation Results")
        print(f"{'='*60}")
        print(f"  Left wrist pos error:  {left_err:.4f} m")
        print(f"  Right wrist pos error: {right_err:.4f} m")
        print(f"  Root pos error:        {root_err:.4f} m")

        # Per-joint FK position error
        print(f"\n  {'Joint':<18} {'Mean L2':>10} {'Max L2':>10}")
        print(f"  {'-'*18} {'-'*10} {'-'*10}")
        for j in range(21):
            gt_pos = gt_posed.Ts_world_joint[:, j, 4:7]
            pred_pos = pred_posed.Ts_world_joint[0, :, j, 4:7]
            err = torch.norm(pred_pos - gt_pos, dim=-1)
            marker = " *" if j in (WRIST_LEFT_IDX, WRIST_RIGHT_IDX) else ""
            print(f"  {JOINT_NAMES[j]:<18} {err.mean().item():>10.4f} {err.max().item():>10.4f}{marker}")
            results[f"joint_{JOINT_NAMES[j]}_pos_m"] = err.mean().item()

        # Wrist orientation error
        pred_lq = pred_posed.Ts_world_joint[0, :, WRIST_LEFT_IDX, :4]
        pred_rq = pred_posed.Ts_world_joint[0, :, WRIST_RIGHT_IDX, :4]
        gt_lq = gt_posed.Ts_world_joint[:, WRIST_LEFT_IDX, :4]
        gt_rq = gt_posed.Ts_world_joint[:, WRIST_RIGHT_IDX, :4]

        def _quat_angle(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            p = p / p.norm(dim=-1, keepdim=True)
            g = g / g.norm(dim=-1, keepdim=True)
            return 2.0 * torch.acos(torch.sum(p * g, dim=-1).abs().clamp(-1, 1))

        lr_rot = _quat_angle(pred_lq, gt_lq).mean().item()
        rr_rot = _quat_angle(pred_rq, gt_rq).mean().item()
        results["left_wrist_rot_deg"] = lr_rot * rad2deg
        results["right_wrist_rot_deg"] = rr_rot * rad2deg
        print(f"\n  Left wrist rot error:  {lr_rot * rad2deg:.2f}°")
        print(f"  Right wrist rot error: {rr_rot * rad2deg:.2f}°")
        print(f"{'='*60}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(args: Args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load SMPL-H body model
    # ------------------------------------------------------------------
    body_model = fncsmpl.SmplhModel.load(args.smplh_npz_path).to(device)
    print(f"[1/7] SMPLH loaded from {args.smplh_npz_path}")

    # ------------------------------------------------------------------
    # 2. Load input data
    # ------------------------------------------------------------------
    root_orient_np, transl_np, body_pose_np, betas_np = load_stageii_smplh_params(
        args.input_npz
    )
    n_frames = root_orient_np.shape[0]
    actual_length = min(args.traj_length, n_frames) if args.traj_length > 0 else n_frames
    if actual_length != args.traj_length and args.traj_length > 0:
        print(f"  Adjusted traj_length: {args.traj_length} -> {actual_length}")
    args = dataclasses.replace(args, traj_length=actual_length)
    print(f"[2/7] Loaded {n_frames} frames from {args.input_npz} (using {actual_length})")

    # Prepare betas
    if betas_np.shape[0] > 1:
        betas_np = betas_np[:1]
    betas_np = betas_np.astype(np.float32)
    if betas_np.shape[1] < 16:
        betas_np = np.concatenate(
            [betas_np, np.zeros((1, 16 - betas_np.shape[1]), dtype=np.float32)],
            axis=1,
        )
    betas = torch.from_numpy(betas_np).to(device)  # (1, 16)

    # Tensors
    root_orient_t = torch.from_numpy(
        root_orient_np[:actual_length].astype(np.float32)
    ).to(device)
    body_pose_t = torch.from_numpy(
        body_pose_np[:actual_length].astype(np.float32)
    ).to(device)
    transl_t = torch.from_numpy(
        transl_np[:actual_length].astype(np.float32)
    ).to(device)

    root_orient_q = SO3.exp(root_orient_t).wxyz  # (T, 4)
    body_quats = SO3.exp(body_pose_t.reshape(actual_length, 21, 3)).wxyz  # (T, 21, 4)

    # ------------------------------------------------------------------
    # 3. FK to get GT joint positions
    # ------------------------------------------------------------------
    shaped = body_model.with_shape(betas)
    root_offset = shaped.root_offset.squeeze()
    if root_offset.dim() > 1:
        root_offset = root_offset.view(-1)
    gt_root_trans = transl_t + root_offset.unsqueeze(0)  # (T, 3)

    T_world_root = torch.cat([root_orient_q, gt_root_trans], dim=-1)  # (T, 7)
    gt_posed = shaped.with_pose_decomposed(
        T_world_root=T_world_root,
        body_quats=body_quats,
        left_hand_quats=None,
        right_hand_quats=None,
    )

    # Extract wrist positions and orientations
    wr_left = gt_posed.Ts_world_joint[:, WRIST_LEFT_IDX, 4:7]   # (T, 3)
    wr_right = gt_posed.Ts_world_joint[:, WRIST_RIGHT_IDX, 4:7]  # (T, 3)
    wq_left = gt_posed.Ts_world_joint[:, WRIST_LEFT_IDX, :4]    # (T, 4)
    wq_right = gt_posed.Ts_world_joint[:, WRIST_RIGHT_IDX, :4]   # (T, 4)
    wrists_pos = torch.stack([wr_left, wr_right], dim=1)   # (T, 2, 3)
    wrists_q = torch.stack([wq_left, wq_right], dim=1)     # (T, 2, 4)
    print(f"[3/7] FK complete. Wrist positions range:")
    print(f"  Left:  x=[{wr_left[:,0].min():.3f}, {wr_left[:,0].max():.3f}] "
          f"y=[{wr_left[:,1].min():.3f}, {wr_left[:,1].max():.3f}] "
          f"z=[{wr_left[:,2].min():.3f}, {wr_left[:,2].max():.3f}]")
    print(f"  Right: x=[{wr_right[:,0].min():.3f}, {wr_right[:,0].max():.3f}] "
          f"y=[{wr_right[:,1].min():.3f}, {wr_right[:,1].max():.3f}] "
          f"z=[{wr_right[:,2].min():.3f}, {wr_right[:,2].max():.3f}]")

    # ------------------------------------------------------------------
    # 4. Transform to right-hand frame (if enabled)
    # ------------------------------------------------------------------
    if args.use_rh_frame:
        R_world_rh = SO3(wq_right)
        R_rh_world = R_world_rh.inverse()
        gt_root_trans_model = R_rh_world @ (gt_root_trans - wr_right)
        gt_root_orient_model = (R_rh_world @ SO3.exp(root_orient_t)).as_matrix()
        print(f"[4/7] Transformed root to right-hand frame")
    else:
        gt_root_trans_model = gt_root_trans
        gt_root_orient_model = SO3.exp(root_orient_t).as_matrix()
        print(f"[4/7] Using world frame (no RH transform)")

    gt_body_rotmats = SO3.exp(
        body_pose_t.reshape(actual_length, 21, 3)
    ).as_matrix()  # (T, 21, 3, 3)

    # Dummy CPF trajectory
    Ts_world_cpf = torch.zeros((actual_length + 1, 7), device=device)
    Ts_world_cpf[:, 0] = 1.0   # identity rotation
    Ts_world_cpf[:, 6] = 1.6   # reasonable height

    # ------------------------------------------------------------------
    # 5. Load denoiser and resolve wrist conditioning
    # ------------------------------------------------------------------
    denoiser = load_denoiser(
        args.checkpoint_dir,
        use_ema=args.use_ema,
        wrist_cond_param=None,
    ).to(device)

    prediction_type = _resolve_prediction_type(denoiser)
    print(f"[5/7] Loaded denoiser from {args.checkpoint_dir}")
    print(f"  prediction_type     = {prediction_type}")
    print(f"  include_hands       = {denoiser.config.include_hands}")
    print(f"  d_state             = {denoiser.get_d_state()}")
    print(f"  use_wrist_cond      = {denoiser.config.use_wrist_conditioning}")

    wrist_cond: torch.Tensor | None = None
    resolved_param = "none"

    if denoiser.config.use_wrist_conditioning:
        inferred_dim = _infer_wrist_cond_dim(denoiser)
        resolved_param = resolve_wrist_cond_param(
            args.wrist_cond_param, denoiser
        )
        wrist_cond = build_wrist_conditioning(wrists_pos, wrists_q, resolved_param)

        # Validate dimension
        expected_dim = WRIST_COND_DIMS[resolved_param]
        assert wrist_cond.shape[-1] == expected_dim == inferred_dim, (
            f"Dim mismatch: built={wrist_cond.shape[-1]}, "
            f"expected={expected_dim}, checkpoint={inferred_dim}"
        )
        print(f"  wrist_cond_param    = {resolved_param} (requested={args.wrist_cond_param})")
        print(f"  wrist_cond dim      = {wrist_cond.shape[-1]}")
        print(f"  wrist_cond shape    = {wrist_cond.shape}")
    else:
        print("  (no wrist conditioning)")

    # ------------------------------------------------------------------
    # 5b. Single-step diagnostic
    # ------------------------------------------------------------------
    gt_x_0 = network.EgoDenoiseTraj(
        betas=betas.expand(actual_length, 16),
        body_rotmats=gt_body_rotmats,
        contacts=torch.zeros(actual_length, 21, device=device),
        root_trans=gt_root_trans_model,
        root_orient=gt_root_orient_model,
        hand_rotmats=(
            torch.eye(3, device=device)
            .unsqueeze(0).unsqueeze(0)
            .expand(actual_length, 30, 3, 3)
            if denoiser.config.include_hands else None
        ),
    )

    if args.run_single_step_diagnostic:
        run_single_step_diagnostic(
            denoiser=denoiser,
            gt_x_0=gt_x_0,
            noise_constants=CosineNoiseScheduleConstants.compute(1000).to(device),
            Ts_world_cpf=Ts_world_cpf,
            wrist_cond=wrist_cond,
            device=device,
            levels=args.diagnostic_noise_levels,
            gt_root_trans_ref=gt_root_trans_model,
            gt_body_rotmats_ref=gt_body_rotmats,
            gt_root_orient_ref=gt_root_orient_model,
        )

    # ------------------------------------------------------------------
    # 6. Build schedule and run sampling
    # ------------------------------------------------------------------
    custom_ts = build_schedule(args)
    print(f"[6/7] Schedule: {args.sampling_schedule} ({len(custom_ts)-1} steps)")
    print(f"  Timesteps (first 8): {custom_ts[:8]}")
    print(f"  Timesteps (last  8): {custom_ts[-8:]}")

    start_time = time.time()
    traj = run_sampling(
        denoiser=denoiser,
        body_model=body_model,
        wrist_cond=wrist_cond,
        Ts_world_cpf=Ts_world_cpf,
        custom_ts=custom_ts,
        num_samples=args.num_samples,
        device=device,
        guidance_mode=args.guidance_mode,
        guidance_inner=args.guidance_inner,
        guidance_post=args.guidance_post,
        eta=args.ddim_eta,
    )
    elapsed = time.time() - start_time
    print(f"  Sampling done in {elapsed:.2f}s")

    # ------------------------------------------------------------------
    # 6b. Transform back to world frame
    # ------------------------------------------------------------------
    if args.use_rh_frame:
        R_world_rh_batch = SO3(wq_right)  # (T,)
        traj_root_trans_world = (R_world_rh_batch @ traj.root_trans) + wr_right
        traj_root_orient_world = (
            R_world_rh_batch @ SO3.from_matrix(traj.root_orient)
        ).as_matrix()
        traj = network.EgoDenoiseTraj(
            betas=traj.betas,
            body_rotmats=traj.body_rotmats,
            contacts=traj.contacts,
            root_trans=traj_root_trans_world,
            root_orient=traj_root_orient_world,
            hand_rotmats=traj.hand_rotmats,
        )
        print("  Transformed back to world frame")

    # ------------------------------------------------------------------
    # 7. Evaluation
    # ------------------------------------------------------------------
    eval_results = evaluate_generation(
        traj=traj,
        body_model=body_model,
        gt_posed=gt_posed,
        gt_root_position=gt_root_trans,
    )
    print(f"[7/7] Evaluation complete")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    if args.save_output:
        args.output_dir.mkdir(exist_ok=True, parents=True)
        tag = (
            time.strftime("%Y%m%d-%H%M%S")
            + f"_unified_{resolved_param}_{prediction_type}"
        )
        out_path = args.output_dir / f"{tag}.npz"

        posed = traj.apply_to_body(body_model)
        pred_root_quat = SO3.from_matrix(traj.root_orient).wxyz
        Ts_world_root_out = torch.cat([pred_root_quat, traj.root_trans], dim=-1)

        np.savez(
            out_path,
            Ts_world_cpf=Ts_world_cpf[1:, :].cpu().numpy(),
            Ts_world_root=Ts_world_root_out.cpu().numpy(),
            body_quats=posed.local_quats[..., :21, :].cpu().numpy(),
            left_hand_quats=posed.local_quats[..., 21:36, :].cpu().numpy(),
            right_hand_quats=posed.local_quats[..., 36:51, :].cpu().numpy(),
            contacts=traj.contacts.cpu().numpy(),
            betas=traj.betas.cpu().numpy(),
            frame_nums=np.arange(actual_length),
            timestamps_ns=(np.arange(actual_length) / 30.0 * 1e9).astype(np.int64),
            wrist_cond_param=resolved_param,
            prediction_type=prediction_type,
            eval_results=eval_results,
            wrist_cond=(
                wrist_cond.cpu().numpy()
                if wrist_cond is not None
                else np.empty((0, 0), dtype=np.float32)
            ),
        )

        args_dict = dataclasses.asdict(args)
        args_dict["diagnostic_noise_levels"] = list(args_dict["diagnostic_noise_levels"])
        args_dict["resolved_wrist_cond_param"] = resolved_param
        args_dict["prediction_type"] = prediction_type
        for k, v in args_dict.items():
            if isinstance(v, Path):
                args_dict[k] = str(v)
        args_out = args.output_dir / f"{tag}_args.yaml"
        args_out.write_text(yaml.dump(args_dict, default_flow_style=False))

        print(f"[SAVE] {out_path}")
        print(f"[SAVE] {args_out}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    if args.visualize:
        import viser
        import viser.transforms as vtf
        from egoallo.vis_helpers import visualize_traj_and_hand_detections

        server = viser.ViserServer(port=args.viser_port)
        server.gui.configure_theme(dark_mode=True)

        num_fncsmpl_joints = body_model.get_num_joints()
        betas_for_viz = betas.unsqueeze(0)
        shaped_viz = body_model.with_shape(betas_for_viz)

        # Splines for root trajectory
        server.scene.add_spline_catmull_rom(
            "/gt/root",
            positions=gt_root_trans.cpu().numpy(),
            line_width=2.0,
            color=(0, 255, 128),
        )
        server.scene.add_spline_catmull_rom(
            "/pred/root",
            positions=traj.root_trans[0].detach().cpu().numpy(),
            line_width=2.0,
            color=(255, 128, 0),
        )

        # GT wrist point clouds
        server.scene.add_point_cloud(
            "/gt/left_wrist",
            points=wr_left.cpu().numpy(),
            colors=(0, 255, 0),
            point_size=0.015,
            point_shape="circle",
        )
        server.scene.add_point_cloud(
            "/gt/right_wrist",
            points=wr_right.cpu().numpy(),
            colors=(0, 127, 255),
            point_size=0.015,
            point_shape="circle",
        )

        # Predicted FK wrist positions
        pred_posed_viz = traj.apply_to_body(body_model)
        gen_lw = pred_posed_viz.Ts_world_joint[0, :, WRIST_LEFT_IDX, 4:7]
        gen_rw = pred_posed_viz.Ts_world_joint[0, :, WRIST_RIGHT_IDX, 4:7]
        server.scene.add_point_cloud(
            "/pred/left_wrist",
            points=gen_lw.cpu().numpy(),
            colors=(255, 0, 0),
            point_size=0.012,
            point_shape="diamond",
        )
        server.scene.add_point_cloud(
            "/pred/right_wrist",
            points=gen_rw.cpu().numpy(),
            colors=(255, 255, 0),
            point_size=0.012,
            point_shape="diamond",
        )

        # GT body mesh (transparent)
        gt_body_handle = server.scene.add_mesh_skinned(
            "/gt_person",
            vertices=shaped_viz.verts_zero[0, 0, :, :].numpy(force=True),
            faces=body_model.faces.numpy(force=True),
            bone_wxyzs=vtf.SO3.identity(
                batch_axes=(num_fncsmpl_joints + 1,)
            ).wxyz,
            bone_positions=np.concatenate(
                [
                    np.zeros((1, 3)),
                    shaped_viz.joints_zero[0, :, :, :].numpy(force=True).squeeze(axis=0),
                ],
                axis=0,
            ),
            color=(200, 200, 200),
            opacity=0.4,
            skin_weights=body_model.weights.numpy(force=True),
        )

        # GT bone transforms
        gt_root_bone = T_world_root.cpu().numpy()
        gt_joints_bone = gt_posed.Ts_world_joint.cpu().numpy()
        while gt_root_bone.ndim > 2:
            gt_root_bone = gt_root_bone.squeeze(0)
        while gt_joints_bone.ndim > 3:
            gt_joints_bone = gt_joints_bone.squeeze(0)

        # Predicted trajectory visualizer
        pred_root_q_viz = SO3.from_matrix(traj.root_orient).wxyz
        Ts_world_root_viz = torch.cat([pred_root_q_viz, traj.root_trans], dim=-1)

        loop_cb = visualize_traj_and_hand_detections(
            server,
            Ts_world_cpf[1:],
            traj,
            body_model,
            hamer_detections=None,
            aria_detections=None,
            points_data=None,
            splat_path=None,
            floor_z=0.0,
            Ts_world_root=Ts_world_root_viz,
        )

        gui_show_gt = server.gui.add_checkbox("Show GT body", initial_value=True)
        gui_gt_opacity = server.gui.add_slider(
            "GT opacity", min=0.0, max=1.0, step=0.01, initial_value=0.4,
        )

        @gui_show_gt.on_update
        def _(_) -> None:
            gt_body_handle.visible = gui_show_gt.value

        @gui_gt_opacity.on_update
        def _(_) -> None:
            gt_body_handle.opacity = gui_gt_opacity.value

        print(f"\nVisualizer running on port {args.viser_port}")
        print("Press Ctrl+C to stop")

        try:
            while True:
                t_idx = loop_cb()
                gt_body_handle.bones[0].wxyz = gt_root_bone[t_idx, :4]
                gt_body_handle.bones[0].position = gt_root_bone[t_idx, 4:7]
                for b in range(num_fncsmpl_joints):
                    gt_body_handle.bones[b + 1].wxyz = gt_joints_bone[t_idx, b, :4]
                    gt_body_handle.bones[b + 1].position = gt_joints_bone[t_idx, b, 4:7]
        except KeyboardInterrupt:
            print("\nVisualization stopped")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
