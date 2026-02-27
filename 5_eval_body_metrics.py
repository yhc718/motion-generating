"""Compute body metrics on the test split of the AMASS dataset.

For the new EgoAllo model that:
  - Predicts root_trans and root_orient (in right-hand coordinate frame)
  - Conditions on wrist positions (14D: pos + orient for left & right wrist)
  - Transforms predictions from RH frame back to world frame for evaluation

Usage:
    python 5_eval_body_metrics.py \
        --dataset-hdf5-path ./path/to/amass.hdf5 \
        --dataset-files-path ./data/amass_file.txt \
        --checkpoint-dir ./experiments/AMASS/v9/checkpoints_125000/ \
        --subseq-len 128 \
        --num-samples 1
"""

from pathlib import Path

import jax.tree
import numpy as np
import torch
import torch.utils.data
import tyro

from egoallo import fncsmpl, network
from egoallo.data.amass import EgoAmassHdf5Dataset
from egoallo.inference_utils import load_denoiser
from egoallo.metrics_helpers import (
    compute_foot_contact,
    compute_foot_skate,
    compute_mpjpe,
    compute_wrist_trans,
)
from egoallo.sampling import run_sampling_with_stitching
from egoallo.transforms import SE3, SO3

# Wrist joint indices in fncsmpl (0-indexed, root excluded)
WRIST_LEFT_IDX = 19
WRIST_RIGHT_IDX = 20


def _compute_gt_wrist_and_rh_frame(
    body_model: fncsmpl.SmplhModel,
    sequence,  # EgoTrainingData on device
    subseq_len: int,
):
    """From GT EgoTrainingData, compute:
    1. GT FK (label_posed) for metric computation
    2. Wrist conditioning (14D) matching training pipeline
    3. Right-hand frame transforms for RH→world conversion

    All GT data uses timesteps [1:] (first frame consumed for delta).

    Returns:
        label_posed: GT FK result
        wrist_positions: (time, 14)  [left_pos(3), right_pos(3), left_quat(4), right_quat(4)]
        R_world_rh: SO3 (time,)
        rh_pos_world: (time, 3)
    """
    device = sequence.T_world_root.device

    # Skip first frame (same as original eval & training convention)
    T_world_root = sequence.T_world_root[1:]       # (subseq_len, 7)
    body_quats = sequence.body_quats[1:]            # (subseq_len, 21, 4)
    betas = sequence.betas[0:1]                     # (1, 16) — all rows identical, take first
    hand_quats = sequence.hand_quats
    if hand_quats is not None:
        hand_quats = hand_quats[1:]                 # (subseq_len, 30, 4)

    time = T_world_root.shape[0]

    # --- GT FK ---
    if hand_quats is not None:
        label_posed = body_model.with_shape(betas).with_pose(
            T_world_root,
            torch.cat([body_quats, hand_quats], dim=1),
        )
    else:
        label_posed = body_model.with_shape(betas).with_pose(
            T_world_root,
            body_quats,
        )

    # --- Wrist conditioning (matching training_loss_no_fk._compute_wrist_fk) ---
    wrist_pos = label_posed.Ts_world_joint[:, [WRIST_LEFT_IDX, WRIST_RIGHT_IDX], 4:7]  # (T, 2, 3)
    wrist_orient = label_posed.Ts_world_joint[:, [WRIST_LEFT_IDX, WRIST_RIGHT_IDX], :4]  # (T, 2, 4)

    wrist_positions = torch.cat([
        wrist_pos.reshape(time, 6),        # left_pos(3), right_pos(3)
        wrist_orient.reshape(time, 8),     # left_quat(4), right_quat(4)
    ], dim=-1)  # (T, 14)

    # --- Right-hand frame ---
    rh_quat_world = wrist_orient[:, 1, :]        # right hand quaternion (T, 4)
    rh_pos_world = wrist_pos[:, 1, :]            # right hand position (T, 3)
    R_world_rh = SO3(rh_quat_world)              # rotation from RH frame to world

    return label_posed, wrist_positions, R_world_rh, rh_pos_world


def main(
    dataset_hdf5_path: Path,
    dataset_files_path: Path,
    subseq_len: int = 128,
    checkpoint_dir: Path = Path("./experiments/AMASS/v9/checkpoints_125000/"),
    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz"),
    num_samples: int = 1,
    # Sampling options
    sigma_scale: float = 0.8,
    init_noise_scale: float = 1.0,
    clip_denoised: bool = False,
    clip_root_trans_range: float = 2.0,
    wrist_correction: bool = False,
    wrist_correction_weight: float = 0.5,
    post_wrist_correction: bool = False,
    post_wrist_correction_iters: int = 3,
    post_wrist_correction_weight: float = 0.8,
) -> None:
    """Compute body metrics on the test split of the AMASS dataset.

    Uses the new EgoAllo model that predicts root in right-hand frame
    and conditions on wrist positions.
    """
    device = torch.device("cuda")

    denoiser_network = load_denoiser(checkpoint_dir).to(device)
    body_model = fncsmpl.SmplhModel.load(smplh_npz_path).to(device)

    print(f"Model config:")
    print(f"  include_hands: {denoiser_network.config.include_hands}")
    print(f"  use_wrist_conditioning: {denoiser_network.config.use_wrist_conditioning}")
    print(f"  d_state: {denoiser_network.get_d_state()}")

    dataset = EgoAmassHdf5Dataset(
        dataset_hdf5_path,
        dataset_files_path,
        splits=("test",),
        subseq_len=subseq_len + 1,
        cache_files=True,
        slice_strategy="deterministic",
        random_variable_len_proportion=0.0,
    )

    metrics = list[dict[str, np.ndarray]]()

    for i in range(len(dataset)):
        sequence = dataset[i].to(device)

        # ----- 1. Build GT wrist conditioning & RH frame from GT FK -----
        label_posed, wrist_positions, R_world_rh, rh_pos_world = (
            _compute_gt_wrist_and_rh_frame(body_model, sequence, subseq_len)
        )

        # ----- 2. Run diffusion sampling (model predicts in RH frame) -----
        # Build a dummy CPF trajectory (not used by wrist-conditioned model,
        # but required by run_sampling_with_stitching's signature)
        Ts_world_cpf = torch.zeros((subseq_len + 1, 7), device=device)
        Ts_world_cpf[:, 0] = 1.0   # identity quaternion
        Ts_world_cpf[:, 6] = 1.6   # reasonable head height

        samples = run_sampling_with_stitching(
            denoiser_network,
            body_model=body_model,
            guidance_mode="off",
            guidance_inner=False,
            guidance_post=False,
            Ts_world_cpf=Ts_world_cpf,
            hamer_detections=None,
            aria_detections=None,
            num_samples=num_samples,
            floor_z=0.0,
            device=device,
            guidance_verbose=False,
            wrist_positions=wrist_positions,
            sigma_scale=sigma_scale,
            init_noise_scale=init_noise_scale,
            clip_denoised=clip_denoised,
            clip_root_trans_range=clip_root_trans_range,
            wrist_correction=wrist_correction,
            wrist_correction_weight=wrist_correction_weight,
            post_wrist_correction=post_wrist_correction,
            post_wrist_correction_iters=post_wrist_correction_iters,
            post_wrist_correction_weight=post_wrist_correction_weight,
        )

        # ----- 3. Transform predictions from RH frame to world frame -----
        # samples.root_trans is in RH frame: root_trans_rh
        # samples.root_orient is in RH frame: root_orient_rh
        # World frame: root_trans_world = R_world_rh @ root_trans_rh + rh_pos_world
        #              root_orient_world = R_world_rh @ root_orient_rh

        # Expand R_world_rh and rh_pos_world for num_samples
        # R_world_rh: (time,) → need to apply to (num_samples, time, ...)
        R_world_rh_expanded = SO3(R_world_rh.wxyz.unsqueeze(0))  # (1, T, 4)
        root_trans_world = (
            R_world_rh_expanded @ samples.root_trans  # (S, T, 3)
        ) + rh_pos_world.unsqueeze(0)  # (1, T, 3)
        root_orient_world = (
            R_world_rh_expanded @ SO3.from_matrix(samples.root_orient)  # (S, T)
        ).as_matrix()  # (S, T, 3, 3)

        samples = network.EgoDenoiseTraj(
            betas=samples.betas,
            body_rotmats=samples.body_rotmats,
            contacts=samples.contacts,
            root_trans=root_trans_world,
            root_orient=root_orient_world,
            hand_rotmats=samples.hand_rotmats,
        )

        # ----- 4. FK with predicted root + body rotations in world frame -----
        pred_root_quat = SO3.from_matrix(samples.root_orient).wxyz  # (S, T, 4)
        pred_T_world_root = torch.cat([
            pred_root_quat,         # (S, T, 4)
            samples.root_trans,     # (S, T, 3)
        ], dim=-1)  # (S, T, 7)

        pred_body_quats = SO3.from_matrix(samples.body_rotmats).wxyz  # (S, T, 21, 4)

        if samples.hand_rotmats is not None:
            pred_hand_quats = SO3.from_matrix(samples.hand_rotmats).wxyz
            pred_left_hand_quats = pred_hand_quats[..., :15, :]
            pred_right_hand_quats = pred_hand_quats[..., 15:30, :]
        else:
            pred_left_hand_quats = None
            pred_right_hand_quats = None

        # Use mean betas across time for FK (same as test_diffusion_sample.py)
        pred_betas_mean = samples.betas.mean(dim=1, keepdim=True)  # (S, 1, 16)
        pred_shaped = body_model.with_shape(pred_betas_mean)

        pred_posed = pred_shaped.with_pose_decomposed(
            T_world_root=pred_T_world_root,
            body_quats=pred_body_quats,
            left_hand_quats=pred_left_hand_quats,
            right_hand_quats=pred_right_hand_quats,
        )

        # ----- 5. Compute metrics -----
        metrics.append(
            {
                "mpjpe": compute_mpjpe(
                    label_T_world_root=label_posed.T_world_root,
                    label_Ts_world_joint=label_posed.Ts_world_joint[:, :21, :],
                    pred_T_world_root=pred_posed.T_world_root,
                    pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                    per_frame_procrustes_align=False,
                ),
                "pampjpe": compute_mpjpe(
                    label_T_world_root=label_posed.T_world_root,
                    label_Ts_world_joint=label_posed.Ts_world_joint[:, :21, :],
                    pred_T_world_root=pred_posed.T_world_root,
                    pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                    per_frame_procrustes_align=True,
                ),
                "foot_skate": compute_foot_skate(
                    pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                ),
                "foot_contact (GND)": compute_foot_contact(
                    pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                ),
                "T_wrist": compute_wrist_trans(
                    label_Ts_world_joint=label_posed.Ts_world_joint[:, :21, :],
                    pred_Ts_world_joint=pred_posed.Ts_world_joint[:, :, :21, :],
                ),
            }
        )

        print("=" * 80)
        print(f"Metrics ({i + 1}/{len(dataset)} processed)")
        for k, v in jax.tree.map(
            lambda *x: f"{np.mean(x):.3f} +/- {np.std(x) / np.sqrt(len(metrics) * num_samples):.3f}",
            *metrics,
        ).items():
            print("\t", k, v)
        print("=" * 80)


if __name__ == "__main__":
    tyro.cli(main)
