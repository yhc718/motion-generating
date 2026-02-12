"""
3_aria_inference_wrist.py

改自 3_aria_inference.py：不再吃 VRS / Aria MPS 資料，
改為讀取 NPZ 檔案中的 SMPL-H body parameters，
用 SMPLH FK 計算 wrist positions 作為 conditioning，
然後跑 diffusion sampling + guidance optimization (foot skating / smoothness)。

支援兩種 NPZ 格式：
  1) StageII 格式：含 root_orient / trans / pose_body / betas (或 poses)
  2) GRAB 格式：含 body['params']['global_orient'] / transl / body_pose / betas

用法 (tyro CLI)：
    python 3_aria_inference_wrist.py \
        --input-npz ./Subject_1_F_1_stageii.npz \
        --checkpoint-dir ./experiments/AMASS/v9/checkpoints_125000/ \
        --traj-length 256 \
        --guidance-mode no_hands \
        --guidance-inner \
        --guidance-post
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import viser
import yaml

from egoallo import fncsmpl, fncsmpl_extensions
from egoallo.guidance_optimizer_jax import GuidanceMode
from egoallo.inference_utils import load_denoiser
from egoallo.network import EgoDenoiseTraj
from egoallo.sampling import run_sampling_with_stitching
from egoallo.transforms import SE3, SO3
from egoallo.vis_helpers import visualize_traj_and_hand_detections


# ── Guidance modes that don't need Aria / HaMeR ──────────────────────────
# "no_hands" : foot skating + smoothness + prior losses only
# 其他 mode (aria_wrist_only, aria_hamer, hamer_wrist, hamer_reproj2)
#   都需要 Aria / HaMeR detections，不建議在此使用。
# ─────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Args:
    input_npz: Path = Path("./Subject_1_F_1_stageii.npz")
    """包含 SMPL-H body parameters 的 NPZ 檔（StageII 或 GRAB 格式）"""

    input_format: str = "stageii"
    """NPZ 格式：'stageii' (root_orient/trans/pose_body) 或 'grab' (body.params.*)"""

    checkpoint_dir: Path = Path("./experiments/AMASS/v12/checkpoints_125000/")
    """Diffusion model checkpoint 路徑"""

    smplh_npz_path: Path = Path("./data/smplh/neutral/model.npz")
    """SMPLH body model 路徑"""

    # ── Sampling 參數 ───────────────────────────────────────────────
    start_index: int = 0
    """從第幾個 frame 開始取"""

    traj_length: int = 2048
    """要生成的 motion 長度 (frames)"""

    num_samples: int = 1
    """要生成幾個 samples"""

    # ── Guidance optimization 參數 ──────────────────────────────────
    guidance_mode: GuidanceMode = "no_hands"
    """Guidance mode。因為沒有 Aria / HaMeR detections，
    建議使用 'no_hands' (foot skating + smoothness + prior losses)。"""

    guidance_inner: bool = True
    """是否在 denoising steps 之間執行 guidance optimization（每步 ~5 LM iterations）"""

    guidance_post: bool = True
    """是否在 diffusion 完成後再執行一次 guidance optimization（~20 LM iterations）"""

    # ── 輸出設置 ────────────────────────────────────────────────────
    save_traj: bool = True
    """是否儲存輸出（放在 ./egoallo_outputs/）"""

    visualize_traj: bool = True
    """是否啟動 viser 可視化"""


# ═══════════════════════════════════════════════════════════════════════
#  NPZ loading helpers
# ═══════════════════════════════════════════════════════════════════════

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    return x[None, :] if x.ndim == 1 else x


def load_stageii(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    StageII 格式：root_orient (T,3), trans (T,3), pose_body (T,63), betas (1,16)
    也支援只有 poses (T, >=66) + trans 的格式。
    """
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)

    # root_orient
    if "root_orient" in keys:
        root_orient = _ensure_2d(data["root_orient"])
    elif "poses" in keys:
        root_orient = _ensure_2d(data["poses"])[:, :3]
    else:
        raise KeyError("Missing root_orient or poses in NPZ.")

    # transl
    if "trans" in keys:
        transl = _ensure_2d(data["trans"])
    else:
        raise KeyError("Missing trans in NPZ.")

    # body pose (21 joints × 3 axis-angle = 63)
    if "pose_body" in keys:
        pose_body = _ensure_2d(data["pose_body"])
    elif "poses" in keys:
        pose_body = _ensure_2d(data["poses"])[:, 3:66]
    else:
        raise KeyError("Missing pose_body or poses in NPZ.")

    # betas
    if "betas" in keys:
        betas = _ensure_2d(data["betas"])
    else:
        betas = np.zeros((1, 16), dtype=np.float32)

    return root_orient.astype(np.float32), transl.astype(np.float32), pose_body.astype(np.float32), betas.astype(np.float32)


def load_grab(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    GRAB 格式（如 airplane_fly_1.npz）：
    body['params']['global_orient'], body['params']['transl'],
    body['params']['body_pose'], body['params']['betas']
    """
    data = np.load(npz_path, allow_pickle=True)
    body_data = data["body"].item()
    body_params = body_data["params"]

    root_orient = body_params["global_orient"]    # (T, 3) axis-angle
    transl = body_params["transl"]                # (T, 3)
    pose_body = body_params["body_pose"]          # (T, 63)
    betas = body_params.get("betas", np.zeros((1, 16), dtype=np.float32))

    return root_orient.astype(np.float32), transl.astype(np.float32), pose_body.astype(np.float32), betas.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
#  Compute wrist positions (14D) using SMPLH FK — matches training pipeline
# ═══════════════════════════════════════════════════════════════════════

def compute_wrist_conditioning(
    body_model: fncsmpl.SmplhModel,
    root_orient: np.ndarray,   # (T, 3) axis-angle
    transl: np.ndarray,        # (T, 3)
    pose_body: np.ndarray,     # (T, 63)
    betas: np.ndarray,         # (1, D)
    device: torch.device,
) -> torch.Tensor:
    """
    用 SMPLH FK 計算 wrist positions + orientations，
    回傳 shape (T, 14) = [left_pos(3), right_pos(3), left_quat(4), right_quat(4)]
    """
    T = root_orient.shape[0]

    # axis-angle → quaternion (wxyz)
    root_orient_quat = SO3.exp(
        torch.from_numpy(root_orient).to(device)
    ).wxyz  # (T, 4)

    body_quats = SO3.exp(
        torch.from_numpy(pose_body).to(device).reshape(T, 21, 3)
    ).wxyz  # (T, 21, 4)

    transl_torch = torch.from_numpy(transl).to(device)  # (T, 3)

    # betas → (1, 16)
    betas_np = betas[:1].copy()
    if betas_np.shape[1] < 16:
        betas_np = np.concatenate(
            [betas_np, np.zeros((1, 16 - betas_np.shape[1]), dtype=np.float32)], axis=1
        )
    betas_torch = torch.from_numpy(betas_np).to(device)

    shaped = body_model.with_shape(betas_torch)

    # FK: T_world_root = (root_orient_quat, transl)
    T_world_root = torch.cat([root_orient_quat, transl_torch], dim=-1)  # (T, 7)
    posed = shaped.with_pose_decomposed(
        T_world_root=T_world_root,
        body_quats=body_quats,
        left_hand_quats=None,
        right_hand_quats=None,
    )

    WRIST_LEFT_IDX = 19
    WRIST_RIGHT_IDX = 20

    left_pos = posed.Ts_world_joint[:, WRIST_LEFT_IDX, 4:7]    # (T, 3)
    right_pos = posed.Ts_world_joint[:, WRIST_RIGHT_IDX, 4:7]  # (T, 3)
    left_quat = posed.Ts_world_joint[:, WRIST_LEFT_IDX, :4]    # (T, 4)
    right_quat = posed.Ts_world_joint[:, WRIST_RIGHT_IDX, :4]  # (T, 4)

    return torch.cat([left_pos, right_pos, left_quat, right_quat], dim=-1)  # (T, 14)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main(args: Args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 1. Load body model ────────────────────────────────────────────
    print(f"\n[1] 載入 SMPLH model from {args.smplh_npz_path}...")
    body_model = fncsmpl.SmplhModel.load(args.smplh_npz_path).to(device)

    # ── 2. Load input NPZ (wrist motion source) ──────────────────────
    print(f"\n[2] 載入輸入 NPZ: {args.input_npz}  (format={args.input_format})")
    if args.input_format == "stageii":
        root_orient, transl, pose_body, betas_raw = load_stageii(args.input_npz)
    elif args.input_format == "grab":
        root_orient, transl, pose_body, betas_raw = load_grab(args.input_npz)
    else:
        raise ValueError(f"Unknown input_format: {args.input_format!r}. Use 'stageii' or 'grab'.")

    n_frames = root_orient.shape[0]
    print(f"  Total frames in file: {n_frames}")
    print(f"  root_orient: {root_orient.shape}, transl: {transl.shape}, pose_body: {pose_body.shape}")

    # 確保 start_index + traj_length 不超過資料長度
    actual_length = min(args.traj_length, n_frames - args.start_index)
    if actual_length < args.traj_length:
        print(f"  Warning: 截斷 traj_length {args.traj_length} → {actual_length}")
    end_index = args.start_index + actual_length

    # Slice
    root_orient_slice = root_orient[args.start_index:end_index]
    transl_slice = transl[args.start_index:end_index]
    pose_body_slice = pose_body[args.start_index:end_index]

    # ── 3. Compute wrist conditioning (14D) via SMPLH FK ─────────────
    print(f"\n[3] 用 SMPLH FK 計算 wrist positions + orientations for conditioning...")
    wrist_positions_cond = compute_wrist_conditioning(
        body_model, root_orient_slice, transl_slice, pose_body_slice, betas_raw, device
    )
    print(f"  wrist_positions_cond shape: {wrist_positions_cond.shape}")  # (T, 14)
    print(f"  Left wrist pos range:  X=[{wrist_positions_cond[:, 0].min():.3f}, {wrist_positions_cond[:, 0].max():.3f}]  "
          f"Y=[{wrist_positions_cond[:, 1].min():.3f}, {wrist_positions_cond[:, 1].max():.3f}]  "
          f"Z=[{wrist_positions_cond[:, 2].min():.3f}, {wrist_positions_cond[:, 2].max():.3f}]")
    print(f"  Right wrist pos range: X=[{wrist_positions_cond[:, 3].min():.3f}, {wrist_positions_cond[:, 3].max():.3f}]  "
          f"Y=[{wrist_positions_cond[:, 4].min():.3f}, {wrist_positions_cond[:, 4].max():.3f}]  "
          f"Z=[{wrist_positions_cond[:, 5].min():.3f}, {wrist_positions_cond[:, 5].max():.3f}]")

    # ── 4. Build dummy Ts_world_cpf ──────────────────────────────────
    #   因為沒有 VRS/Aria head tracking，用 dummy identity trajectory。
    #   wrist conditioning model 不會真的用 CPF，但 sampling 函式簽名需要。
    print(f"\n[4] 構建 dummy Ts_world_cpf (length={actual_length + 1})...")
    Ts_world_cpf = torch.zeros((actual_length + 1, 7), device=device)
    Ts_world_cpf[:, 0] = 1.0   # identity rotation (qw=1)
    Ts_world_cpf[:, 6] = 1.6   # 合理的頭部高度

    # ── 5. Load denoiser ─────────────────────────────────────────────
    print(f"\n[5] 載入 denoiser from {args.checkpoint_dir}...")
    denoiser_network = load_denoiser(args.checkpoint_dir).to(device)
    print(f"  include_hands={denoiser_network.config.include_hands}")
    print(f"  use_wrist_conditioning={denoiser_network.config.use_wrist_conditioning}")
    print(f"  d_state={denoiser_network.get_d_state()}")

    # ── 6. Diffusion sampling + guidance optimization ────────────────
    print(f"\n[6] 開始 diffusion sampling + guidance optimization...")
    print(f"  guidance_mode  = {args.guidance_mode}")
    print(f"  guidance_inner = {args.guidance_inner}")
    print(f"  guidance_post  = {args.guidance_post}")
    print(f"  num_samples    = {args.num_samples}")
    print(f"  traj_length    = {actual_length}")

    start_time = time.time()
    traj = run_sampling_with_stitching(
        denoiser_network,
        body_model=body_model,
        guidance_mode=args.guidance_mode,
        guidance_inner=args.guidance_inner,
        guidance_post=args.guidance_post,
        Ts_world_cpf=Ts_world_cpf,
        hamer_detections=None,
        aria_detections=None,
        num_samples=args.num_samples,
        device=device,
        floor_z=0.0,
        object_sdf_data=None,
        guidance_verbose=True,
        wrist_positions=wrist_positions_cond,  # ← wrist conditioning!
        use_predicted_root=True,  # ← 用 model predicted root 做 FK，讓 skating loss 生效！
    )
    elapsed = time.time() - start_time
    print(f"  Sampling 完成！耗時: {elapsed:.1f}s")

    # ── 7. Save outputs ──────────────────────────────────────────────
    if args.save_traj:
        output_dir = Path("./egoallo_outputs")
        output_dir.mkdir(exist_ok=True, parents=True)

        save_name = (
            time.strftime("%Y%m%d-%H%M%S")
            + f"_{args.start_index}-{args.start_index + actual_length}"
        )
        out_path = output_dir / (save_name + ".npz")
        args_path = output_dir / (save_name + "_args.yaml")

        print(f"\n[7] 儲存結果到 {out_path}...")

        posed = traj.apply_to_body(body_model)

        # 使用 diffusion model 預測的 root orient + root trans
        pred_root_quat = SO3.from_matrix(traj.root_orient).wxyz   # (S, T, 4)
        Ts_world_root = torch.cat([pred_root_quat, traj.root_trans], dim=-1)  # (S, T, 7)

        # synthetic timestamps (30 fps)
        pose_timestamps_sec = np.arange(actual_length) / 30.0

        np.savez(
            out_path,
            Ts_world_cpf=Ts_world_cpf[1:, :].cpu().numpy(),
            Ts_world_root=Ts_world_root.cpu().numpy(),
            body_quats=posed.local_quats[..., :21, :].cpu().numpy(),
            left_hand_quats=posed.local_quats[..., 21:36, :].cpu().numpy(),
            right_hand_quats=posed.local_quats[..., 36:51, :].cpu().numpy(),
            contacts=traj.contacts.cpu().numpy(),
            betas=traj.betas.cpu().numpy(),
            frame_nums=np.arange(args.start_index, args.start_index + actual_length),
            timestamps_ns=(pose_timestamps_sec * 1e9).astype(np.int64),
        )
        args_path.write_text(yaml.dump(dataclasses.asdict(args)))
        print(f"  Saved!  NPZ={out_path}  Args={args_path}")

    # ── 8. Visualize ─────────────────────────────────────────────────
    if args.visualize_traj:
        print(f"\n[8] 啟動 viser 可視化...")
        server = viser.ViserServer()
        server.gui.configure_theme(dark_mode=True)

        # 計算生成的 root transform for visualization
        pred_root_quat_viz = SO3.from_matrix(traj.root_orient).wxyz
        Ts_world_root_viz = torch.cat([pred_root_quat_viz, traj.root_trans], dim=-1)

        # 可視化 conditioning wrist positions (綠/藍 = GT left/right wrist)
        server.scene.add_point_cloud(
            "/condition/left_wrist",
            points=wrist_positions_cond[:, 0:3].cpu().numpy(),
            colors=(0, 255, 0),
            point_size=0.015,
            point_shape="circle",
        )
        server.scene.add_point_cloud(
            "/condition/right_wrist",
            points=wrist_positions_cond[:, 3:6].cpu().numpy(),
            colors=(0, 127, 255),
            point_size=0.015,
            point_shape="circle",
        )

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

        print("  可視化已啟動，請開瀏覽器查看。按 Ctrl+C 停止。")
        try:
            while True:
                loop_cb()
        except KeyboardInterrupt:
            print("\n停止可視化")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
