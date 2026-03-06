"""Training loss configuration with wrist FK error and right-hand coordinate frame.

This is an extended version of training_loss.py that:
1. Uses the condition's right hand as the coordinate system for root prediction.
2. Adds wrist position error computed via Forward Kinematics (FK).
3. Removes foot FK and skating losses.
"""

import dataclasses
from pathlib import Path
from typing import Literal

import torch
import torch.utils.data
from jaxtyping import Bool, Float, Int
from torch import Tensor
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel

from . import fncsmpl, network
from .data.amass import EgoTrainingData
from .sampling import CosineNoiseScheduleConstants
from .transforms import SE3, SO3


# Wrist joint indices in SMPL-H body model (0-indexed, excluding root)
# Joint 19 = left wrist, Joint 20 = right wrist
WRIST_LEFT_IDX = 19
WRIST_RIGHT_IDX = 20

# Foot/ankle joint indices (0-indexed, excluding root)
# These match the indices used in guidance_optimizer_jax.py for skating cost.
# Joint 6 = left ankle, Joint 7 = left foot
# Joint 9 = right ankle, Joint 10 = right foot
FOOT_JOINT_INDICES = [6, 7, 9, 10]
NUM_FOOT_JOINTS = len(FOOT_JOINT_INDICES)


@dataclasses.dataclass(frozen=True)
class TrainingLossConfigWithWrist:
    """Training loss config with additional wrist FK error."""
    
    cond_dropout_prob: float = 0.0
    beta_coeff_weights: tuple[float, ...] = tuple(1 / (i + 1) for i in range(16))
    loss_weights: dict[str, float] = dataclasses.field(
        default_factory={
            "betas": 0.1,
            "body_rotmats": 3.0,
            "contacts": 0.1,
            # We don't have many hands in the AMASS dataset...
            "hand_rotmats": 0.01,
            "root_trans": 1.0,
            "root_orient": 1.0,
            # Wrist FK error (in world frame, after transforming root back from RH frame)
            "wrist_fk": 3.0,
            # Root translation velocity loss (per-frame displacement matching, in RH frame)
            "root_vel": 1.0,
        }.copy
    )
    weight_loss_by_t: Literal["emulate_eps_pred"] = "emulate_eps_pred"
    """Weights to apply to the loss at each noise level."""
    
    # Path to SMPL-H model for FK computation
    smplh_model_path: Path | None = None
    """Path to SMPL-H npz model file. If None, wrist FK loss will be skipped."""


class TrainingLossComputerWithWrist:
    """Helper class for computing the training loss with wrist FK error.
    
    This extends the original TrainingLossComputer by adding a wrist position
    loss computed via Forward Kinematics.
    """

    def __init__(
        self, 
        config: TrainingLossConfigWithWrist, 
        device: torch.device,
        body_model: fncsmpl.SmplhModel | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.noise_constants = (
            CosineNoiseScheduleConstants.compute(timesteps=1000)
            .to(device)
            .map(lambda tensor: tensor.to(torch.float32))
        )

        # Emulate loss weight that would be ~equivalent to epsilon prediction.
        #
        # This will penalize later errors (close to the end of sampling) much
        # more than earlier ones (at the start of sampling).
        assert self.config.weight_loss_by_t == "emulate_eps_pred"
        weight_t = self.noise_constants.alpha_bar_t / (
            1 - self.noise_constants.alpha_bar_t
        )
        # Pad for numerical stability, and scale between [padding, 1.0].
        padding = 0.01
        self.weight_t = weight_t / weight_t[1] * (1.0 - padding) + padding
        
        # Load body model for FK computation
        if body_model is not None:
            self.body_model = body_model.to(device)
        elif config.smplh_model_path is not None:
            self.body_model = fncsmpl.SmplhModel.load(config.smplh_model_path).to(device)
        else:
            self.body_model = None

    def _compute_wrist_fk(
        self,
        betas: Float[Tensor, "batch time 16"],
        body_rotmats: Float[Tensor, "batch time 21 3 3"],
        root_trans: Float[Tensor, "batch time 3"],
        root_orient: Float[Tensor, "batch time 3 3"],
    ) -> dict[str, Float[Tensor, "..."]]:
        """Compute wrist positions and orientations using Forward Kinematics.
        
        Args:
            betas: Body shape parameters.
            body_rotmats: Body joint rotation matrices (21 joints, NOT including root).
            root_trans: Root translation in world frame.
            root_orient: Root orientation as rotation matrix (3x3). This is the global
                pelvis rotation from T_world_root, NOT body_quats[0] which is
                the left hip local rotation.
            
        Returns:
            Dict with:
                'positions': Wrist positions (left and right) in world frame, shape (batch, time, 2, 3).
                'orientations': Wrist orientations (left and right) as wxyz quaternions, shape (batch, time, 2, 4).
        """
        assert self.body_model is not None, "Body model required for FK computation"
        
        batch, time, num_joints, _, _ = body_rotmats.shape
        assert num_joints == 21
        
        # Convert rotation matrices to quaternions
        body_quats = SO3.from_matrix(body_rotmats).wxyz  # (batch, time, 21, 4)
        root_quat = SO3.from_matrix(root_orient).wxyz  # (batch, time, 4)
        
        # Flatten batch and time for FK computation
        body_quats_flat = body_quats.reshape(batch * time, 21, 4)
        betas_flat = betas[:, 0, :]  # (batch, 16) - betas are same across time
        root_trans_flat = root_trans.reshape(batch * time, 3)
        root_quat_flat = root_quat.reshape(batch * time, 4)
        
        # Construct T_world_root: (batch*time, 7) as wxyz_xyz
        T_world_root = torch.cat([root_quat_flat, root_trans_flat], dim=-1)
        
        # Apply shape to get shaped model
        betas_expanded = betas_flat.unsqueeze(1).expand(batch, time, 16).reshape(batch * time, 16)
        shaped = self.body_model.with_shape(betas_expanded)
        
        # Pose the body (only body joints, no hands)
        posed = shaped.with_pose_decomposed(
            T_world_root=T_world_root,
            body_quats=body_quats_flat,
            left_hand_quats=None,
            right_hand_quats=None,
        )
        
        # Extract wrist joint positions and orientations from Ts_world_joint
        # Ts_world_joint has shape (batch*time, num_joints, 7) as wxyz_xyz
        wrist_positions_flat = posed.Ts_world_joint[:, [WRIST_LEFT_IDX, WRIST_RIGHT_IDX], 4:7]
        wrist_orientations_flat = posed.Ts_world_joint[:, [WRIST_LEFT_IDX, WRIST_RIGHT_IDX], :4]
        
        # Reshape back to (batch, time, 2, ...)
        wrist_positions = wrist_positions_flat.reshape(batch, time, 2, 3)
        wrist_orientations = wrist_orientations_flat.reshape(batch, time, 2, 4)
        
        return {
            'positions': wrist_positions,
            'orientations': wrist_orientations,
        }

    def _compute_joint_positions_fk(
        self,
        betas: Float[Tensor, "batch time 16"],
        body_rotmats: Float[Tensor, "batch time 21 3 3"],
        root_trans: Float[Tensor, "batch time 3"],
        root_orient: Float[Tensor, "batch time 3 3"],
        joint_indices: list[int] | None = None,
    ) -> Float[Tensor, "batch time J 3"]:
        """Compute world-frame joint positions via FK for specified joints.

        Args:
            betas: Body shape parameters.
            body_rotmats: Body joint rotation matrices (21 joints).
            root_trans: Root translation in world frame.
            root_orient: Root orientation as rotation matrix (3x3).
            joint_indices: Which joints to extract. If None, returns all joints.

        Returns:
            World-frame positions of the requested joints, shape (batch, time, J, 3).
        """
        assert self.body_model is not None, "Body model required for FK computation"

        batch, time, num_joints, _, _ = body_rotmats.shape
        assert num_joints == 21

        # Convert rotation matrices to quaternions
        body_quats = SO3.from_matrix(body_rotmats).wxyz  # (batch, time, 21, 4)
        root_quat = SO3.from_matrix(root_orient).wxyz  # (batch, time, 4)

        # Flatten batch and time for FK computation
        body_quats_flat = body_quats.reshape(batch * time, 21, 4)
        root_trans_flat = root_trans.reshape(batch * time, 3)
        root_quat_flat = root_quat.reshape(batch * time, 4)

        # Construct T_world_root: (batch*time, 7) as wxyz_xyz
        T_world_root = torch.cat([root_quat_flat, root_trans_flat], dim=-1)

        # Apply shape
        betas_expanded = betas[:, 0, :].unsqueeze(1).expand(batch, time, 16).reshape(batch * time, 16)
        shaped = self.body_model.with_shape(betas_expanded)

        # Pose the body
        posed = shaped.with_pose_decomposed(
            T_world_root=T_world_root,
            body_quats=body_quats_flat,
            left_hand_quats=None,
            right_hand_quats=None,
        )

        # Extract requested joint positions
        if joint_indices is not None:
            positions_flat = posed.Ts_world_joint[:, joint_indices, 4:7]
            num_out = len(joint_indices)
        else:
            positions_flat = posed.Ts_world_joint[:, :, 4:7]
            num_out = posed.Ts_world_joint.shape[1]

        return positions_flat.reshape(batch, time, num_out, 3)

    @staticmethod
    def _get_first_valid_indices(mask: Bool[Tensor, "batch time"]) -> Int[Tensor, "batch"]:
        """Get first valid timestep per sequence, fallback to 0 if all masked."""
        first_valid = mask.to(torch.int64).argmax(dim=1)
        has_valid = mask.any(dim=1)
        zeros = torch.zeros_like(first_valid)
        return torch.where(has_valid, first_valid, zeros)

    @staticmethod
    def _build_wrist_cond_absolute(
        wrist_positions_world: Float[Tensor, "batch time 2 3"],
        wrist_orientations_world: Float[Tensor, "batch time 2 4"],
    ) -> Float[Tensor, "batch time 14"]:
        batch, time, _, _ = wrist_positions_world.shape
        return torch.cat(
            [
                wrist_positions_world.reshape(batch, time, 6),
                wrist_orientations_world.reshape(batch, time, 8),
            ],
            dim=-1,
        )

    def _build_wrist_cond_rh_first_base(
        self,
        wrist_positions_world: Float[Tensor, "batch time 2 3"],
        wrist_orientations_world: Float[Tensor, "batch time 2 4"],
        mask: Bool[Tensor, "batch time"],
        keep_global_z: bool,
    ) -> Float[Tensor, "batch time 14"]:
        """Build per-hand first-frame wrist conditioning (without hand communication)."""
        batch, time, _, _ = wrist_positions_world.shape

        first_idx = self._get_first_valid_indices(mask)
        batch_idx = torch.arange(batch, device=wrist_positions_world.device)

        lh0_pos_world = wrist_positions_world[batch_idx, first_idx, 0, :]  # (b, 3)
        lh0_quat_world = wrist_orientations_world[batch_idx, first_idx, 0, :]  # (b, 4)
        rh0_pos_world = wrist_positions_world[batch_idx, first_idx, 1, :]  # (b, 3)
        rh0_quat_world = wrist_orientations_world[batch_idx, first_idx, 1, :]  # (b, 4)

        lh0_pos_world_flat = lh0_pos_world[:, None, :].expand(-1, time, -1).reshape(batch * time, 3)
        rh0_pos_world_flat = rh0_pos_world[:, None, :].expand(-1, time, -1).reshape(batch * time, 3)
        lh0_quat_world_flat = lh0_quat_world[:, None, :].expand(-1, time, -1).reshape(batch * time, 4)
        rh0_quat_world_flat = rh0_quat_world[:, None, :].expand(-1, time, -1).reshape(batch * time, 4)

        lh_pos_world_flat = wrist_positions_world[:, :, 0, :].reshape(batch * time, 3)
        rh_pos_world_flat = wrist_positions_world[:, :, 1, :].reshape(batch * time, 3)
        lh_ori_world_flat = wrist_orientations_world[:, :, 0, :].reshape(batch * time, 4)
        rh_ori_world_flat = wrist_orientations_world[:, :, 1, :].reshape(batch * time, 4)

        lh_pos_rel = (SO3(lh0_quat_world_flat).inverse() @ (lh_pos_world_flat - lh0_pos_world_flat)).reshape(
            batch, time, 3
        )
        rh_pos_rel = (SO3(rh0_quat_world_flat).inverse() @ (rh_pos_world_flat - rh0_pos_world_flat)).reshape(
            batch, time, 3
        )

        lh_ori_rel = (SO3(lh0_quat_world_flat).inverse() @ SO3(lh_ori_world_flat)).wxyz.reshape(
            batch, time, 4
        )
        rh_ori_rel = (SO3(rh0_quat_world_flat).inverse() @ SO3(rh_ori_world_flat)).wxyz.reshape(
            batch, time, 4
        )

        if keep_global_z:
            lh_pos_rel = torch.cat(
                [lh_pos_rel[..., :2], wrist_positions_world[:, :, 0, 2:3]],
                dim=-1,
            )
            rh_pos_rel = torch.cat(
                [rh_pos_rel[..., :2], wrist_positions_world[:, :, 1, 2:3]],
                dim=-1,
            )

        pos_rel = torch.stack([lh_pos_rel, rh_pos_rel], dim=2)
        ori_rel = torch.stack([lh_ori_rel, rh_ori_rel], dim=2)

        base = torch.cat(
            [
                pos_rel.reshape(batch, time, 6),
                ori_rel.reshape(batch, time, 8),
            ],
            dim=-1,
        )
        return base

    def _build_wrist_cond_rh_first(
        self,
        wrist_positions_world: Float[Tensor, "batch time 2 3"],
        wrist_orientations_world: Float[Tensor, "batch time 2 4"],
        mask: Bool[Tensor, "batch time"],
        keep_global_z: bool,
    ) -> Float[Tensor, "batch time 21"]:
        """Build wrist conditioning in each hand's first-frame coordinates, plus hand comm."""
        base = self._build_wrist_cond_rh_first_base(
            wrist_positions_world=wrist_positions_world,
            wrist_orientations_world=wrist_orientations_world,
            mask=mask,
            keep_global_z=keep_global_z,
        )
        hand_comm = self._build_wrist_cond_hand_comm(
            wrist_positions_world=wrist_positions_world,
            wrist_orientations_world=wrist_orientations_world,
        )
        return torch.cat([base, hand_comm], dim=-1)

    @staticmethod
    def _build_wrist_cond_hand_comm(
        wrist_positions_world: Float[Tensor, "batch time 2 3"],
        wrist_orientations_world: Float[Tensor, "batch time 2 4"],
    ) -> Float[Tensor, "batch time 7"]:
        """Left-right communication in left frame: [rel_pos(3), rel_ori(4)]."""
        batch, time, _, _ = wrist_positions_world.shape
        lr_pos_world = wrist_positions_world[:, :, 1, :] - wrist_positions_world[:, :, 0, :]
        lr_pos_world = lr_pos_world.reshape(batch * time, 3)
        lh_quat_world = wrist_orientations_world[:, :, 0, :].reshape(batch * time, 4)
        rh_quat_world = wrist_orientations_world[:, :, 1, :].reshape(batch * time, 4)

        lr_pos_left = (SO3(lh_quat_world).inverse() @ lr_pos_world).reshape(batch, time, 3)
        lr_ori_left = (
            SO3(lh_quat_world).inverse() @ SO3(rh_quat_world)
        ).wxyz.reshape(batch, time, 4)
        return torch.cat([lr_pos_left, lr_ori_left], dim=-1)

    @staticmethod
    def _build_wrist_cond_hand_comm_deltas(
        wrist_positions_world: Float[Tensor, "batch time 2 3"],
        wrist_orientations_world: Float[Tensor, "batch time 2 4"],
    ) -> Float[Tensor, "batch time 7"]:
        """Left-right communication per-frame delta in previous LR frame."""
        batch, time, _, _ = wrist_positions_world.shape
        comm_deltas = torch.zeros((batch, time, 7), device=wrist_positions_world.device, dtype=wrist_positions_world.dtype)

        lr_pos_world = wrist_positions_world[:, :, 1, :] - wrist_positions_world[:, :, 0, :]
        lh_quat_world = wrist_orientations_world[:, :, 0, :].reshape(batch * time, 4)
        rh_quat_world = wrist_orientations_world[:, :, 1, :].reshape(batch * time, 4)
        lr_ori = (SO3(lh_quat_world).inverse() @ SO3(rh_quat_world)).wxyz.reshape(
            batch, time, 4
        )
        lr_pos_left = (SO3(lh_quat_world).inverse() @ lr_pos_world.reshape(
            batch * time, 3
        )).reshape(batch, time, 3)

        if time > 1:
            lr_pos_delta_world = lr_pos_left[:, 1:, :] - lr_pos_left[:, :-1, :]
            lr_ori_prev = SO3(lr_ori[:, :-1, :].reshape((batch * (time - 1), 4)))
            lr_pos_delta_flat = lr_pos_delta_world.reshape((batch * (time - 1), 3))
            comm_deltas[:, 1:, :3] = (
                lr_ori_prev.inverse() @ lr_pos_delta_flat
            ).reshape((batch, time - 1, 3))

            lr_ori_next_flat = SO3(lr_ori[:, 1:, :].reshape((batch * (time - 1), 4)))
            comm_deltas[:, 1:, 3:] = (lr_ori_prev.inverse() @ lr_ori_next_flat).wxyz.reshape(
                (batch, time - 1, 4)
            )

        return comm_deltas

    def _build_wrist_cond_ours(
        self,
        wrist_positions_world: Float[Tensor, "batch time 2 3"],
        wrist_orientations_world: Float[Tensor, "batch time 2 4"],
        mask: Bool[Tensor, "batch time"],
    ) -> Float[Tensor, "batch time 37"]:
        """Relative wrist deltas + global-z + absolute hand comm + first-frame-relative base.

        Inspired by the original cond_param 'ours' which combines:
          - frame-to-frame deltas  (local motion)
          - global height          (absolute vertical reference)
          - canonicalized rotation (absolute orientation context)

        Here we analogously combine:
          - per-hand frame-to-frame pos/ori deltas  (local motion, 14D)
          - per-hand global z height                 (absolute vertical, 2D)
          - absolute hand communication              (inter-hand spatial, 7D)
          - first-frame-relative pos+ori per hand    (absolute context, 14D)

        Layout: [pos_deltas(6), ori_deltas(8), global_z(2), hand_comm(7), rh_first_base(14)]
        """
        batch, time, _, _ = wrist_positions_world.shape

        pos_deltas = torch.zeros_like(wrist_positions_world)
        ori_deltas = torch.zeros_like(wrist_orientations_world)
        ori_deltas[..., 0] = 1.0  # identity quaternion for t=0

        if time > 1:
            # Mask-aware deltas: only compute where both current and previous
            # frames are valid.  Invalid positions are zeroed (pos) or set to
            # identity quaternion (ori) so they don't leak garbage into the
            # conditioning signal that other timesteps attend to.
            delta_mask = mask[:, 1:] & mask[:, :-1]  # (batch, time-1)
            delta_mask_expand = delta_mask.unsqueeze(-1)  # (batch, time-1, 1)

            for hand_idx in range(2):
                # Translation deltas projected into previous frame.
                pos_delta_world = (
                    wrist_positions_world[:, 1:, hand_idx, :]
                    - wrist_positions_world[:, :-1, hand_idx, :]
                )
                prev_quat = wrist_orientations_world[:, :-1, hand_idx, :]
                prev = SO3(prev_quat.reshape((batch * (time - 1), 4)))
                pos_delta_rel = (
                    prev.inverse() @ pos_delta_world.reshape((batch * (time - 1), 3))
                ).reshape((batch, time - 1, 3))
                pos_deltas[:, 1:, hand_idx, :] = pos_delta_rel * delta_mask_expand

                # Orientation deltas.
                ori_prev = SO3(
                    wrist_orientations_world[:, :-1, hand_idx, :].reshape(
                        (batch * (time - 1), 4)
                    )
                )
                ori_next = SO3(
                    wrist_orientations_world[:, 1:, hand_idx, :].reshape(
                        (batch * (time - 1), 4)
                    )
                )
                ori_delta = (ori_prev.inverse() @ ori_next).wxyz.reshape(
                    (batch, time - 1, 4)
                )
                identity_q = torch.zeros_like(ori_delta)
                identity_q[..., 0] = 1.0
                ori_deltas[:, 1:, hand_idx, :] = torch.where(
                    delta_mask_expand.expand_as(ori_delta), ori_delta, identity_q
                )

        # Per-hand global z height (absolute vertical reference).
        global_z = wrist_positions_world[:, :, :, 2:3].reshape(batch, time, 2)

        # Absolute hand communication (inter-hand relationship each frame).
        hand_comm = self._build_wrist_cond_hand_comm(
            wrist_positions_world=wrist_positions_world,
            wrist_orientations_world=wrist_orientations_world,
        )

        # First-frame-relative positions + orientations per hand.
        # This provides the model with absolute spatial context (analogous to
        # the canonicalized rotation in the original cond_param 'ours'),
        # ensuring that even at t=0 the model knows where each hand is.
        rh_first_base = self._build_wrist_cond_rh_first_base(
            wrist_positions_world=wrist_positions_world,
            wrist_orientations_world=wrist_orientations_world,
            mask=mask,
            keep_global_z=True,
        )

        out = torch.cat(
            [
                pos_deltas.reshape(batch, time, 6),
                ori_deltas.reshape(batch, time, 8),
                global_z,
                hand_comm,
                rh_first_base,
            ],
            dim=-1,
        )
        assert out.shape == (batch, time, 37), f"ours wrist conditioning expected (batch,time,37), got {out.shape}"
        return out

    def _build_wrist_conditioning(
        self,
        wrist_positions_world: Float[Tensor, "batch time 2 3"],
        wrist_orientations_world: Float[Tensor, "batch time 2 4"],
        mask: Bool[Tensor, "batch time"],
        wrist_cond_param: str,
    ) -> Float[Tensor, "batch time wrist_cond_dim"]:
        """Build wrist conditioning features according to model config."""
        absolute = self._build_wrist_cond_absolute(
            wrist_positions_world=wrist_positions_world,
            wrist_orientations_world=wrist_orientations_world,
        )

        if wrist_cond_param == "absolute":
            return absolute
        elif wrist_cond_param == "rh_first":
            return self._build_wrist_cond_rh_first(
                wrist_positions_world=wrist_positions_world,
                wrist_orientations_world=wrist_orientations_world,
                mask=mask,
                keep_global_z=False,
            )
        elif wrist_cond_param == "rh_first_global_z":
            return self._build_wrist_cond_rh_first(
                wrist_positions_world=wrist_positions_world,
                wrist_orientations_world=wrist_orientations_world,
                mask=mask,
                keep_global_z=True,
            )
        elif wrist_cond_param == "absrel":
            rel = self._build_wrist_cond_rh_first_base(
                wrist_positions_world=wrist_positions_world,
                wrist_orientations_world=wrist_orientations_world,
                mask=mask,
                keep_global_z=True,
            )
            hand_comm = self._build_wrist_cond_hand_comm(
                wrist_positions_world=wrist_positions_world,
                wrist_orientations_world=wrist_orientations_world,
            )
            return torch.cat([absolute, rel, hand_comm], dim=-1)
        elif wrist_cond_param == "ours":
            return self._build_wrist_cond_ours(
                wrist_positions_world=wrist_positions_world,
                wrist_orientations_world=wrist_orientations_world,
                mask=mask,
            )
        else:
            raise ValueError(f"Unknown wrist_cond_param: {wrist_cond_param}")

    def compute_denoising_loss(
        self,
        model: network.EgoDenoiser | DistributedDataParallel | OptimizedModule,
        unwrapped_model: network.EgoDenoiser,
        train_batch: EgoTrainingData,
    ) -> tuple[Tensor, dict[str, Tensor | float]]:
        """Compute a training loss for the EgoDenoiser model with wrist FK error.

        Returns:
            A tuple (loss tensor, dictionary of things to log).
        """
        log_outputs: dict[str, Tensor | float] = {}

        batch, time, num_joints, _ = train_batch.body_quats.shape
        assert num_joints == 21
        
        # Extract root translation and orientation from T_world_root (world frame)
        root_trans_world = train_batch.T_world_root[:, :, 4:7]  # (batch, time, 3)
        root_orient_world = SO3(train_batch.T_world_root[:, :, :4]).as_matrix()  # (batch, time, 3, 3)
        
        gt_body_rotmats = SO3(train_batch.body_quats).as_matrix()
        gt_betas_expanded = train_batch.betas.expand((batch, time, 16))
        
        # ============================================================
        # Compute right hand transform via FK for coordinate frame change
        # The condition's right hand defines the new coordinate system.
        # ============================================================
        assert self.body_model is not None, "Body model is required for right-hand coordinate frame"
        gt_wrist_fk = self._compute_wrist_fk(
            betas=gt_betas_expanded,
            body_rotmats=gt_body_rotmats,
            root_trans=root_trans_world,
            root_orient=root_orient_world,
        )
        
        # Right hand position and orientation in world frame
        rh_pos_world = gt_wrist_fk['positions'][:, :, 1, :]  # (batch, time, 3)
        rh_quat_world = gt_wrist_fk['orientations'][:, :, 1, :]  # (batch, time, 4) wxyz
        
        # R_world_rh: rotation from right-hand frame to world frame
        R_world_rh = SO3(rh_quat_world)
        R_rh_world = R_world_rh.inverse()
        
        # Transform root translation and orientation to right hand frame
        root_trans_rh = R_rh_world @ (root_trans_world - rh_pos_world)  # (batch, time, 3)
        root_orient_rh = (R_rh_world @ SO3.from_matrix(root_orient_world)).as_matrix()  # (batch, time, 3, 3)
        
        if unwrapped_model.config.include_hands:
            assert train_batch.hand_quats is not None
            x_0 = network.EgoDenoiseTraj(
                betas=gt_betas_expanded,
                body_rotmats=gt_body_rotmats,
                contacts=train_batch.contacts,
                hand_rotmats=SO3(train_batch.hand_quats).as_matrix(),
                root_trans=root_trans_rh,
                root_orient=root_orient_rh,
            )
        else:
            x_0 = network.EgoDenoiseTraj(
                betas=gt_betas_expanded,
                body_rotmats=gt_body_rotmats,
                contacts=train_batch.contacts,
                hand_rotmats=None,
                root_trans=root_trans_rh,
                root_orient=root_orient_rh,
            )
        x_0_packed = x_0.pack()
        device = x_0_packed.device
        assert x_0_packed.shape == (batch, time, unwrapped_model.get_d_state())

        # Diffuse.
        t = torch.randint(
            low=1,
            high=unwrapped_model.config.max_t + 1,
            size=(batch,),
            device=device,
        )
        eps = torch.randn(x_0_packed.shape, dtype=x_0_packed.dtype, device=device)
        assert self.noise_constants.alpha_bar_t.shape == (
            unwrapped_model.config.max_t + 1,
        )
        alpha_bar_t = self.noise_constants.alpha_bar_t[t, None, None]
        assert alpha_bar_t.shape == (batch, 1, 1)
        x_t_packed = (
            torch.sqrt(alpha_bar_t) * x_0_packed + torch.sqrt(1.0 - alpha_bar_t) * eps
        )

        hand_positions_wrt_cpf: Tensor | None = None
        if unwrapped_model.config.include_hand_positions_cond:
            # Joints 19 and 20 are the hand positions.
            hand_positions_wrt_cpf = train_batch.joints_wrt_cpf[:, :, 19:21, :].reshape(
                (batch, time, 6)
            )

            # Exclude hand positions for some items in the batch. We'll just do
            # this by passing in zeros.
            hand_positions_wrt_cpf = torch.where(
                # Uniformly drop out with some uniformly sampled probability.
                # :)
                (
                    torch.rand((batch, time, 1), device=device)
                    < torch.rand((batch, 1, 1), device=device)
                ),
                hand_positions_wrt_cpf,
                0.0,
            )

        # Build wrist conditioning features according to model config.
        wrist_positions: Tensor | None = None
        if unwrapped_model.config.use_wrist_conditioning:
            wrist_positions = self._build_wrist_conditioning(
                wrist_positions_world=gt_wrist_fk["positions"],
                wrist_orientations_world=gt_wrist_fk["orientations"],
                mask=train_batch.mask,
                wrist_cond_param=unwrapped_model.config.wrist_cond_param,
            )

        # Denoise.
        x_0_packed_pred = model.forward(
            x_t_packed=x_t_packed,
            t=t,
            T_world_cpf=train_batch.T_world_cpf,
            T_cpf_tm1_cpf_t=train_batch.T_cpf_tm1_cpf_t,
            hand_positions_wrt_cpf=hand_positions_wrt_cpf,
            wrist_positions=wrist_positions,
            project_output_rotmats=False,
            mask=train_batch.mask,
            cond_dropout_keep_mask=torch.rand((batch,), device=device)
            > self.config.cond_dropout_prob
            if self.config.cond_dropout_prob > 0.0
            else None,
        )
        assert isinstance(x_0_packed_pred, torch.Tensor)
        x_0_pred = network.EgoDenoiseTraj.unpack(
            x_0_packed_pred, include_hands=unwrapped_model.config.include_hands
        )

        weight_t = self.weight_t[t].to(device)
        assert weight_t.shape == (batch,)

        def weight_and_mask_loss(
            loss_per_step: Float[Tensor, "b t d"],
            # bt stands for "batch time"
            bt_mask: Bool[Tensor, "b t"] = train_batch.mask,
            bt_mask_sum: Int[Tensor, ""] = torch.sum(train_batch.mask),
        ) -> Float[Tensor, ""]:
            """Weight and mask per-timestep losses (squared errors)."""
            b_local, t_local, _ = loss_per_step.shape
            assert b_local == batch
            assert bt_mask.shape == (batch, t_local)
            assert weight_t.shape == (batch,)
            return (
                # Sum across b axis.
                torch.sum(
                    # Sum across t axis.
                    torch.sum(
                        # Mean across d axis.
                        torch.mean(loss_per_step, dim=-1) * bt_mask,
                        dim=-1,
                    )
                    * weight_t
                )
                / bt_mask_sum
            )

        loss_terms: dict[str, Tensor | float] = {
            "betas": weight_and_mask_loss(
                # (b, t, 16)
                (x_0_pred.betas - x_0.betas) ** 2
                # (16,)
                * x_0.betas.new_tensor(self.config.beta_coeff_weights),
            ),
            "body_rotmats": weight_and_mask_loss(
                # (b, t, 21 * 3 * 3)
                (x_0_pred.body_rotmats - x_0.body_rotmats).reshape(
                    (batch, time, 21 * 3 * 3)
                )
                ** 2,
            ),
            "contacts": weight_and_mask_loss((x_0_pred.contacts - x_0.contacts) ** 2),
            "root_trans": weight_and_mask_loss((x_0_pred.root_trans - x_0.root_trans) ** 2),
            "root_orient": weight_and_mask_loss(
                (x_0_pred.root_orient - x_0.root_orient).reshape((batch, time, 9)) ** 2
            ),
        }

        # ============================================================
        # Compute wrist FK error
        # Transform predicted root from right-hand frame back to world
        # frame, then run FK to get predicted wrist world positions.
        # ============================================================
        if self.body_model is not None and "wrist_fk" in self.config.loss_weights:
            # GT wrist positions are already in world frame (from gt_wrist_fk)
            gt_wrist_positions = gt_wrist_fk['positions']  # (batch, time, 2, 3)
            
            # Transform predicted root back from right-hand frame to world frame
            pred_root_trans_world = (R_world_rh @ x_0_pred.root_trans) + rh_pos_world
            pred_root_orient_world = (R_world_rh @ SO3.from_matrix(x_0_pred.root_orient)).as_matrix()
            
            # Compute predicted wrist positions via FK in world frame
            pred_wrist_fk = self._compute_wrist_fk(
                betas=x_0_pred.betas,
                body_rotmats=x_0_pred.body_rotmats,
                root_trans=pred_root_trans_world,
                root_orient=pred_root_orient_world,
            )
            pred_wrist_positions = pred_wrist_fk['positions']  # (batch, time, 2, 3)
            
            # Compute L2 error for wrist positions
            wrist_error = (pred_wrist_positions - gt_wrist_positions).reshape(batch, time, 6)
            loss_terms["wrist_fk"] = weight_and_mask_loss(wrist_error ** 2)
            
            # Log individual wrist errors for debugging
            left_wrist_error = torch.mean((pred_wrist_positions[:, :, 0, :] - gt_wrist_positions[:, :, 0, :]) ** 2)
            right_wrist_error = torch.mean((pred_wrist_positions[:, :, 1, :] - gt_wrist_positions[:, :, 1, :]) ** 2)
            log_outputs["wrist_fk/left_mse"] = left_wrist_error
            log_outputs["wrist_fk/right_mse"] = right_wrist_error
        else:
            loss_terms["wrist_fk"] = 0.0

        # ============================================================
        # Root translation velocity loss (in right-hand frame)
        # Supervises per-frame displacement so the model learns how
        # much the body moves each frame relative to the right hand.
        # ============================================================
        if "root_vel" in self.config.loss_weights:
            # Per-frame velocity: (batch, time-1, 3)
            pred_root_vel = x_0_pred.root_trans[:, 1:, :] - x_0_pred.root_trans[:, :-1, :]
            gt_root_vel = x_0.root_trans[:, 1:, :] - x_0.root_trans[:, :-1, :]

            root_vel_error = (pred_root_vel - gt_root_vel) ** 2  # (batch, time-1, 3)

            # Mask for velocity: both frames must be valid
            vel_mask = train_batch.mask[:, 1:] & train_batch.mask[:, :-1]  # (batch, time-1)
            vel_mask_sum = torch.maximum(
                torch.sum(vel_mask), torch.tensor(1, device=device)
            )
            loss_terms["root_vel"] = weight_and_mask_loss(
                root_vel_error, bt_mask=vel_mask, bt_mask_sum=vel_mask_sum
            )
            log_outputs["root_vel/mean_speed_gt"] = torch.mean(
                torch.norm(gt_root_vel, dim=-1)
            )
            log_outputs["root_vel/mean_speed_pred"] = torch.mean(
                torch.norm(pred_root_vel, dim=-1)
            )
        else:
            loss_terms["root_vel"] = 0.0

        # Include hand objective.
        # We didn't use this in the paper.
        if unwrapped_model.config.include_hands:
            assert x_0_pred.hand_rotmats is not None
            assert x_0.hand_rotmats is not None
            assert x_0.hand_rotmats.shape == (batch, time, 30, 3, 3)

            # Detect whether or not hands move in a sequence.
            # We should only supervise sequences where the hands are actully tracked / move;
            # we mask out hands in AMASS sequences where they are not tracked.
            gt_hand_flatmat = x_0.hand_rotmats.reshape((batch, time, -1))
            hand_motion = (
                torch.sum(  # (b,) from (b, t)
                    torch.sum(  # (b, t) from (b, t, d)
                        torch.abs(gt_hand_flatmat - gt_hand_flatmat[:, 0:1, :]), dim=-1
                    )
                    # Zero out changes in masked frames.
                    * train_batch.mask,
                    dim=-1,
                )
                > 1e-5
            )
            assert hand_motion.shape == (batch,)

            hand_bt_mask = torch.logical_and(hand_motion[:, None], train_batch.mask)
            loss_terms["hand_rotmats"] = torch.sum(
                weight_and_mask_loss(
                    (x_0_pred.hand_rotmats - x_0.hand_rotmats).reshape(
                        batch, time, 30 * 3 * 3
                    )
                    ** 2,
                    bt_mask=hand_bt_mask,
                    # We want to weight the loss by the number of frames where
                    # the hands actually move, but gradients here can be too
                    # noisy and put NaNs into mixed-precision training when we
                    # inevitably sample too few frames. So we clip the
                    # denominator.
                    bt_mask_sum=torch.maximum(
                        torch.sum(hand_bt_mask), torch.tensor(256, device=device)
                    ),
                )
            )
        else:
            loss_terms["hand_rotmats"] = 0.0

        assert loss_terms.keys() == self.config.loss_weights.keys(), \
            f"Loss terms keys {loss_terms.keys()} != config keys {self.config.loss_weights.keys()}"

        # Log loss terms.
        for name, term in loss_terms.items():
            log_outputs[f"loss_term/{name}"] = term

        # Return loss.
        loss = sum([loss_terms[k] * self.config.loss_weights[k] for k in loss_terms])
        assert isinstance(loss, Tensor)
        assert loss.shape == ()
        log_outputs["train_loss"] = loss

        return loss, log_outputs
