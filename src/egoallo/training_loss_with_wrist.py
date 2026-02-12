"""Training loss configuration with wrist FK error.

This is an extended version of training_loss.py that adds wrist position error
computed via Forward Kinematics (FK).
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


@dataclasses.dataclass(frozen=True)
class TrainingLossConfigWithWrist:
    """Training loss config with additional wrist FK error."""
    
    cond_dropout_prob: float = 0.0
    beta_coeff_weights: tuple[float, ...] = tuple(1 / (i + 1) for i in range(16))
    loss_weights: dict[str, float] = dataclasses.field(
        default_factory={
            "betas": 0.1,
            "body_rotmats": 1.0,
            "contacts": 0.1,
            # We don't have many hands in the AMASS dataset...
            "hand_rotmats": 0.01,
            "root_trans": 1.0,
            "root_orient": 1.0,
            # Wrist FK error - higher weight as requested
            "wrist_fk": 5.0,
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
        
        # Extract root translation and orientation from T_world_root
        root_trans = train_batch.T_world_root[:, :, 4:7]  # Last 3 elements are translation
        root_orient = SO3(train_batch.T_world_root[:, :, :4]).as_matrix()  # (batch, time, 3, 3)
        
        if unwrapped_model.config.include_hands:
            assert train_batch.hand_quats is not None
            x_0 = network.EgoDenoiseTraj(
                betas=train_batch.betas.expand((batch, time, 16)),
                body_rotmats=SO3(train_batch.body_quats).as_matrix(),
                contacts=train_batch.contacts,
                hand_rotmats=SO3(train_batch.hand_quats).as_matrix(),
                root_trans=root_trans,
                root_orient=root_orient,
            )
        else:
            x_0 = network.EgoDenoiseTraj(
                betas=train_batch.betas.expand((batch, time, 16)),
                body_rotmats=SO3(train_batch.body_quats).as_matrix(),
                contacts=train_batch.contacts,
                hand_rotmats=None,
                root_trans=root_trans,
                root_orient=root_orient,
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

        # Extract wrist positions+orientations for conditioning (joints 19 and 20)
        # Use FK to compute both wrist positions and quaternion orientations
        wrist_positions: Tensor | None = None
        gt_wrist_fk: dict | None = None
        if unwrapped_model.config.use_wrist_conditioning:
            if self.body_model is not None:
                # Compute GT wrist positions + orientations via FK
                gt_wrist_fk = self._compute_wrist_fk(
                    betas=x_0.betas,
                    body_rotmats=x_0.body_rotmats,
                    root_trans=x_0.root_trans,
                    root_orient=x_0.root_orient,
                )
                # 14D: [left_pos(3), right_pos(3), left_quat(4), right_quat(4)]
                wrist_positions = torch.cat([
                    gt_wrist_fk['positions'].reshape((batch, time, 6)),
                    gt_wrist_fk['orientations'].reshape((batch, time, 8)),
                ], dim=-1)  # (batch, time, 14)
            else:
                # Fallback: positions from preprocessed data, identity quaternions
                pos = train_batch.joints_wrt_world[:, :, 19:21, :].reshape((batch, time, 6))
                identity_quats = torch.zeros((batch, time, 8), device=device)
                identity_quats[..., 0] = 1.0  # w=1 for left wrist
                identity_quats[..., 4] = 1.0  # w=1 for right wrist
                wrist_positions = torch.cat([pos, identity_quats], dim=-1)

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
            _, _, d = loss_per_step.shape
            assert loss_per_step.shape == (batch, time, d)
            assert bt_mask.shape == (batch, time)
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
        # ============================================================
        if self.body_model is not None and "wrist_fk" in self.config.loss_weights:
            # Get GT wrist positions (reuse from conditioning if available)
            if gt_wrist_fk is not None:
                gt_wrist_positions = gt_wrist_fk['positions']  # (batch, time, 2, 3)
            else:
                gt_wrist_fk = self._compute_wrist_fk(
                    betas=x_0.betas,
                    body_rotmats=x_0.body_rotmats,
                    root_trans=x_0.root_trans,
                    root_orient=x_0.root_orient,
                )
                gt_wrist_positions = gt_wrist_fk['positions']
            
            # Compute predicted wrist positions using FK
            # Now uses PREDICTED root_orient (model predicts root orientation!)
            pred_wrist_fk = self._compute_wrist_fk(
                betas=x_0_pred.betas,
                body_rotmats=x_0_pred.body_rotmats,
                root_trans=x_0_pred.root_trans,
                root_orient=x_0_pred.root_orient,
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
