# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Dict
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from pxr import UsdPhysics

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.operational_space import OperationalSpaceController
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sim.utils import find_matching_prims

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import actions_cfg

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from ...utils import AsyncGr00tInferenceClient
from ...utils import AsyncPi05InferenceClient
from ..observations import get_vla_observations

# import logger
logger = logging.getLogger(__name__)


class DifferentialInverseKinematicsChunkedAction(DifferentialInverseKinematicsAction):

    def __init__(self, cfg: actions_cfg.DifferentialInverseKinematicsChunkedActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self._chunk_size = self.cfg.chunk_size
        self._vla_model_name = self.cfg.vla_model_name
        if self._vla_model_name == "gr00t":
            self._vla_policy = AsyncGr00tInferenceClient(host="localhost", port=self.cfg.vla_server_port)
        elif self._vla_model_name == "pi05":
            self._vla_policy = AsyncPi05InferenceClient(host="localhost", port=self.cfg.vla_server_port)
        else:
            raise ValueError("vla_model_name should be 'gr00t' or 'pi05'")
        self._vla_observations: Dict[str, Any] | None = None
        self._vla_actions: Dict[str, Any] | None = None
        self._processed_vla_actions: torch.Tensor | None = None
        self._episode_length: int = 0
        self._vla_only = self.cfg.vla_only

        if not hasattr(self._env, "vla_shared_buffer"):
            self._env.vla_shared_buffer = {
                "action.eef_position_delta": None,
                "action.eef_rotation_delta": None,
                "action.gripper_close": None,
                "chunk_size": self._chunk_size,
                "vla_only": self._vla_only
            }

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._ik_controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        
        # process vla actions
        if self._episode_length % self._chunk_size == 0:
            if self._episode_length != 0:
                all_vla_actions = self._vla_policy.get_result()
                # save vla actions in global
                self._env.vla_shared_buffer["action.eef_position_delta"] = all_vla_actions["action.eef_position_delta"]
                self._env.vla_shared_buffer["action.eef_rotation_delta"] = all_vla_actions["action.eef_rotation_delta"]
                self._env.vla_shared_buffer["action.gripper_close"] = all_vla_actions["action.gripper_close"]

            vla_actions_pos = torch.tensor(self._env.vla_shared_buffer["action.eef_position_delta"], dtype=torch.float32, device=self.device)
            vla_actions_rot = torch.tensor(self._env.vla_shared_buffer["action.eef_rotation_delta"], dtype=torch.float32, device=self.device)
            vla_actions_delta_pose = torch.cat(
                (vla_actions_pos, vla_actions_rot), dim=-1
            )
            self._processed_vla_actions = vla_actions_delta_pose

        elif self._episode_length % self._chunk_size == int(self._chunk_size / 2):
            self._vla_policy.request_action(get_vla_observations(env=self._env))

        if self._vla_only:
            self._processed_actions[:] = self._scale * self._processed_vla_actions[:, self._episode_length % self._chunk_size, :]
        else:
            self._processed_actions[:] = self._scale * (
                self.raw_actions + self._processed_vla_actions[:, self._episode_length % self._chunk_size, :]
            )

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()

        # set command into controller
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)
        self._episode_length += 1

    def apply_actions(self):
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        # compute the delta in joint-space
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        # set the joint position command
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._episode_length = 0
        self._vla_observations = get_vla_observations(env=self._env)
        all_vla_actions = self._vla_policy.get_action_sync(self._vla_observations)
        self._env.vla_shared_buffer["action.eef_position_delta"] = all_vla_actions["action.eef_position_delta"]
        self._env.vla_shared_buffer["action.eef_rotation_delta"] = all_vla_actions["action.eef_rotation_delta"]
        self._env.vla_shared_buffer["action.gripper_close"] = all_vla_actions["action.gripper_close"]


class BinaryJointPositionChunkedAction(BinaryJointPositionAction):
    """Binary joint action that sets the binary action into joint position targets."""

    cfg: actions_cfg.BinaryJointPositionChunkedActionCfg

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._processed_vla_actions: torch.Tensor | None = None
        self._episode_length: int = 0

    def process_actions(self, actions: torch.Tensor):
        # 1. Raw Actions 저장 (RL Agent의 출력, 보통 -1 ~ 1 사이 값)
        self._raw_actions[:] = actions
        
        chunk_size = self._env.vla_shared_buffer["chunk_size"]
        
        # 2. VLA 데이터 업데이트 (Chunk 단위)
        if self._episode_length % chunk_size == 0:
            # 공유 버퍼에서 가져오기 (0 또는 1)
            vla_raw_output = torch.tensor(
                self._env.vla_shared_buffer["action.gripper_close"],
                dtype=torch.float32, device=self.device
            )
            
            # [핵심] VLA 출력을 Score Space로 변환 (-1.0 ~ 1.0)
            # 가정: 입력 0 -> Close(-1.0), 입력 1 -> Open(+1.0)
            # 수식: (Input * 2.0) - 1.0
            self._processed_vla_actions = (vla_raw_output * 2.0) - 1.0

        # 3. 현재 스텝의 VLA Score 가져오기
        current_step_idx = self._episode_length % chunk_size
        vla_score = self._processed_vla_actions[:, current_step_idx, :]

        # 4. 잔차 적용 (Residual Application)
        if self._env.vla_shared_buffer["vla_only"]:
            combined_score = vla_score
        else:
            # VLA Score(-1 or 1) + RL Action(-1~1 * Scale)
            # Scale을 1.0보다 크게 주면(예: 1.5), RL이 VLA를 완전히 뒤집을 수 있습니다.
            combined_score = vla_score + (actions * self._scale)

        # 5. 최종 판정 (Thresholding)
        # 음수면 Close, 양수면 Open (사용자 코드 로직 유지)
        close_mask = combined_score < 0.0
        
        # 디버깅용 출력 (필요시 주석 해제)
        # print(f"VLA Score: {vla_score[0]}, RL: {actions[0]}, Combined: {combined_score[0]}")
        # print(f"Decision: {'Close' if close_mask[0] else 'Open'}")

        # 6. 실제 로봇 커맨드 결정
        self._processed_actions = torch.where(
            close_mask,
            self._close_command,
            self._open_command
        )

        # 7. 클리핑 (필요한 경우)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        self._episode_length += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._episode_length = 0