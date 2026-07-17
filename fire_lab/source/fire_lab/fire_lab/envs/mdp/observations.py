from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import numpy as np
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.observations import image as _image
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def state(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, :3]
    ee_quat = ee_frame.data.target_quat_w[:, 0, :]

    finger_joint_ids, _ = robot.find_joints("panda_finger_joint.*")
    finger_pos = robot.data.joint_pos[:, finger_joint_ids]
    gripper = torch.cat([finger_pos[:, :1], -finger_pos[:, 1:]], dim=1)

    return torch.cat((ee_pos, ee_quat, gripper), dim=-1)


def task_description(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros((env.num_envs, 1), device=env.device)


def task_index(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros((env.num_envs, 1), device=env.device)


def episode_index(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros((env.num_envs, 1), device=env.device)


def next_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros((env.num_envs, 1), device=env.device)


def next_done(env: ManagerBasedRLEnv) -> torch.Tensor:
    is_last = env.episode_length_buf.unsqueeze(1) == env.max_episode_length_s - 1
    return is_last.to(torch.bool)


def get_vla_observations(
    env: ManagerBasedRLEnv,
    left_camera_cfg: SceneEntityCfg = SceneEntityCfg("left_camera"),
    right_camera_cfg: SceneEntityCfg = SceneEntityCfg("right_camera"),
    wrist_camera_cfg: SceneEntityCfg = SceneEntityCfg("wrist_camera"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    left_img = _image(env=env, sensor_cfg=left_camera_cfg, data_type="rgb", normalize=False)
    right_img = _image(env=env, sensor_cfg=right_camera_cfg, data_type="rgb", normalize=False)
    wrist_img = _image(env=env, sensor_cfg=wrist_camera_cfg, data_type="rgb", normalize=False)

    state_ = state(env=env, ee_frame_cfg=ee_frame_cfg, robot_cfg=robot_cfg)

    return {
        "video.left_view": np.expand_dims(left_img.cpu().numpy().astype(np.uint8), axis=1),
        "video.right_view": np.expand_dims(right_img.cpu().numpy().astype(np.uint8), axis=1),
        "video.wrist_view": np.expand_dims(wrist_img.cpu().numpy().astype(np.uint8), axis=1),
        "state.eef_position": np.expand_dims(state_[:, 0:3].cpu().numpy().astype(np.float64), axis=1),
        "state.eef_quaternion": np.expand_dims(state_[:, 3:7].cpu().numpy().astype(np.float64), axis=1),
        "state.gripper_qpos": np.expand_dims(state_[:, 7:9].cpu().numpy().astype(np.float64), axis=1),
    }
