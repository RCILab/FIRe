# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import TiledCamera

from fire_lab.tasks.direct.base_line.factory import factory_utils
from fire_lab.tasks.direct.base_line.factory.factory_env import FactoryEnv

from fire_lab.tasks.direct.vla.gr00t.factory.factory_gr00t_env_cfg import FactoryGr00tEnvCfg

import pandas as pd
import sys
import os
from torchvision.io import write_video


class FactoryGr00tDemoSaveEnv(FactoryEnv):
    cfg: FactoryGr00tEnvCfg

    def __init__(self, cfg: FactoryGr00tEnvCfg, render_mode: str | None = None, **kwargs):
        # Update number of obs/states
        super().__init__(cfg, render_mode, **kwargs)

        self.demo_save_cfg = self.cfg.demo_save_cfg

        if self.demo_save_cfg:
            self.gr00t_data_buffers = [{} for _ in range(self.num_envs)]
            self.gr00t_img_buffers = [{} for _ in range(self.num_envs)]

            self._initialize_parquet_buffers()
            self._initialize_image_buffers()


    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        self._wrist_camera = TiledCamera(self.cfg.wrist_camera)
        self._left_camera = TiledCamera(self.cfg.left_camera)
        self._right_camera = TiledCamera(self.cfg.right_camera)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        self.scene.sensors["wrist_camera"] = self._wrist_camera
        self.scene.sensors["left_camera"] = self._left_camera
        self.scene.sensors["right_camera"] = self._right_camera

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])

        if self.cfg.is_demo_save:

            self._update_gr00t_observations(
                eef_position=obs_dict["fingertip_pos"],
                eef_quaternion=obs_dict["fingertip_quat"],
                gripper_qpos=self.joint_pos[:, 7:9]
            )

        return {"policy": obs_tensors, "critic": state_tensors}

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        # Save last actions for gr00t data logging
        self.last_actions = action.clone()
        super()._pre_physics_step(action) # This changes action to smoothed action

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )

        rew_buf = torch.where(curr_successes, 1.0, 0.0)

        return rew_buf
    
    def _reset_idx(self, env_ids):
        """Perform additional randomizations."""
        super()._reset_idx(env_ids)

        if self.cfg.is_demo_save:
            self._save_gr00t_observations(
                data_dir=self.demo_save_cfg.data_dir,
                video_dir=self.demo_save_cfg.video_dir,
            )
            self._initialize_parquet_buffers()
    
    def _get_gr00t_observations(self):
        # This is for gr00t observations

        observations = {
            "video.left_view": np.expand_dims(self._left_camera.data.output["rgb"].cpu().numpy().astype(np.uint8), axis=1),
            "video.right_view": np.expand_dims(self._right_camera.data.output["rgb"].cpu().numpy().astype(np.uint8), axis=1),
            "video.wrist_view": np.expand_dims(self._wrist_camera.data.output["rgb"].cpu().numpy().astype(np.uint8), axis=1),
            "state.eef_position": np.expand_dims(self.fingertip_midpoint_pos.cpu().numpy().astype(np.float64), axis=1),
            "state.eef_quaternion": np.expand_dims(self.fingertip_midpoint_quat.cpu().numpy().astype(np.float64), axis=1),
            "state.gripper_qpos": np.expand_dims(self.joint_pos[:, 7:9].cpu().numpy().astype(np.float64), axis=1),
        }
        return observations


    def _initialize_parquet_buffers(self):
        """Clears and initializes the data buffer for a specific environment."""
        for env_id in range(self.num_envs):
            self.gr00t_data_buffers[env_id] = {
                "observation.state": [],
                "action": [],
                "timestamp": [],
                "annotation.human.action.task_description": [],
                "task_index": [],
                "annotation.human.validity": [],
                "episode_index": [],
                "index": [],
                "next.reward": [],
                "next.done": [],
            }

    def _initialize_image_buffers(self):
        """Clears and initializes the image buffer for a specific environment."""
        for env_id in range(self.num_envs):
            self.gr00t_img_buffers[env_id] = {
                "left_camera": [],
                "right_camera": [],
                "wrist_camera": [],
            }

    def _save_gr00t_observations(self, data_dir: str, video_dir: str):
        """Saves the collected gr00t data and videos to the specified directories."""
        # Prevent saving if no data collected: Initial reset
        if len(self.gr00t_data_buffers[0]["observation.state"]) != self.max_episode_length - 2:
            print("No data collected for this episode. Skipping save.")
            return

        for env_id in range(self.num_envs):
            # Save Datas
            df = pd.DataFrame(self.gr00t_data_buffers[env_id])

            gr00t_data_path = os.path.join(
                data_dir, f"episode_{env_id:06d}.parquet"
            )
            os.makedirs(os.path.dirname(gr00t_data_path), exist_ok=True)

            df.to_parquet(gr00t_data_path, index=False)

            # Save Videos
            for camera_name, img_list in self.gr00t_img_buffers[env_id].items():

                img_tensor = torch.stack(img_list, dim=0).to(device=self.device, dtype=torch.uint8) # (T, C, H, W)

                gr00t_video_path = os.path.join(
                    video_dir[camera_name], f"episode_{env_id:06d}.mp4"
                )
                os.makedirs(os.path.dirname(gr00t_video_path), exist_ok=True)

                write_video(gr00t_video_path, img_tensor, fps=int(1/self.step_dt), video_codec="h264")


        print(f"Saved {self.num_envs} episodes to {data_dir} and {video_dir}")
        print(f"number of frames: {len(self.gr00t_data_buffers[0]['observation.state'])}")
        print(f"Exiting program after saving gr00t data...")

        # Quit Program
        sys.exit(0)

    def _update_gr00t_observations(self, eef_position, eef_quaternion, gripper_qpos):

        if self.episode_length_buf[0].item() == 0:
            return # Skip the first step (reset step)

        check_rot = self.cfg_task.name == "nut_thread"
        true_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )

        state = torch.cat((eef_position, eef_quaternion, gripper_qpos), dim=-1)
        arm_action = self.last_actions[:, :6]
        gripper_action = torch.ones((self.num_envs, 1), device=self.device) # This Task forces gripper to be closed
        action = torch.cat((arm_action, gripper_action), dim=-1) # rescale success_pred to [0, 1]

        # Get Camera Datas -> # (num_envs, height, width, channel)
        left_camera_data = self._left_camera.data.output["rgb"].clone()
        right_camera_data = self._right_camera.data.output["rgb"].clone()
        wrist_camera_data = self._wrist_camera.data.output["rgb"].clone()

        print("================== Parquet Data Stats =================")
        print(f"current / max: {self.episode_length_buf[0].item()} / {self.max_episode_length - 2}")
        print(f"success / total: {true_successes.sum().item()} / {self.num_envs}")

        for i in range(self.num_envs):
            current_step_in_ep = self.episode_length_buf[i].item()

            # update data buffers
            self.gr00t_data_buffers[i]["observation.state"].append(state[i].cpu().numpy().tolist())
            self.gr00t_data_buffers[i]["action"].append(action[i].cpu().numpy().tolist()) # action[6] is success_pred
            self.gr00t_data_buffers[i]["timestamp"].append(round(current_step_in_ep * self.step_dt, 3))
            self.gr00t_data_buffers[i]["annotation.human.action.task_description"].append(0) # Task index 0
            self.gr00t_data_buffers[i]["task_index"].append(0)
            self.gr00t_data_buffers[i]["annotation.human.validity"].append(true_successes[i].item())
            self.gr00t_data_buffers[i]["episode_index"].append(i)
            self.gr00t_data_buffers[i]["index"].append(i * (self.max_episode_length - 1) + current_step_in_ep - 1)
            
            if true_successes[i]:
                self.gr00t_data_buffers[i]["next.reward"].append(1.0)
            else:
                self.gr00t_data_buffers[i]["next.reward"].append(0.0)
            
            if current_step_in_ep == self.max_episode_length - 2: # Final Step of episode
                self.gr00t_data_buffers[i]["next.done"].append(True)
            else:
                self.gr00t_data_buffers[i]["next.done"].append(False)

            # update image buffers
            self.gr00t_img_buffers[i]["left_camera"].append(
                left_camera_data[i]
            )
            self.gr00t_img_buffers[i]["right_camera"].append(
                right_camera_data[i]
            )
            self.gr00t_img_buffers[i]["wrist_camera"].append(
                wrist_camera_data[i]
            )

