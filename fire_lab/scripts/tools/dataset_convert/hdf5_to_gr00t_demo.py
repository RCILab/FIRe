import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import h5py
import pandas as pd
import torch
import torchvision.io as io
from tqdm import tqdm
import numpy as np

def convert_to_gr00t_fixed_physics(h5_path, output_base_dir, start_ep_idx, fps=30, max_length=1000):
    # 모든 데이터를 하나의 chunk 폴더에 저장
    chunk_name = "chunk-000"
    data_dir = os.path.join(output_base_dir, "data", chunk_name)
    video_base_dir = os.path.join(output_base_dir, "videos", chunk_name)
    os.makedirs(data_dir, exist_ok=True)
    
    cam_map = {
        'left_camera': 'observation.images.left_view',
        'right_camera': 'observation.images.right_view',
        'wrist_camera': 'observation.images.wrist_view'
    }
    
    skipped_count = 0
    current_episode_count = 0

    with h5py.File(h5_path, 'r', swmr=True) as f:
        demos = sorted(list(f['data'].keys()))

        for demo_name in tqdm(demos, desc=f"Processing {os.path.basename(h5_path)}"):
            demo_group = f['data'][demo_name]
            obs = demo_group['obs']
            
            curr_len = obs['state'].shape[0]
            if curr_len > max_length:
                skipped_count += 1
                continue

            # 전역 에피소드 번호 계산 (기존 오프셋 + 현재 파일 내 순서)
            global_ep_num = start_ep_idx + current_episode_count
            ep_filename = f"episode_{global_ep_num:06d}"

            pad_len = max_length - curr_len

            # --- 액션 패딩 (팔 0, 그리퍼 유지) ---
            def get_action_padded():
                data = obs['action'][:]
                if pad_len > 0:
                    arm_padding = np.zeros((pad_len, 6), dtype=data.dtype)
                    gripper_last_val = data[-1, 6]
                    gripper_padding = np.full((pad_len, 1), gripper_last_val, dtype=data.dtype)
                    action_padding = np.concatenate([arm_padding, gripper_padding], axis=1)
                    data = np.concatenate([data, action_padding], axis=0)
                return data[:max_length]

            # --- 일반 패딩 ---
            def get_padded_list(key):
                data = obs[key][:]
                if pad_len > 0:
                    padding = np.repeat(data[-1:], pad_len, axis=0)
                    data = np.concatenate([data, padding], axis=0)
                data = data[:max_length]
                return [row.tolist() if isinstance(row, np.ndarray) else row for row in data]

            state_list = get_padded_list('state')
            action_data = get_action_padded()
            action_list = [row.tolist() for row in action_data]
            reward_list = get_padded_list('next_reward')
            validity_list = get_padded_list('validity')
            done_list = [False] * max_length
            done_list[-1] = True

            data_dict = {
                "observation.state": state_list,
                "action": action_list,
                "timestamp": [round(i * (1/fps), 3) for i in range(max_length)],
                "next.reward": reward_list,
                "next.done": done_list,
                "index": [global_ep_num * max_length + i for i in range(max_length)],
                "episode_index": [global_ep_num] * max_length,
                "task_index": [0] * max_length,
                "annotation.human.validity": validity_list,
                "annotation.human.action.task_description": [0] * max_length,
            }
            
            df = pd.DataFrame(data_dict)
            df.to_parquet(os.path.join(data_dir, f"{ep_filename}.parquet"), index=False)

            # 비디오 처리
            for h5_cam, gr00t_cam in cam_map.items():
                if h5_cam in obs:
                    v_dir = os.path.join(video_base_dir, gr00t_cam)
                    os.makedirs(v_dir, exist_ok=True)
                    imgs = obs[h5_cam][:]
                    if pad_len > 0:
                        padding_imgs = np.repeat(imgs[-1:], pad_len, axis=0)
                        imgs = np.concatenate([imgs, padding_imgs], axis=0)
                    imgs = imgs[:max_length]
                    io.write_video(os.path.join(v_dir, f"{ep_filename}.mp4"), 
                                   torch.from_numpy(imgs).to(torch.uint8), fps=fps, video_codec="h264")
            
            current_episode_count += 1
            
    return current_episode_count # 이번 파일에서 처리된 실제 에피소드 수 반환

if __name__ == "__main__":
    # 통합 저장 경로
    final_output_dir = '/datasets/vla_rl/base_line/cube_stack/lerobot'
    total_episodes_processed = 0
    
    h5_file_path = '/datasets/vla_rl/base_line/cube_stack/hdf5/generated_dataset.hdf5'
        
    # 파일이 존재하는지 확인
    if os.path.exists(h5_file_path):
        # 처리된 에피소드 수를 누적하여 다음 파일의 시작 인덱스로 사용
        num_processed = convert_to_gr00t_fixed_physics(
            h5_path=h5_file_path,
            output_base_dir=final_output_dir,
            start_ep_idx=total_episodes_processed,
            fps=60, 
            max_length=350
        )
        total_episodes_processed += num_processed
        print(f"현재까지 총 {total_episodes_processed}개 에피소드 변환 완료.")

    print(f"\n✨ 모든 변환이 완료되었습니다. 총 {total_episodes_processed}개 에피소드 저장됨.")