import pandas as pd
import os
import glob # 파일 경로를 다루기 위한 라이브러리

# --------------------------------------------------------------------------
# 1. Parquet 파일들이 저장된 폴더 경로를 입력하세요.
#    예시: folder_path = "./outputs/data/chunk-000/"
# --------------------------------------------------------------------------
file_path = "/home/hyunho_RCI/datasets/gr00t-rl/forge/peg_insert/data/chunk-000/episode_000063.parquet"

df = pd.read_parquet(file_path)

print(df)