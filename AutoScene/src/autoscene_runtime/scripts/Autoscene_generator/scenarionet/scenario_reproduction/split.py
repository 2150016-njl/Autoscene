# # 加载pkl文件
import pickle
# path = '/media/mj/EXTERNAL_USB/Pycharm-Projects/scenarionet-main/scenarionet/new'
path = '/home/mj/PycharmProjects/scenarionet-main/scenario_reproduction/tjwest260528obj_info_1'
with open(f'{path}/dataset_summary.pkl', 'rb') as f:
    dataset_summary = pickle.load(f)
with open(f'{path}/sd_tjwest260528obj_info_1.pkl', 'rb') as f:
    dataset = pickle.load(f)
#
# # 拆分pkl文件
# #!/usr/bin/env python3
import argparse
import os
import pickle
import copy
from typing import Dict, Any

import numpy as np
from metadrive.scenario.scenario_description import ScenarioDescription


# 递归处理 numpy/set -> list/tuple
def dict_recursive_remove_array_and_set(d):
    if hasattr(d, 'tolist'):
        return d.tolist()
    if isinstance(d, set):
        return tuple(d)
    if isinstance(d, dict):
        return {k: dict_recursive_remove_array_and_set(v) for k, v in d.items()}
    if isinstance(d, list):
        return [dict_recursive_remove_array_and_set(i) for i in d]
    return d

# 保存 summary 和 mapping
def save_summary_and_mapping(summary_path, mapping_path, summary, mapping):
    summary = dict_recursive_remove_array_and_set(summary)
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    with open(mapping_path, 'wb') as f:
        pickle.dump(mapping, f)
    print(f"[INFO] Saved {len(summary)} splits to {summary_path}")

# 主函数
def split_scenario(input_path: str, output_dir: str, step: int):
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始场景
    with open(input_path, 'rb') as f:
        scenario: Dict[str, Any] = pickle.load(f)

    # 计算总时间步
    total_steps = None
    for track in scenario['tracks'].values():
        total_steps = len(track['state']['position'])
        break
    if total_steps is None:
        raise ValueError("No tracks found in scenario.")

    stride = step - 30  # 重叠30步
    num_splits = max(1, (total_steps - step + stride) // stride)
    print(f"[INFO] Total steps: {total_steps}, splits with overlap: {num_splits}")

    summary = {}
    mapping = {}
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    saved_idx = 0

    for i in range(num_splits):
        # 深拷贝一份
        sub_scenario = copy.deepcopy(scenario)
        sub_scenario = ScenarioDescription(sub_scenario)

        start = i * stride
        end = min(start + step, total_steps)
        actual_len = end - start
        if actual_len < step:  # 太短就不要了
            continue

        # 裁减length
        sub_scenario['length'] = end - start

        # 裁剪 tracks
        for track in sub_scenario['tracks'].values():
            for k in ['position', 'heading', 'velocity','valid']:
                track['state'][k] = track['state'][k][start:end]

        # 裁剪 dynamic_map_states（如果有）
        if 'dynamic_map_states' in sub_scenario:
            for dynamic_map_states in sub_scenario['dynamic_map_states'].values():
                for k in ['object_state']:
                    dynamic_map_states['state'][k] = dynamic_map_states['state'][k][start:end]

        # 裁剪 metadata
        sub_scenario['metadata']['ts'] = np.arange(0, (end - start)/10, 0.1)
        sub_scenario['metadata']['track_length'] = end - start
        SD = ScenarioDescription
        # add agents summary
        summary_dict = {}
        for track_id, track in sub_scenario[SD.TRACKS].items():
            summary_dict[track_id] = SD.get_object_summary(object_dict=track, object_id=track_id)
        sub_scenario[SD.METADATA][SD.SUMMARY.OBJECT_SUMMARY] = summary_dict
        # count some objects occurrence
        sub_scenario[SD.METADATA][SD.SUMMARY.NUMBER_SUMMARY] = SD.get_number_summary(sub_scenario)

        if sub_scenario[SD.METADATA]['object_summary']['ego']['moving_distance'] > 3:
            sub_scenario['id'] = str(saved_idx)
            # 生成新文件名
            split_name = f"{base_name}_split_{i:03d}.pkl"
            split_path = os.path.join(output_dir, split_name)

            # 保存子场景
            with open(split_path, 'wb') as f:
                pickle.dump(sub_scenario, f)

            # 构造 metadata（子集）
            metadata = copy.deepcopy(sub_scenario.get(ScenarioDescription.METADATA, {}))
            summary[split_name] = metadata
            mapping[split_name] = "./"

            saved_idx += 1

    # 保存 summary 和 mapping
    save_summary_and_mapping(
        os.path.join(output_dir, ScenarioDescription.DATASET.SUMMARY_FILE),
        os.path.join(output_dir, ScenarioDescription.DATASET.MAPPING_FILE),
        summary,
        mapping
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    path = '/home/mj/PycharmProjects/scenarionet-main/scenario_reproduction/tjwest260528obj_info_1/sd_tjwest260528obj_info_1.pkl'
    parser.add_argument('--input', default=path, help='Path to the original .pkl scenario file')
    parser.add_argument('--output_dir', default='./output/tjwest260528obj_info_200', help='Directory to save split scenarios')
    parser.add_argument('--step', type=int, default=200, help='Number of timesteps per split')
    args = parser.parse_args()

    split_scenario(args.input, args.output_dir, args.step)



# #!/usr/bin/env python3
# import argparse
# import os
# import pickle
# import copy
# from typing import Dict, Any, List
#
# import numpy as np
# from metadrive.scenario.scenario_description import ScenarioDescription
#
# # 原来的 dict_recursive_remove_array_and_set 不变
# def dict_recursive_remove_array_and_set(d):
#     if hasattr(d, 'tolist'):
#         return d.tolist()
#     if isinstance(d, set):
#         return tuple(d)
#     if isinstance(d, dict):
#         return {k: dict_recursive_remove_array_and_set(v) for k, v in d.items()}
#     if isinstance(d, list):
#         return [dict_recursive_remove_array_and_set(i) for i in d]
#     return d
#
# def save_summary_and_mapping(summary_path, mapping_path, summary, mapping):
#     summary = dict_recursive_remove_array_and_set(summary)
#     with open(summary_path, 'wb') as f:
#         pickle.dump(summary, f)
#     with open(mapping_path, 'wb') as f:
#         pickle.dump(mapping, f)
#     print(f"[INFO] Saved {len(summary)} splits to {summary_path}")
#
# # 主函数
# def split_scenario(input_path: str,
#                    output_dir: str,
#                    starts: List[int],
#                    length: int):
#     os.makedirs(output_dir, exist_ok=True)
#
#     with open(input_path, 'rb') as f:
#         scenario: Dict[str, Any] = pickle.load(f)
#
#     # 取总步数
#     total_steps = None
#     for track in scenario['tracks'].values():
#         total_steps = len(track['state']['position'])
#         break
#     if total_steps is None:
#         raise ValueError("No tracks found in scenario.")
#
#     summary, mapping = {}, {}
#     base_name = os.path.splitext(os.path.basename(input_path))[0]
#     SD = ScenarioDescription
#
#     for idx, start in enumerate(starts):
#         end = start + length
#         if end > total_steps:
#             print(f"[WARN] segment {idx} exceeds total steps, skip.")
#             continue
#
#         sub = copy.deepcopy(scenario)
#         sub = ScenarioDescription(sub)
#         sub['length'] = length
#
#         # 裁剪 tracks
#         for track in sub['tracks'].values():
#             for k in ['position', 'heading', 'velocity', 'valid']:
#                 track['state'][k] = track['state'][k][start:end]
#
#         # 裁剪 dynamic_map_states
#         if 'dynamic_map_states' in sub:
#             for dms in sub['dynamic_map_states'].values():
#                 for k in ['object_state']:
#                     dms['state'][k] = dms['state'][k][start:end]
#
#         # 更新 metadata
#         sub['metadata']['ts'] = np.arange(0, length / 10, 0.1)
#         sub['metadata']['track_length'] = length
#
#         # summary
#         summary_dict = {}
#         for tid, tr in sub[SD.TRACKS].items():
#             summary_dict[tid] = SD.get_object_summary(object_dict=tr, object_id=tid)
#         sub[SD.METADATA][SD.SUMMARY.OBJECT_SUMMARY] = summary_dict
#         sub[SD.METADATA][SD.SUMMARY.NUMBER_SUMMARY] = SD.get_number_summary(sub)
#
#         # 保存
#         split_name = f"{base_name}_slice_{idx:03d}_{start:06d}.pkl"
#         split_path = os.path.join(output_dir, split_name)
#         with open(split_path, 'wb') as f:
#             pickle.dump(sub, f)
#         summary[split_name] = copy.deepcopy(sub.get(SD.METADATA, {}))
#         mapping[split_name] = "./"
#
#     save_summary_and_mapping(
#         os.path.join(output_dir, ScenarioDescription.DATASET.SUMMARY_FILE),
#         os.path.join(output_dir, ScenarioDescription.DATASET.MAPPING_FILE),
#         summary, mapping
#     )
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input', required=True, help='Path to the original .pkl scenario file')
#     parser.add_argument('--output_dir', required=True, help='Directory to save split scenarios')
#     parser.add_argument('--starts', nargs='+', type=int, required=True,
#                         help='List of start timestamps (frame indices)')
#     parser.add_argument('--length', type=int, default=250,
#                         help='Length of each slice (in frames)')
#     args = parser.parse_args()
#     split_scenario(args.input, args.output_dir, args.starts, args.length)

    # python split.py --input /data_set/Geely_avs_train_obs_trafficflow/dataset/dataset_geely_trafficlights_1/sd_scenario_test_2025-06-24-11-02-56.pkl --output_dir /data_set/UniTraj/unitraj/data_samples/geely_data --starts 0 1800 --length 250




# #!/usr/bin/env python3
# import argparse
# import os
# import pickle
# import copy
#
# def merge_summaries(directories, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#
#     merged_summary = {}
#     merged_mapping = {}
#
#     for d in directories:
#         summary_path = os.path.join(d, 'dataset_summary.pkl')
#         mapping_path = os.path.join(d, 'dataset_mapping.pkl')
#
#         if not (os.path.exists(summary_path) and os.path.exists(mapping_path)):
#             print(f"[WARN] {d} 缺少 summary 或 mapping，跳过")
#             continue
#
#         with open(summary_path, 'rb') as f:
#             summary = pickle.load(f)
#         with open(mapping_path, 'rb') as f:
#             mapping = pickle.load(f)
#
#         # 避免键名冲突，可加前缀（可选）
#         merged_summary.update(summary)
#         merged_mapping.update(mapping)
#         print(f"[INFO] 已合并 {len(summary)} 条记录，来自 {d}")
#
#     # 保存合并后的文件
#     with open(os.path.join(output_dir, 'dataset_summary.pkl'), 'wb') as f:
#         pickle.dump(merged_summary, f)
#     with open(os.path.join(output_dir, 'dataset_mapping.pkl'), 'wb') as f:
#         pickle.dump(merged_mapping, f)
#
#     print(f"[INFO] 合并完成，共 {len(merged_summary)} 条记录，输出到 {output_dir}")
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dirs', nargs='+', required=True,
#                         help='需要合并的目录列表（每个目录里需含 dataset_summary.pkl / dataset_mapping.pkl）')
#     parser.add_argument('--output_dir', required=True,
#                         help='合并后输出目录')
#     args = parser.parse_args()
#     merge_summaries(args.dirs, args.output_dir)


import pickle
import argparse

keys_to_del = ['sd_scenario_test_2025-06-24-11-02-56_split_005.pkl', 'sd_scenario_test_2025-06-24-11-02-56_split_015.pkl']

pkl_path = '/data_set/Geely_avs_train_obs_trafficflow/split/filt_output/dataset_summary.pkl'
out_path = '/data_set/Geely_avs_train_obs_trafficflow/split/3/dataset_summary.pkl'

# 2) 读原始数据
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)  # data 可以是 dict / list / 其他

# 3) 按 key 删除
if isinstance(data, dict):
    for k in keys_to_del:
        data.pop(k, None)
elif isinstance(data, list):
    # 假设列表元素是 dict，且 dict 里有 'id' / 'index' 字段
    data = [d for d in data if str(d.get('id', d.get('index'))) not in keys_to_del]
else:
    raise TypeError("只支持 dict 或 list(dict) 两种结构")

# 4) 保存
with open(out_path, 'wb') as f:
    pickle.dump(data, f)
print(f"已删除 {len(keys_to_del)} 条记录，保存至 {out_path}")