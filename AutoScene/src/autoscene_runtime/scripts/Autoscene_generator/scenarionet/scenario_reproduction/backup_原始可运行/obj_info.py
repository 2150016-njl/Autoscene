import csv
import re
import pickle as pkl

import numpy as np

# 初始化字典来存储所有车辆和行人的数据
obj_data = {
    'vehicles': {},
    'walkers': {}
}

# 打开CSV文件
with open('Obj_info_batch_1.csv', 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)  # 读取标题行，获取所有列名

    # 遍历标题行，找到所有车辆和行人的数据列
    for header in headers:
        # 使用正则表达式匹配车辆的列
        match_vehicle = re.match(r'field\.vehicle_info_batch(\d+).*actor_pos\.x', header)
        if match_vehicle:
            vehicle_id = int(match_vehicle.group(1))  # 提取车辆ID
            if vehicle_id not in obj_data['vehicles']:
                obj_data['vehicles'][vehicle_id] = {
                    'position': [],
                    'velocity': [],
                    'psi': [],
                    'width': [],
                    'length': []
                }

        # 使用正则表达式匹配行人的列
        match_walker = re.match(r'field\.walker_info_batch(\d+).*actor_pos\.x', header)
        if match_walker:
            walker_id = int(match_walker.group(1))  # 提取行人ID
            if walker_id not in obj_data['walkers']:
                obj_data['walkers'][walker_id] = {
                    'position': [],
                    'velocity': [],
                    'psi': [],
                    'width': [],
                    'length': []
                }

    # 再次遍历文件，这次是数据行
    for row in reader:
        # 处理车辆数据
        for vehicle_id, vehicle_info in obj_data['vehicles'].items():
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_pos.x')
            position_x = row[index] if index != -1 else None
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_pos.y')
            position_y = row[index] if index != -1 else None
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_pos.z')
            position_z = row[index] if index != -1 else None
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_vel.x')
            velocity_x = row[index] if index != -1 else None
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_vel.y')
            velocity_y = row[index] if index != -1 else None
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_vel.z')
            velocity_z = row[index] if index != -1 else None
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_psi')
            psi = row[index] if index != -1 else None
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_width')
            width = row[index] if index != -1 else None
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_length')
            length = row[index] if index != -1 else None

            # 将数据添加到对应的列表中
            vehicle_info['position'].append((float(position_x), -float(position_y), 0))
            vehicle_info['velocity'].append((float(velocity_x), -float(velocity_y), float(velocity_z)))
            # vehicle_info['psi'].append((2*np.pi-float(psi)) % (2*np.pi))  # 为解决右舵驾驶问题地图关于x轴镜像，并进行角度调整
            angle = 2 * np.pi - float(psi)
            vehicle_info['psi'].append((angle + np.pi) % (2 * np.pi) - np.pi)
            # vehicle_info['psi'].append(float(psi))
            vehicle_info['width'].append(float(width))
            vehicle_info['length'].append(float(length))

        # 处理行人数据
        for walker_id, walker_info in obj_data['walkers'].items():
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_pos.x')
            position_x = row[index] if index != -1 else None
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_pos.y')
            position_y = row[index] if index != -1 else None
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_pos.z')
            position_z = row[index] if index != -1 else None
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_vel.x')
            velocity_x = row[index] if index != -1 else None
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_vel.y')
            velocity_y = row[index] if index != -1 else None
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_vel.z')
            velocity_z = row[index] if index != -1 else None
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_psi')
            psi = row[index] if index != -1 else None
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_width')
            width = row[index] if index != -1 else None
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_length')
            length = row[index] if index != -1 else None

            # 将数据添加到对应的列表中
            # walker_info['position'].append((abs(float(position_x)), abs(float(position_y)), float(position_z)))
            walker_info['position'].append((float(position_x), -float(position_y), 0))
            walker_info['velocity'].append((float(velocity_x), -float(velocity_y), float(velocity_z)))
            # walker_info['psi'].append((2*np.pi-float(psi)) % (2*np.pi))
            angle = 2 * np.pi - float(psi)
            walker_info['psi'].append((angle + np.pi) % (2 * np.pi) - np.pi)
            walker_info['width'].append(float(width))
            walker_info['length'].append(float(length))

# 筛除不存在的键值对
obj_data['vehicles'] = {k: v for k, v in obj_data['vehicles'].items() if not all(w == 0 for w in v['width'])}
obj_data['walkers'] = {k: v for k, v in obj_data['walkers'].items() if not all(w == 0 for w in v['width'])}
for vehicle_id, vehicle_info in obj_data['vehicles'].items():
    vehicle_info['position'] = np.array(vehicle_info['position'])
    vehicle_info['velocity'] = np.array(vehicle_info['velocity'])
    vehicle_info['psi'] = np.array(vehicle_info['psi'])
    vehicle_info['width'] = np.array(vehicle_info['width'])
    vehicle_info['length'] = np.array(vehicle_info['length'])
for walker_id, walker_info in obj_data['walkers'].items():
    walker_info['position'] = np.array(walker_info['position'])
    walker_info['velocity'] = np.array(walker_info['velocity'])
    walker_info['psi'] = np.array(walker_info['psi'])
    walker_info['width'] = np.array(walker_info['width'])
    walker_info['length'] = np.array(walker_info['length'])

pickle_data = pkl.dumps(obj_data)
# 保存到文件
with open('obj_info_1.pkl', 'wb') as f:
    f.write(pickle_data)