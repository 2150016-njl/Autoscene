import os
import csv
from typing import List, Tuple

class VehicleHistoryLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.history: List[dict] = []
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, frame_id: int, vehicle_id: str, position: Tuple[float, float], heading: float):
        """
        记录每帧自车的历史 position、heading 及当前位置与朝向
        """
        self.history.append({
            'frame_id': frame_id,
            'vehicle_id': vehicle_id,
            'position_x': position[0],
            'position_y': position[1],
            'heading': heading
        })

    def save(self):
        """
        保存所有历史数据到 CSV 文件
        """
        if not self.history:
            return
        with open(self.log_path, 'w', newline='') as csvfile:
            fieldnames = ['frame_id', 'vehicle_id', 'position_x', 'position_y', 'heading']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)

