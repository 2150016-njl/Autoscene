import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def dump_lane_graph_to_json(lane_graph, out_file="lane_graph.json"):
    """
    Export the lane graph topology and geometry to a JSON file.
    """
    data = {}
    for lid, node in lane_graph.lanes.items():
        data[lid] = {
            "centerline": node.centerline.tolist() if node.centerline is not None else [],
            "entry": node.entry,
            "exit": node.exit,
            "left_neighbors": node.left_neighbors,
            "right_neighbors": node.right_neighbors,
            "feature_type": node.feature_type
        }
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Lane graph JSON saved to: {out_file}")

def plot_lane_graph(lane_graph, route_seq=None, ego_pos=None, out_file="lane_graph_viz.png", map_features_list=None):
    """
    Visualize the lane graph using Matplotlib.
    Green: Normal lanes
    Red arrows: Topological links (exit from A to B)
    Blue Bold: Route sequence
    Black Star: Ego position
    Grey: Other map features (if map_features_list is provided)
    """
    plt.figure(figsize=(15, 15))

    # Plot entire raw map features as a background
    if map_features_list:
        for map_id, feat in map_features_list.items():
            pts = None
            if 'polygon' in feat and feat['polygon'] is not None and len(feat['polygon']) > 0:
                pts = np.array(feat['polygon'])
                plt.plot(pts[:, 0], pts[:, 1], color='gray', linewidth=1, alpha=0.3)
                # close loop for polygon
                plt.plot([pts[-1, 0], pts[0, 0]], [pts[-1, 1], pts[0, 1]], color='gray', linewidth=1, alpha=0.3)
            elif 'polyline' in feat and feat['polyline'] is not None and len(feat['polyline']) > 0:
                pts = np.array(feat['polyline'])
                plt.plot(pts[:, 0], pts[:, 1], color='lightgray', linewidth=1, alpha=0.3, linestyle='--')

    route_set = set(route_seq) if route_seq else set()
    for lid, node in lane_graph.lanes.items():
        if node.centerline is None or len(node.centerline) < 2:
            continue
        pts = node.centerline
        # Highlight Route
        if lid in route_set:
            plt.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=4, zorder=5, alpha=0.8, label='Route' if lid==route_seq[0] else "")
        else:
            plt.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=1, zorder=2, alpha=0.5)
        # Draw Connections (Successors)
        for succ_id in node.exit:
            if succ_id in lane_graph.lanes:
                succ_node = lane_graph.lanes[succ_id]
                if succ_node.centerline is not None and len(succ_node.centerline) > 0:
                    start_pt = pts[-1]
                    end_pt = succ_node.centerline[0]
                    # Draw a red arrow for connection
                    plt.annotate(
                        '', xy=(end_pt[0], end_pt[1]), xytext=(start_pt[0], start_pt[1]),
                        arrowprops=dict(arrowstyle="->", color="r", lw=1.5, alpha=0.6),
                        zorder=4
                    )
    if ego_pos is not None:
        plt.plot(ego_pos[0], ego_pos[1], 'k*', markersize=20, zorder=10, label='Ego')
    plt.axis("equal")
    plt.title("Lane Graph Topology")
    # plt.legend()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Lane graph Visualization saved to: {out_file}")


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Set


def visualize_lane_graph_integrity(
        lane_graph,
        out_file: str = "lane_graph_integrity.png",
        highlight_problems: bool = True,
        figsize: tuple = (15, 15)
):
    """
    专门用于检查lane_graph信息完整性的可视化函数

    重点检查：
    1. 车道中心线是否存在
    2. 拓扑连接是否完整
    3. 孤立车道
    4. 断头路

    颜色编码：
    - 绿色：正常车道（有中心线，有连接）
    - 黄色：警告（中心线短或连接少）
    - 红色：严重问题（无中心线或完全孤立）
    - 蓝色箭头：拓扑连接
    - 红色圆圈：孤立车道
    - 紫色虚线：断头路标记
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_facecolor('#f5f5f5')

    # 统计信息
    total_lanes = len(lane_graph.lanes)
    lanes_with_centerline = 0
    lanes_without_centerline = 0
    isolated_lanes = 0
    dead_end_lanes = 0

    all_points = []  # 用于自动确定绘图范围

    # 第一轮：先绘制所有正常的车道中心线
    for lid, node in lane_graph.lanes.items():
        if node.centerline is None or len(node.centerline) < 2:
            lanes_without_centerline += 1
            continue

        pts = node.centerline
        all_points.append(pts)
        lanes_with_centerline += 1

        # 判断车道健康状况
        is_healthy = True
        warnings = []

        # 检查1：中心线点数
        if len(pts) < 3:
            warnings.append("中心线过短")
            is_healthy = False

        # 检查2：拓扑连接
        num_entries = len(node.entry) if hasattr(node, 'entry') else 0
        num_exits = len(node.exit) if hasattr(node, 'exit') else 0

        if num_entries == 0 and num_exits == 0:
            warnings.append("完全孤立")
            isolated_lanes += 1
            is_healthy = False
        elif num_exits == 0:
            warnings.append("断头路")
            dead_end_lanes += 1
            is_healthy = False

        # 根据健康状况确定颜色和线宽
        if not is_healthy and highlight_problems:
            if "完全孤立" in warnings:
                color = '#dc3545'  # 红色：严重问题
                linewidth = 1.5
                alpha = 0.8
                zorder = 4
            elif "断头路" in warnings:
                color = '#6f42c1'  # 紫色：断头路
                linewidth = 1.5
                alpha = 0.8
                zorder = 4
            else:
                color = '#fd7e14'  # 橙色：警告
                linewidth = 1.2
                alpha = 0.7
                zorder = 3
        else:
            color = '#28a745'  # 绿色：健康
            linewidth = 1.0
            alpha = 0.6
            zorder = 2

        # 绘制车道中心线
        ax.plot(pts[:, 0], pts[:, 1], color=color,
                linewidth=linewidth, alpha=alpha, zorder=zorder,
                solid_capstyle='round')

        # 标记车道ID
        mid_idx = len(pts) // 2
        if mid_idx < len(pts):
            # 添加车道ID标签
            ax.text(pts[mid_idx, 0], pts[mid_idx, 1], str(lid),
                    fontsize=6, color=color, alpha=0.8,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor=color, alpha=0.6),
                    zorder=zorder + 1, ha='center', va='center')

            # 如果有警告，在ID旁边显示警告图标
            if warnings and highlight_problems:
                warning_text = "\n".join(warnings)
                ax.text(pts[mid_idx, 0], pts[mid_idx, 1] - 1, "⚠",
                        fontsize=8, color='red', ha='center', va='top',
                        bbox=dict(boxstyle="round,pad=0.1", facecolor="yellow", alpha=0.7),
                        zorder=zorder + 2)

    # 第二轮：绘制拓扑连接
    connection_count = 0
    for lid, node in lane_graph.lanes.items():
        if node.centerline is None or len(node.centerline) < 2:
            continue

        pts = node.centerline
        start_pt = pts[-1]  # 车道终点

        # 绘制出口连接
        for succ_id in node.exit:
            if succ_id in lane_graph.lanes:
                succ_node = lane_graph.lanes[succ_id]
                if succ_node.centerline is not None and len(succ_node.centerline) > 0:
                    end_pt = succ_node.centerline[0]

                    # 计算距离，过滤过长的连接
                    dist = np.linalg.norm(end_pt - start_pt)
                    if dist < 50:  # 合理连接距离阈值
                        # 绘制蓝色箭头
                        ax.annotate('', xy=end_pt, xytext=start_pt,
                                    arrowprops=dict(arrowstyle="->",
                                                    color="#007bff",
                                                    lw=1.0,
                                                    alpha=0.6,
                                                    connectionstyle="arc3,rad=0.1"),
                                    zorder=1)
                        connection_count += 1

    # 第三轮：标记无中心线的车道位置
    if lanes_without_centerline > 0:
        for lid, node in lane_graph.lanes.items():
            if node.centerline is None or len(node.centerline) < 2:
                # 尝试获取车道的其他位置信息
                if hasattr(node, 'boundary') and node.boundary is not None:
                    # 使用边界中心
                    boundary = np.array(node.boundary)
                    center = boundary.mean(axis=0)
                elif hasattr(node, 'reference_point'):
                    # 使用参考点
                    center = node.reference_point[:2]
                else:
                    # 没有位置信息，跳过
                    continue

                # 绘制红色X标记
                ax.plot(center[0], center[1], 'rx', markersize=10,
                        markeredgewidth=2, zorder=5)
                ax.text(center[0], center[1], f"ID:{lid}\nNo CL",
                        fontsize=7, color='red', ha='center', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor='red', alpha=0.7),
                        zorder=6)

    # 设置绘图范围
    if all_points:
        all_pts = np.vstack(all_points)
        x_min, y_min = all_pts.min(axis=0) - 10
        x_max, y_max = all_pts.max(axis=0) + 10
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # 添加统计信息文本框
    stats_text = f"""
    Lane Graph 完整性统计
    ========================
    总车道数: {total_lanes}
    ✓ 有中心线: {lanes_with_centerline}
    ✗ 无中心线: {lanes_without_centerline}
    ⚠ 孤立车道: {isolated_lanes}
    ⚠ 断头路: {dead_end_lanes}
    ➔ 拓扑连接: {connection_count}

    健康度: {lanes_with_centerline / total_lanes * 100:.1f}%
    """

    # 在右上角添加统计信息
    ax.text(0.98, 0.98, stats_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="black", alpha=0.9),
            zorder=100)

    # 添加图例
    legend_elements = [
        mpatches.Patch(facecolor='#28a745', alpha=0.6, label='正常车道'),
        mpatches.Patch(facecolor='#fd7e14', alpha=0.7, label='警告车道'),
        mpatches.Patch(facecolor='#dc3545', alpha=0.8, label='孤立车道'),
        mpatches.Patch(facecolor='#6f42c1', alpha=0.8, label='断头路'),
        Line2D([0], [0], color='#007bff', lw=1, label='拓扑连接'),
        Line2D([0], [0], color='red', marker='x', linestyle='None',
                        markersize=8, label='无中心线')
    ]

    ax.legend(handles=legend_elements, loc='upper left',
              fontsize=9, framealpha=0.9)

    # 设置标题
    ax.set_title(f"Lane Graph 信息完整性检查", fontsize=14, fontweight='bold')
    ax.set_xlabel("X (meters)", fontsize=10)
    ax.set_ylabel("Y (meters)", fontsize=10)

    # 添加网格（可选）
    ax.grid(True, alpha=0.2, linestyle='--', zorder=0)

    # 保存图片
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()

    # 控制台输出详细报告
    print("=" * 60)
    print("LANE GRAPH 完整性检查报告")
    print("=" * 60)
    print(f"总车道数: {total_lanes}")
    print(f"有中心线的车道: {lanes_with_centerline} ({lanes_with_centerline / total_lanes * 100:.1f}%)")
    print(f"无中心线的车道: {lanes_without_centerline} ({lanes_without_centerline / total_lanes * 100:.1f}%)")
    print(f"孤立车道: {isolated_lanes}")
    print(f"断头路: {dead_end_lanes}")
    print(f"拓扑连接数: {connection_count}")
    print("-" * 60)

    if lanes_without_centerline > 0:
        print("❌ 发现无中心线的车道ID:")
        for lid, node in lane_graph.lanes.items():
            if node.centerline is None or len(node.centerline) < 2:
                print(f"  - 车道 {lid}")

    if isolated_lanes > 0:
        print("⚠ 发现孤立车道（无进出连接）")

    if dead_end_lanes > 0:
        print("⚠ 发现断头路（无出口）")

    print("=" * 60)
    print(f"可视化已保存: {out_file}")

    # 返回统计数据供程序使用
    return {
        'total_lanes': total_lanes,
        'lanes_with_centerline': lanes_with_centerline,
        'lanes_without_centerline': lanes_without_centerline,
        'isolated_lanes': isolated_lanes,
        'dead_end_lanes': dead_end_lanes,
        'connection_count': connection_count,
        'health_percentage': lanes_with_centerline / total_lanes * 100
    }


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MPLPolygon
from typing import List, Optional, Tuple


def plot_focused_lane_graph(
        lane_graph,
        route_seq: Optional[List[int]] = None,
        ego_pos: Optional[np.ndarray] = None,
        out_file: str = "focused_lane_graph.png",
        focus_radius: float = 100.0,  # 聚焦半径100米
        figsize: tuple = (12, 12)
):
    """
    聚焦路由附近的简化版车道图可视化

    特点：
    1. 不显示车道数字ID标签
    2. 自动聚焦路由附近100米范围
    3. 简化颜色和样式

    颜色编码：
    - 亮蓝色：路由车道
    - 浅绿色：非路由车道
    - 红色：拓扑连接
    - 黑色：自车位置
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_facecolor('#f8f9fa')

    # 获取路由车道
    route_set = set(route_seq) if route_seq else set()
    route_lanes = []
    all_lanes = []

    # 收集所有点用于计算范围
    all_points = []

    for lid, node in lane_graph.lanes.items():
        if node.centerline is None or len(node.centerline) < 2:
            continue

        pts = node.centerline
        all_points.append(pts)

        if lid in route_set:
            route_lanes.append(pts)
        else:
            all_lanes.append(pts)

    # 计算聚焦中心点
    if ego_pos is not None and len(ego_pos) >= 2:
        center_x, center_y = ego_pos[0], ego_pos[1]
    elif route_lanes:
        # 计算路由车道的中心
        route_points = np.vstack(route_lanes)
        center_x, center_y = route_points.mean(axis=0)
    elif all_points:
        all_pts = np.vstack(all_points)
        center_x, center_y = all_pts.mean(axis=0)
    else:
        center_x, center_y = 0, 0

    # 设置聚焦范围（中心点 ± 100米）
    x_min, x_max = center_x - focus_radius, center_x + focus_radius
    y_min, y_max = center_y - focus_radius, center_y + focus_radius
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 绘制非路由车道（简化，不显示ID）
    for lid, node in lane_graph.lanes.items():
        if node.centerline is None or len(node.centerline) < 2:
            continue

        pts = node.centerline

        # 只绘制在焦点范围内的车道
        pts_in_view = []
        for pt in pts:
            if x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max:
                pts_in_view.append(pt)

        if len(pts_in_view) < 2:
            continue

        pts_in_view = np.array(pts_in_view)

        if lid in route_set:
            # 路由车道 - 亮蓝色
            ax.plot(pts_in_view[:, 0], pts_in_view[:, 1],
                    color='#007bff', linewidth=2.5, alpha=0.9, zorder=3)
        else:
            # 非路由车道 - 浅绿色
            ax.plot(pts_in_view[:, 0], pts_in_view[:, 1],
                    color='#20c997', linewidth=1.0, alpha=0.4, zorder=2)

    # 绘制拓扑连接（只绘制在视野内的）
    for lid, node in lane_graph.lanes.items():
        if node.centerline is None or len(node.centerline) < 2:
            continue

        pts = node.centerline
        if len(pts) == 0:
            continue

        start_pt = pts[-1]  # 车道终点

        # 检查起点是否在视野内
        if not (x_min <= start_pt[0] <= x_max and y_min <= start_pt[1] <= y_max):
            continue

        for succ_id in node.exit:
            if succ_id in lane_graph.lanes:
                succ_node = lane_graph.lanes[succ_id]
                if succ_node.centerline is not None and len(succ_node.centerline) > 0:
                    end_pt = succ_node.centerline[0]

                    # 检查终点是否在视野内
                    if not (x_min <= end_pt[0] <= x_max and y_min <= end_pt[1] <= y_max):
                        continue

                    # 绘制红色箭头
                    ax.annotate('', xy=end_pt, xytext=start_pt,
                                arrowprops=dict(arrowstyle="->",
                                                color="#dc3545",
                                                lw=1.0,
                                                alpha=0.6),
                                zorder=4)

    # 绘制自车位置
    if ego_pos is not None and len(ego_pos) >= 2:
        if x_min <= ego_pos[0] <= x_max and y_min <= ego_pos[1] <= y_max:
            # 绘制车辆轮廓
            car_length, car_width = 4.8, 1.8

            # 车辆矩形
            corners = np.array([
                [car_length / 2, car_width / 2],
                [car_length / 2, -car_width / 2],
                [-car_length / 2, -car_width / 2],
                [-car_length / 2, car_width / 2]
            ])

            # 假设朝向为0度
            corners += ego_pos[:2]

            # 绘制车辆
            car_poly = MPLPolygon(corners, closed=True,
                                  facecolor='#212529', edgecolor='black',
                                  linewidth=2, alpha=0.9, zorder=10)
            ax.add_patch(car_poly)

            # 车辆中心点
            ax.plot(ego_pos[0], ego_pos[1], 'w.', markersize=4, zorder=11)

    # 添加焦点范围指示
    focus_circle = plt.Circle((center_x, center_y), focus_radius,
                              color='gray', fill=False, linestyle='--',
                              linewidth=0.5, alpha=0.3, zorder=0)
    ax.add_patch(focus_circle)

    # 添加范围标签
    ax.text(center_x + focus_radius * 0.7, center_y, f'{focus_radius}m',
            fontsize=8, color='gray', alpha=0.6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            zorder=1)

    # 添加图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color='#007bff', lw=2.5, label='路由车道'),
        Line2D([0], [0], color='#20c997', lw=1.0, label='其他车道'),
        Line2D([0], [0], color='#dc3545', lw=1, marker='>',
               markersize=8, label='车道连接'),
        Patch(facecolor='#212529', edgecolor='black', label='自车'),
        Line2D([0], [0], color='gray', linestyle='--', lw=0.5, label=f'范围 ({focus_radius}m)')
    ]

    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=9, framealpha=0.9)

    # 设置标题和标签
    route_info = f" (路由: {len(route_set)}条车道)" if route_seq else ""
    ax.set_title(f"车道图可视化{route_info}\n范围: {focus_radius}m",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)

    # 添加网格
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, zorder=0)

    # 保存图片
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 简化版车道图已保存: {out_file}")
    print(f"   范围: {x_min:.1f} ~ {x_max:.1f}, {y_min:.1f} ~ {y_max:.1f}")
    print(f"   中心点: ({center_x:.1f}, {center_y:.1f})")


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Set
import warnings


def visualize_lane_graph_by_type(
        lane_graph,
        map_features_list: Optional[Dict] = None,
        route_seq: Optional[List[int]] = None,
        ego_pos: Optional[np.ndarray] = None,
        out_file: str = "lane_graph_by_type.png",
        focus_radius: float = 500.0,
        figsize: tuple = (14, 14)
):
    """
    按类型分类绘制车道图，不同类型使用不同颜色

    支持的地图元素类型和颜色编码：

    === 车道类型 (LANE) ===
    - LANE_SURFACE_STREET       : 蓝色 (#007bff)    - 结构化街道车道
    - LANE_SURFACE_UNSTRUCTURE  : 橙色 (#fd7e14)    - 非结构化区域
    - LANE_SURFACE_PARKING      : 紫色 (#6f42c1)    - 停车区域
    - LANE_SURFACE_SIDEWALK     : 棕色 (#a0522d)    - 人行道
    - LANE_SURFACE_BIKE         : 青色 (#20c997)    - 自行车道
    - 默认车道                  : 绿色 (#28a745)    - 其他车道类型

    === 道路标线类型 (ROAD_LINE) ===
    - ROAD_LINE_BROKEN_SINGLE_WHITE  : 白色虚线 (#ffffff)   - 白色虚线
    - ROAD_LINE_SOLID_SINGLE_WHITE   : 白色实线 (#f8f9fa)   - 白色实线
    - ROAD_LINE_SOLID_DOUBLE_WHITE   : 白色双实线 (#e9ecef)  - 白色双实线
    - ROAD_LINE_BROKEN_SINGLE_YELLOW : 黄色虚线 (#ffc107)   - 黄色虚线
    - ROAD_LINE_SOLID_SINGLE_YELLOW  : 黄色实线 (#ffca2c)   - 黄色实线
    - ROAD_LINE_SOLID_DOUBLE_YELLOW  : 黄色双实线 (#e0a800)  - 黄色双实线

    === 特殊元素 ===
    - STOP_LINE                  : 红色 (#dc3545)     - 停止线
    - CROSSWALK                  : 黄色+斜线 (#fff3cd) - 斑马线
    - SPEED_BUMP                 : 棕色 (#8b4513)     - 减速带
    - 路由车道                   : 亮蓝色+加粗 (#0056b3) - 规划路径
    - 自车位置                   : 黑色星号            - 车辆位置
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_facecolor('#2b2b2b')  # 深色背景提高对比度

    # 类型到颜色的映射字典
    type_to_color = {
        # 车道类型
        'LANE_SURFACE_STREET': '#007bff',  # 蓝色
        'LANE_SURFACE_UNSTRUCTURE': '#fd7e14',  # 橙色
        'LANE_SURFACE_PARKING': '#6f42c1',  # 紫色
        'LANE_SURFACE_SIDEWALK': '#a0522d',  # 棕色
        'LANE_SURFACE_BIKE': '#20c997',  # 青色
        'LANE': '#28a745',  # 绿色（默认）

        # 道路标线
        'ROAD_LINE_BROKEN_SINGLE_WHITE': '#ffffff',  # 白色
        'ROAD_LINE_SOLID_SINGLE_WHITE': '#f8f9fa',  # 浅白
        'ROAD_LINE_SOLID_DOUBLE_WHITE': '#e9ecef',  # 灰白
        'ROAD_LINE_BROKEN_SINGLE_YELLOW': '#ffc107',  # 黄色
        'ROAD_LINE_SOLID_SINGLE_YELLOW': '#ffca2c',  # 亮黄
        'ROAD_LINE_SOLID_DOUBLE_YELLOW': '#e0a800',  # 深黄
        'ROAD_LINE': '#adb5bd',  # 灰色（默认）

        # 特殊元素
        'STOP_LINE': '#dc3545',  # 红色
        'CROSSWALK': '#fff3cd',  # 浅黄
        'SPEED_BUMP': '#8b4513',  # 棕色
    }

    # 类型到线型的映射
    type_to_linestyle = {
        'BROKEN': '--',  # 虚线
        'SOLID': '-',  # 实线
        'DOUBLE': ':',  # 点线（表示双线）
        'default': '-'  # 默认实线
    }

    # 确定中心点和范围
    center_x, center_y = 0, 0
    if ego_pos is not None and len(ego_pos) >= 2:
        center_x, center_y = ego_pos[0], ego_pos[1]

    x_min, x_max = center_x - focus_radius, center_x + focus_radius
    y_min, y_max = center_y - focus_radius, center_y + focus_radius
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 统计不同类型数量
    type_counts = {}

    # 首先绘制地图特征（背景）
    if map_features_list:
        for map_id, feat in map_features_list.items():
            # 获取特征类型
            feat_type = feat.get('type', 'unknown')
            feat_type_str = str(feat_type).upper()

            # 确定颜色
            color = type_to_color.get(feat_type_str, '#6c757d')  # 灰色默认

            # 确定线型
            linestyle = '-'
            for style_key in type_to_linestyle:
                if style_key in feat_type_str and style_key != 'default':
                    linestyle = type_to_linestyle[style_key]
                    break

            # 绘制多边形
            if 'polygon' in feat and feat['polygon'] is not None and len(feat['polygon']) > 0:
                pts = np.array(feat['polygon'])
                # 只绘制在视野内的
                pts_in_view = []
                for pt in pts:
                    if x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max:
                        pts_in_view.append(pt)

                if len(pts_in_view) >= 2:
                    pts_in_view = np.array(pts_in_view)
                    # 特殊处理斑马线
                    if 'CROSSWALK' in feat_type_str:
                        # 斑马线使用填充+斜线
                        from matplotlib.patches import Polygon
                        poly = Polygon(pts_in_view, closed=True,
                                       facecolor=color, edgecolor='#ffc107',
                                       linewidth=1, alpha=0.7, hatch='//', zorder=1)
                        ax.add_patch(poly)
                    else:
                        ax.plot(pts_in_view[:, 0], pts_in_view[:, 1],
                                color=color, linewidth=1.5, alpha=0.5,
                                linestyle=linestyle, zorder=1)

                    # 统计
                    type_counts[feat_type_str] = type_counts.get(feat_type_str, 0) + 1

            # 绘制折线
            elif 'polyline' in feat and feat['polyline'] is not None and len(feat['polyline']) > 0:
                pts = np.array(feat['polyline'])
                # 只绘制在视野内的
                pts_in_view = []
                for pt in pts:
                    if x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max:
                        pts_in_view.append(pt)

                if len(pts_in_view) >= 2:
                    pts_in_view = np.array(pts_in_view)
                    linewidth = 2.0 if 'DOUBLE' in feat_type_str else 1.5

                    # 停止线加粗
                    if 'STOP_LINE' in feat_type_str:
                        linewidth = 3.0
                        alpha = 0.9
                    else:
                        alpha = 0.7

                    ax.plot(pts_in_view[:, 0], pts_in_view[:, 1],
                            color=color, linewidth=linewidth, alpha=alpha,
                            linestyle=linestyle, zorder=2)

                    # 统计
                    type_counts[feat_type_str] = type_counts.get(feat_type_str, 0) + 1

    # 绘制车道图
    route_set = set(route_seq) if route_seq else set()
    lane_type_counts = {}

    for lid, node in lane_graph.lanes.items():
        if node.centerline is None or len(node.centerline) < 2:
            continue

        pts = node.centerline

        # 获取车道类型
        lane_type = 'LANE'
        if hasattr(node, 'feature_type'):
            lane_type = str(node.feature_type).upper()
        elif hasattr(node, 'lane_type'):
            lane_type = str(node.lane_type).upper()

        # 确定颜色
        color = type_to_color.get(lane_type, type_to_color['LANE'])

        # 是否是路由车道
        is_route = lid in route_set

        # 只绘制在视野内的
        pts_in_view = []
        for pt in pts:
            if x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max:
                pts_in_view.append(pt)

        if len(pts_in_view) < 2:
            continue

        pts_in_view = np.array(pts_in_view)

        # 设置线宽和样式
        if is_route:
            # 路由车道加粗高亮
            linewidth = 3.0
            color = '#0056b3'  # 更深的蓝色
            alpha = 0.9
            zorder = 5
        else:
            linewidth = 1.5
            alpha = 0.7
            zorder = 3

        # 绘制车道中心线
        ax.plot(pts_in_view[:, 0], pts_in_view[:, 1],
                color=color, linewidth=linewidth, alpha=alpha,
                solid_capstyle='round', zorder=zorder)

        # 统计车道类型
        lane_type_counts[lane_type] = lane_type_counts.get(lane_type, 0) + 1

        # 绘制拓扑连接
        if hasattr(node, 'exit'):
            for succ_id in node.exit:
                if succ_id in lane_graph.lanes:
                    succ_node = lane_graph.lanes[succ_id]
                    if succ_node.centerline is not None and len(succ_node.centerline) > 0:
                        start_pt = pts[-1]
                        end_pt = succ_node.centerline[0]

                        # 检查是否在视野内
                        if (x_min <= start_pt[0] <= x_max and y_min <= start_pt[1] <= y_max and
                                x_min <= end_pt[0] <= x_max and y_min <= end_pt[1] <= y_max):
                            ax.annotate('', xy=end_pt, xytext=start_pt,
                                        arrowprops=dict(arrowstyle="->",
                                                        color="#dc3545",
                                                        lw=1.0,
                                                        alpha=0.6,
                                                        connectionstyle="arc3,rad=0.1"),
                                        zorder=4)

    # 绘制自车位置
    if ego_pos is not None and len(ego_pos) >= 2:
        if x_min <= ego_pos[0] <= x_max and y_min <= ego_pos[1] <= y_max:
            # 绘制车辆
            ax.plot(ego_pos[0], ego_pos[1], 'k*',
                    markersize=20, markeredgewidth=2, zorder=10, label='Ego')

            # 添加朝向箭头
            if len(ego_pos) >= 3:  # 有heading信息
                heading = ego_pos[2]
                arrow_length = 5.0
                dx = arrow_length * np.cos(heading)
                dy = arrow_length * np.sin(heading)
                ax.arrow(ego_pos[0], ego_pos[1], dx, dy,
                         head_width=1.5, head_length=2.0, fc='white', ec='black',
                         linewidth=2, zorder=9)

    # 添加范围指示圆
    focus_circle = plt.Circle((center_x, center_y), focus_radius,
                              color='gray', fill=False, linestyle='--',
                              linewidth=0.5, alpha=0.3, zorder=0)
    ax.add_patch(focus_circle)

    # 添加范围标签
    ax.text(center_x + focus_radius * 0.7, center_y, f'{focus_radius}m',
            fontsize=8, color='gray', alpha=0.6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            zorder=1)

    # 创建详细的图例
    legend_elements = []

    # 车道类型图例
    legend_elements.append(Line2D([0], [0], color='#007bff', lw=2, label='街道车道 (STREET)'))
    legend_elements.append(Line2D([0], [0], color='#fd7e14', lw=2, label='非结构化区域 (UNSTRUCTURE)'))
    legend_elements.append(Line2D([0], [0], color='#6f42c1', lw=2, label='停车区 (PARKING)'))
    legend_elements.append(Line2D([0], [0], color='#a0522d', lw=2, label='人行道 (SIDEWALK)'))
    legend_elements.append(Line2D([0], [0], color='#20c997', lw=2, label='自行车道 (BIKE)'))
    legend_elements.append(Line2D([0], [0], color='#28a745', lw=2, label='其他车道'))

    # 道路标线图例
    legend_elements.append(Line2D([0], [0], color='#ffffff', lw=2, label='白色虚线'))
    legend_elements.append(Line2D([0], [0], color='#f8f9fa', lw=2, label='白色实线'))
    legend_elements.append(Line2D([0], [0], color='#ffc107', lw=2, label='黄色虚线'))
    legend_elements.append(Line2D([0], [0], color='#ffca2c', lw=2, label='黄色实线'))

    # 特殊元素图例
    legend_elements.append(Line2D([0], [0], color='#dc3545', lw=3, label='停止线'))
    legend_elements.append(Patch(facecolor='#fff3cd', edgecolor='#ffc107', hatch='//', label='斑马线'))
    legend_elements.append(Line2D([0], [0], color='#8b4513', lw=2, label='减速带'))

    # 路由和拓扑
    legend_elements.append(Line2D([0], [0], color='#0056b3', lw=3, label='路由车道'))
    legend_elements.append(Line2D([0], [0], color='#dc3545', lw=1, marker='>',
                                  markersize=8, label='拓扑连接'))
    legend_elements.append(Line2D([0], [0], color='black', marker='*',
                                  markersize=10, linestyle='None', label='自车位置'))

    # 创建图例（分两列显示）
    legend1 = ax.legend(handles=legend_elements[:12], loc='upper left',
                        fontsize=8, framealpha=0.9, ncol=2)
    ax.add_artist(legend1)  # 需要添加第一个图例

    if len(legend_elements) > 12:
        legend2 = ax.legend(handles=legend_elements[12:], loc='lower left',
                            fontsize=8, framealpha=0.9, ncol=2)

    # 添加统计信息
    stats_text = f"视野范围: {focus_radius}m\n"
    stats_text += f"中心点: ({center_x:.1f}, {center_y:.1f})\n"
    stats_text += f"总车道数: {sum(lane_type_counts.values())}\n"

    for lane_type, count in sorted(lane_type_counts.items()):
        stats_text += f"{lane_type}: {count}\n"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
            zorder=100)

    # 设置标题
    route_info = f" (路由: {len(route_set)}车道)" if route_seq else ""
    ax.set_title(f"车道图类型可视化{route_info}", fontsize=14, fontweight='bold', color='white')
    ax.set_xlabel("X (m)", fontsize=10, color='white')
    ax.set_ylabel("Y (m)", fontsize=10, color='white')

    # 设置坐标轴颜色
    ax.tick_params(colors='white')

    # 添加网格
    ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.5, zorder=0, color='gray')

    # 保存图片
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='#2b2b2b')
    plt.close()

    # 打印统计信息
    print("=" * 60)
    print("车道图类型统计")
    print("=" * 60)
    print(f"总车道数: {sum(lane_type_counts.values())}")
    for lane_type, count in sorted(lane_type_counts.items()):
        print(f"  {lane_type}: {count}")

    if type_counts:
        print(f"\n地图特征统计:")
        for feat_type, count in sorted(type_counts.items()):
            print(f"  {feat_type}: {count}")

    print(f"\n✅ 可视化已保存: {out_file}")
    print(f"   视野范围: {focus_radius}m, 中心: ({center_x:.1f}, {center_y:.1f})")

    return {
        'lane_type_counts': lane_type_counts,
        'feature_type_counts': type_counts,
        'center': (center_x, center_y),
        'radius': focus_radius
    }