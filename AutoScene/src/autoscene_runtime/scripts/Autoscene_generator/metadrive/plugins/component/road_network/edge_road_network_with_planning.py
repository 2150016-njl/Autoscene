"""
添加了一些功能，包括找到没死胡同路径的图结构，随机车道选择功能，a_star_paths功能，
更改了get_peer_lanes_from_index函数和bfs_paths函数，增加
get_closest_lane_index函数，判断给定位置是否在车道内，返回车道索引。
"""
import logging
from collections import namedtuple
from typing import List
import copy

from metadrive.component.road_network.base_road_network import BaseRoadNetwork
from metadrive.component.road_network.base_road_network import LaneIndex
from metadrive.scenario.scenario_description import ScenarioDescription as SD
from metadrive.type import MetaDriveType
from metadrive.utils.math import get_boxes_bounding_box
from metadrive.utils.pg.utils import get_lanes_bounding_box
from typing import List, Generator
from collections import deque

lane_info = namedtuple("edge_lane", ["lane", "entry_lanes", "exit_lanes", "left_lanes", "right_lanes"])


class EdgeRoadNetworkWithPlanning(BaseRoadNetwork):
    """
    Compared to NodeRoadNetwork representing the relation of lanes in a node-based graph, EdgeRoadNetwork stores the
    relationship in edge-based graph, which is more common in real map representation
    """
    def __init__(self):
        super(EdgeRoadNetworkWithPlanning, self).__init__()
        self.graph = {}
        self.no_dead_lane_graph = {}
        self._dead_end_computed = False
        self._pathfinding_neighbor_cache = {}

    def _invalidate_pathfinding_cache(self):
        if self._pathfinding_neighbor_cache is not None:
            self._pathfinding_neighbor_cache.clear()

    @staticmethod
    def _stringify_lane_index(index):
        return None if index is None else str(index)

    def _resolve_graph_key(self, index):
        if index in self.graph:
            return index
        normalized = self._stringify_lane_index(index)
        if normalized in self.graph:
            return normalized
        return normalized

    def _normalize_neighbor_ref(self, neighbor):
        if isinstance(neighbor, dict):
            if "feature_id" in neighbor:
                return self._resolve_graph_key(neighbor["feature_id"])
            if "id" in neighbor:
                return self._resolve_graph_key(neighbor["id"])
        if hasattr(neighbor, "feature_id"):
            return self._resolve_graph_key(neighbor.feature_id)
        if hasattr(neighbor, "lane"):
            lane = neighbor.lane
            if hasattr(lane, "index"):
                return self._resolve_graph_key(lane.index)
        if hasattr(neighbor, "index") and not isinstance(neighbor, str):
            return self._resolve_graph_key(neighbor.index)
        return self._resolve_graph_key(neighbor)

    def _normalize_lane_refs(self, refs):
        if refs is None:
            return []
        if not isinstance(refs, (list, tuple)):
            refs = [refs]
        return [ref for ref in (self._normalize_neighbor_ref(item) for item in refs) if ref is not None]

    def add_lane(self, lane) -> None:
        assert lane.index is not None, "Lane index can not be None"
        lane.index = self._stringify_lane_index(lane.index)
        self.graph[lane.index] = lane_info(
            lane=lane,
            entry_lanes=self._normalize_lane_refs(lane.entry_lanes),
            exit_lanes=self._normalize_lane_refs(lane.exit_lanes),
            left_lanes=self._normalize_lane_refs(lane.left_lanes),
            right_lanes=self._normalize_lane_refs(lane.right_lanes)
        )
        self._invalidate_pathfinding_cache()

    def get_lane(self, index: LaneIndex):
        return self.graph[self._resolve_graph_key(index)].lane

    @staticmethod
    def _normalize_lane_type_filter(excluded_lane_types):
        if excluded_lane_types is None:
            return frozenset()
        if isinstance(excluded_lane_types, str):
            return frozenset([excluded_lane_types])
        return frozenset(excluded_lane_types)

    def _get_lane_object(self, lane_or_index):
        if lane_or_index is None:
            return None
        if hasattr(lane_or_index, "lane"):
            return lane_or_index.lane
        if hasattr(lane_or_index, "index") and not isinstance(lane_or_index, str):
            return lane_or_index
        lane_key = self._resolve_graph_key(lane_or_index)
        if lane_key in self.graph:
            return self.graph[lane_key].lane
        return None

    def is_lane_excluded(self, lane_or_index, excluded_lane_types=None):
        excluded_lane_types = self._normalize_lane_type_filter(excluded_lane_types)
        if not excluded_lane_types:
            return False
        lane = self._get_lane_object(lane_or_index)
        if lane is None:
            return False
        return any(
            lane_type in excluded_lane_types
            for lane_type in (getattr(lane, "lane_type", None), getattr(lane, "metadrive_type", None))
            if lane_type is not None
        )

    def is_bike_lane(self, lane_or_index):
        return self.is_lane_excluded(lane_or_index, {MetaDriveType.LANE_BIKE_LANE})

    def filter_lane_indices(self, lane_indices, excluded_lane_types=None):
        excluded_lane_types = self._normalize_lane_type_filter(excluded_lane_types)
        filtered_indices = []
        for lane_index in lane_indices or []:
            lane_key = self._resolve_graph_key(lane_index)
            if lane_key not in self.graph:
                continue
            if self.is_lane_excluded(lane_key, excluded_lane_types):
                continue
            filtered_indices.append(lane_key)
        return filtered_indices
    
    # def get_lane_lists(self,current_lane):
    #     """
    #     分别获取当前车道左侧和右侧的所有车道列表。
    #     列表顺序均为：由近及远（索引0为紧邻当前车道的车道）。
    #     """
        
    #     # --- 内部辅助函数：通用遍历逻辑 ---
    #     def _collect_lanes(start_node, direction_attr):
    #         lanes_list = []
    #         # 从传入的起点开始（不包含起点本身）
    #         pointer = start_node
            
    #         while True:
    #             # 1. 获取相邻数据 (可能是 list, Lane对象, 或 None)
    #             neighbor_data = getattr(pointer, direction_attr, None)
                
    #             # 2. 解析出单一的车道对象
    #             next_lane = None
    #             if isinstance(neighbor_data, list):
    #                 # 如果返回的是列表，通常第一个元素是最近的邻居
    #                 if len(neighbor_data) > 0:
    #                     next_lane = neighbor_data[0]
    #             else:
    #                 # 如果直接返回对象
    #                 next_lane = neighbor_data
                
    #             # 3. 如果找不到邻居，说明到路边了，停止循环
    #             if next_lane is None:
    #                 break
                
    #             # 4. 加入列表，并将指针移向刚找到的这条车道，继续找它的邻居
    #             lanes_list.append(next_lane)
    #             pointer = self.graph[next_lane]
                
    #         return lanes_list

    #     # --- 执行查找 ---
    #     # 获取所有左侧车道 (L1, L2, L3...)
    #     all_left = _collect_lanes(current_lane, 'left_lanes')
    #     all_left = [sublist for sublist in all_left if len(sublist)!=0]
        
    #     # 获取所有右侧车道 (R1, R2, R3...)
    #     all_right = _collect_lanes(current_lane, 'right_lanes')
    #     all_right = [sublist for sublist in all_right if len(sublist)!=0]
        
    #     return all_left, all_right
    def get_lane_lists(self, current_lane):
        """
        分别获取当前车道左侧和右侧的所有车道列表。
        列表顺序均为：由近及远（索引0为紧邻当前车道的车道）。

        兼容 neighbor 表达形式：
        - lane_index
        - lane 对象（带 .index）
        - dict: {"feature_id": ...}
        - 对象: 带 .feature_id
        - feature_id 本身也可能是 lane 对象或 lane_index
        """

        def _as_lane_index(nei):
            """把各种 neighbor 形式归一化为 graph 的 key（lane_index）。解析失败返回 None。"""
            return self._normalize_neighbor_ref(nei)

        def _collect_lanes(start_node, direction_attr):
            lanes_list = []

            # start_node 可能是 lane_index、lane 对象、或 lane_info(namedtuple)
            start_key = self._normalize_neighbor_ref(start_node)
            if start_key in self.graph:
                pointer = self.graph[start_key]
            else:
                pointer = start_node  # 假设就是 lane_info

            while True:
                neighbor_data = getattr(pointer, direction_attr, None)

                # neighbor_data 可能是 list / 单个值 / None
                if isinstance(neighbor_data, list):
                    if len(neighbor_data) == 0:
                        break
                    next_nei = neighbor_data[0]
                else:
                    next_nei = neighbor_data

                next_lane_index = _as_lane_index(next_nei)
                if next_lane_index is None or next_lane_index not in self.graph:
                    break

                lanes_list.append(next_lane_index)
                pointer = self.graph[next_lane_index]

            return lanes_list

        all_left = _collect_lanes(current_lane, "left_lanes")
        all_right = _collect_lanes(current_lane, "right_lanes")
        return all_left, all_right

    
    def get_random_lane(self):
        from metadrive.utils import get_np_random
        rng = get_np_random()
        index= rng.choice(list(self.graph.keys()))
        return self.graph[index].lane

    def __isub__(self, other):
        for id, lane_info in other.graph.items():
            self.graph.pop(id)
        self._invalidate_pathfinding_cache()
        return self

    def add(self, other, no_intersect=True):
        for id, lane_info in other.graph.items():
            if no_intersect:
                assert id not in self.graph.keys(), "Intersect: {} exists in two network".format(id)
            self.graph[id] = other.graph[id]
        self._invalidate_pathfinding_cache()
        return self

    def _get_bounding_box(self):
        """
       By using this bounding box, the edge length of x, y direction and the center of this road network can be
       easily calculated.
       :return: minimum x value, maximum x value, minimum y value, maximum y value
       """
        lanes = []
        for id, lane_info, in self.graph.items():
            lanes.append(lane_info.lane)
        res_x_max, res_x_min, res_y_max, res_y_min = get_boxes_bounding_box([get_lanes_bounding_box(lanes)])
        return res_x_min, res_x_max, res_y_min, res_y_max

    def shortest_path(self, start: str, goal: str, excluded_lane_types=None):
        return self.fast_shortest_path(start, goal, excluded_lane_types=excluded_lane_types)

    # def shortest_path(self, start: str, goal: str, excluded_lane_types=None):
    #     start = self._resolve_graph_key(start)
    #     goal = self._resolve_graph_key(goal)
    #     return next(
    #         self.bfs_paths(start, goal, excluded_lane_types=excluded_lane_types), []
    #     ) # 如果 bfs_paths 找到至少一条路径，则返回第一条路径；如果没有路径，则返回空列表
    #     # return next(self.a_star_paths(start,goal),[])

    def _get_pathfinding_neighbors(self, lane_id, excluded_lane_types=None):
        excluded_lane_types = self._normalize_lane_type_filter(excluded_lane_types)
        cache = self._pathfinding_neighbor_cache.get(excluded_lane_types)
        if cache is None:
            cache = {}
            for current_lane_id, current_node in self.graph.items():
                next_candidates = []
                if current_node.exit_lanes:
                    next_candidates.extend(current_node.exit_lanes)
                if current_node.left_lanes:
                    next_candidates.extend(current_node.left_lanes)
                if current_node.right_lanes:
                    next_candidates.extend(current_node.right_lanes)

                filtered_neighbors = []
                seen = set()
                for next_lane_id in next_candidates:
                    next_lane_key = self._resolve_graph_key(next_lane_id)
                    if next_lane_key is None or next_lane_key in seen:
                        continue
                    if next_lane_key not in self.graph:
                        continue
                    if self.is_lane_excluded(next_lane_key, excluded_lane_types):
                        continue
                    seen.add(next_lane_key)
                    filtered_neighbors.append(next_lane_key)
                cache[current_lane_id] = tuple(filtered_neighbors)
            self._pathfinding_neighbor_cache[excluded_lane_types] = cache
        return cache.get(lane_id, ())

    def fast_shortest_path(self, start: str, goal: str, excluded_lane_types=None):
        """
        Fast single-route search for global navigation.

        Compared with ``bfs_paths()``, this method only keeps one parent pointer
        per visited lane and reconstructs the first shortest route once the goal
        is reached.
        """
        start = self._resolve_graph_key(start)
        goal = self._resolve_graph_key(goal)
        excluded_lane_types = self._normalize_lane_type_filter(excluded_lane_types)

        if start is None or goal is None:
            return []
        if start not in self.graph or goal not in self.graph:
            return []
        if self.is_lane_excluded(start, excluded_lane_types) or self.is_lane_excluded(goal, excluded_lane_types):
            return []
        if start == goal:
            return [start]

        queue = deque([start])
        parents = {start: None}

        while queue:
            current_lane_id = queue.popleft()
            for next_lane_id in self._get_pathfinding_neighbors(current_lane_id, excluded_lane_types):
                if next_lane_id in parents:
                    continue
                parents[next_lane_id] = current_lane_id
                if next_lane_id == goal:
                    path = [goal]
                    cursor = current_lane_id
                    while cursor is not None:
                        path.append(cursor)
                        cursor = parents[cursor]
                    path.reverse()
                    return path
                queue.append(next_lane_id)

        return []
    
    def random_path(self,start:str):
        # 深拷贝图形结构
        # self.remove_dead_end_paths()
        # self.extract_cycles_from_graph()
        graph = self.no_dead_lane_graph
        # self.graph_show(graph)

        # 检查起始车道是否合法
        start = self._resolve_graph_key(start)
        if start not in graph:
            raise ValueError("lane not in graph")

        # 随机选择目标车道
        all_lanes = [lane_info.lane for lane_info in graph.values()]
        from metadrive.utils import get_np_random
        rng = get_np_random()
        goal = rng.choice(all_lanes).index

        # 初始化队列self.graph[start]
        lanes = graph[start].left_lanes + graph[start].right_lanes + [start]
        queue = [(lane, [lane]) for lane in lanes]

        # BFS 搜索路径
        while queue:
            lane, path = queue.pop(0)

            if lane not in graph:
                continue  # 在有效的图内继续搜索

            # if len(graph[lane].exit_lanes) == 0:
            #     continue  # 当前车道没有出口，继续下一条路径

            for _next in set(graph[lane].exit_lanes):
                if _next in path:
                    # 检测循环，不继续
                    continue
                if _next == goal:
                    # 找到路径，返回
                    return path + [_next]
                elif _next in graph:
                    # 将有效的下一车道添加到队列中
                    queue.append((_next, path + [_next]))

        # 当没有找到路径时返回空列表
        return []
    def find_dead_end_paths(self,graph):
        """
        找到所有通往死胡同的道路。
        存在问题，不够完善。
        """
        visited = set()
        dead_end_paths = set()

        def dfs(lane_id):
            # lane_id不在图中
            if lane_id == '53117':
                pass
            if lane_id not in graph:
                dead_end_paths.add(lane_id)  # 标记当前车道为死胡同路径
                return
                
            if lane_id in visited:
                return

            visited.add(lane_id)
            current_lane = graph[lane_id]
            
            # 如果当前车道没有出车道，它就是一个终点路径
            if len(current_lane.exit_lanes) == 0:
                dead_end_paths.add(lane_id)
                return
            
            if len(current_lane.exit_lanes) == 1:
                # 只有一个出口，继续深度搜索下去
                next_lane_id = current_lane.exit_lanes[0]
                dfs(next_lane_id)
                
                # 查看下一条车道是否也没有出口或是终点路径
                # if next_lane_id in dead_end_paths:
                #     dead_end_paths.add(lane_id)  # 标记当前车道为单向通道
            
            # 对于当前车道有多个出车道的情况，如果是单出车道则会直接返回
            for next_lane_id in current_lane.exit_lanes:
                dfs(next_lane_id)

            cur_dead_lane = [(next_lane_id in dead_end_paths) for next_lane_id in current_lane.exit_lanes ]
            if all(cur_dead_lane) == True:
                dead_end_paths.add(lane_id)


        # 从图的每一个起始点开始 DFS
        for lane_id in graph.keys():
            dfs(lane_id)

        return dead_end_paths

    def remove_dead_end_paths(self):
        if self._dead_end_computed:
            return
        dead_end_paths = self.find_dead_end_paths(self.graph)
        self.no_dead_lane_graph = copy.deepcopy(self.graph)
        for key in dead_end_paths:
            if key in self.no_dead_lane_graph:
                del self.no_dead_lane_graph[key]
        self._dead_end_computed = True
    
    def find_cycles(self,graph):
        visited = set()
        stack = []
        cycles = set()
        
        def dfs(lane_id, path):
            if lane_id in path:
                # 找到一个环
                cycle_start_index = path.index(lane_id)
                cycle = tuple(path[cycle_start_index:])  # 创建一个环的元组
                cycles.add(cycle)
                return
            
            if lane_id in visited:
                return
            
            if lane_id not in graph:
                visited.add(lane_id)
                return

            visited.add(lane_id)
            path.append(lane_id)
            
            for next_lane_id in graph[lane_id].exit_lanes:
                dfs(next_lane_id, path)

            path.pop()  # 移除当前节点，回溯

        for lane_id in graph.keys():
            dfs(lane_id, [])
        
        return cycles

    def extract_cycles_from_graph(self):
        cycles = self.find_cycles(self.graph)

        # 构建包含环的子图
        if len(self.no_dead_lane_graph) == 0:
            for cycle in cycles:
                for lane_id in cycle:
                    if lane_id not in self.no_dead_lane_graph:
                        self.no_dead_lane_graph[lane_id] = self.graph[lane_id]
        self.no_dead_lane_graph = self.graph

    # def bfs_paths(self, start: str, goal: str) -> List[List[str]]:
    #     """
    #     Breadth-first search of all routes from start to goal.

    #     :param start: starting edges
    #     :param goal: goal edge
    #     :return: list of paths from start to goal.
    #     """
    #     if self.graph[start].left_lanes is None or self.graph[start].right_lanes is None:
    #         lanes = [start]    
    #     else:
    #         lanes = self.graph[start].left_lanes + self.graph[start].right_lanes + [start]
        
    #     queue = [(lane, [lane]) for lane in lanes]
    #     # lanes = [start]
    #     if self.graph[start].left_lanes is not None:
    #         lanes.extend(self.graph[start].left_lanes)
    #     if self.graph[start].right_lanes is not None:
    #         lanes.extend(self.graph[start].right_lanes)
        
    #     queue = []
    #     for lane in lanes:
    #         if lane == start:
    #             # 原始车道：直接开始
    #             queue.append((lane, [start]))
    #         else:
    #             # 相邻车道：路径包含原始车道 -> 换道
    #             queue.append((lane, [start, lane]))
    #     while queue:
    #         (lane, path) = queue.pop(0)
    #         if lane not in self.graph:
    #             yield []
    #         if len(self.graph[lane].exit_lanes) == 0:
    #             continue
    #         for _next in set(self.graph[lane].exit_lanes):
    #             if _next in path:
    #                 # circle
    #                 continue
    #             if _next == goal:
    #                 yield path + [_next]
    #             elif _next in self.graph:
    #                 queue.append((_next, path + [_next]))

    

    def bfs_paths(self, start: str, goal: str, excluded_lane_types=None) -> Generator[List[str], None, None]:
        """
        搜索从 start 到 goal 的所有路径。
        1. 严格从 start 车道开始。
        2. 将“换道”（Left/Right）视为可能的移动步骤，允许在路径中包含换道操作。
        """
        # 1. 基础检查
        start = self._resolve_graph_key(start)
        goal = self._resolve_graph_key(goal)
        excluded_lane_types = self._normalize_lane_type_filter(excluded_lane_types)
        if start not in self.graph or goal is None:
            return
        if self.is_lane_excluded(start, excluded_lane_types) or self.is_lane_excluded(goal, excluded_lane_types):
            return

        # 2. 初始化队列：严格只包含起点
        # queue item: (当前所在车道ID, 累计路径列表)
        queue = deque()
        queue.append((start, [start]))

        # 3. 开始 BFS 循环
        while queue:
            (current_lane_id, path) = queue.popleft()

            # 鲁棒性检查
            if current_lane_id not in self.graph:
                continue

            current_node = self.graph[current_lane_id]
            
            # --- 核心修改：收集所有可能的“下一跳” ---
            next_candidates = []

            # A. 纵向连接（向前开）
            if current_node.exit_lanes:
                next_candidates.extend(current_node.exit_lanes)
            
            # B. 横向连接（换道）
            # 将左右邻居也加入候选，视作一步合法的移动
            if current_node.left_lanes:
                next_candidates.extend(current_node.left_lanes)
            if current_node.right_lanes:
                next_candidates.extend(current_node.right_lanes)


            # 4. 遍历所有候选节点
            # 使用 set 去重，防止某些数据结构里重复定义连接
            for next_lane_id in set(next_candidates):
                if self.is_lane_excluded(next_lane_id, excluded_lane_types):
                    continue
                
                # 环路检测：防止在两个车道间无限反复换道 (A->B->A...)
                if next_lane_id in path:
                    continue

                # 找到目标
                if next_lane_id == goal:
                    yield path + [next_lane_id]
                
                # 未找到目标，且该节点有效，加入队列继续搜
                elif next_lane_id in self.graph:
                    queue.append((next_lane_id, path + [next_lane_id]))
        
    def a_star_paths(self, start: str, goal: str) -> List[List[str]]:
        """
        A* search of all routes from start to goal.

        :param start: starting edge
        :param goal: goal edge
        :return: list of paths from start to goal.
        """
        start = self._resolve_graph_key(start)
        goal = self._resolve_graph_key(goal)
        from typing import List, Dict, Tuple, Any
        import heapq
        # 初始化优先队列和路径
        open_set = []
        heapq.heappush(open_set, (0, start, [start]))  # (f_score, current_node, path)

        # g_score 初始化
        g_score = {start: 0}
        
        while open_set:
            f_score, lane, path = heapq.heappop(open_set)

            # 如果达到目标节点，返回当前路径
            if lane == goal:
                yield path

            # 如果节点不在图中，跳过
            if lane not in self.graph:
                continue

            # 遍历当前节点的出口车道
            for _next in set(self.graph[lane].exit_lanes):
                if _next in path:
                    # 检查是否形成环
                    continue

                # 计算 g_score
                tentative_g_score = g_score[lane] + 1  # 假设每条边的权重为 1
                if _next not in g_score or tentative_g_score < g_score[_next]:
                    # 更新 g_score 和 f_score
                    g_score[_next] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(_next, goal)  # 计算 f_score

                    # 将下一个节点和当前路径加入优先队列
                    heapq.heappush(open_set, (f_score, _next, path + [_next]))

    def heuristic(self, node: str, goal: str) -> float:
        # 定义启发式函数，假设使用简单的距离或其他合适的度量
        # 这里可以根据具体情况调整
        return 0  # 需要根据实际情况修改

    def get_peer_lanes_from_index(self, lane_index, excluded_lane_types=None):
        """
        从给定的车道索引获取相邻的车道信息
        return:[中 左 右]
        """
        excluded_lane_types = self._normalize_lane_type_filter(excluded_lane_types)
        lane_index = self._resolve_graph_key(lane_index)
        info = self.graph[lane_index]

        def lane_from_neighbor(n):
            # n 可能是 dict(含 'feature_id')、对象(含 .feature_id)，或是 lane 索引
            if isinstance(n, dict) and 'feature_id' in n:
                fid = n['feature_id']
            elif hasattr(n, 'feature_id'):
                fid = n.feature_id
            else:
                # 直接认为是索引
                return self.graph[self._resolve_graph_key(n)].lane

            # feature_id 可能是 lane 对象或索引
            return fid.lane if hasattr(fid, 'lane') else self.graph[self._resolve_graph_key(fid)].lane

        ret = []
        if not self.is_lane_excluded(info.lane, excluded_lane_types):
            ret.append(info.lane)
        for n in (info.left_lanes or []):
            lane = lane_from_neighbor(n)
            if not self.is_lane_excluded(lane, excluded_lane_types):
                ret.append(lane)
        for n in (info.right_lanes or []):
            lane = lane_from_neighbor(n)
            if not self.is_lane_excluded(lane, excluded_lane_types):
                ret.append(lane)
        return ret

    # def get_peer_lanes_from_index(self, lane_index):
    #     """
    #     从给定的车道索引获取相邻的所有车道信息（支持无限延伸）
    #     return: [当前, 左1, 左2..., 右1, 右2...]
    #     """
        
    #     # --- 1. 提取一个辅助函数，用于解析 ID 和 Lane 对象 ---
    #     def parse_neighbor_info(n):
    #         # 返回元组: (lane_id, lane_object)
    #         fid = None
    #         if isinstance(n, dict) and 'feature_id' in n:
    #             fid = n['feature_id']
    #         elif hasattr(n, 'feature_id'):
    #             fid = n.feature_id
    #         else:
    #             fid = n # 直接认为是索引
            
    #         # 获取 lane 对象
    #         # 如果 fid 本身就是对象且有 lane 属性
    #         if hasattr(fid, 'lane'):
    #             return fid.id, fid.lane # 假设它有 id 属性，视具体对象结构而定
    #         else:
    #             # 否则去 graph 里查
    #             return fid, self.graph[fid].lane

    #     # 获取当前车道信息
    #     center_info = self.graph[lane_index]
    #     center_lane = center_info.lane
        
    #     # --- 2. 向左链式搜索 (Iterative Search Left) ---
    #     left_lanes = []
    #     current_node_info = center_info
        
    #     # 防止死循环（比如环岛或数据错误导致 A左是B，B左是A）
    #     visited = {lane_index} 
        
    #     while True:
    #         # 如果没有左车道，停止
    #         if not current_node_info.left_lanes or len(current_node_info.left_lanes) == 0:
    #             break
            
    #         # 通常取第一个作为紧邻的左侧（假设数据结构是这样定义的）
    #         # 如果 left_lanes 列表里有多个，说明遇到了分叉，这里简单策略取第一个
    #         neighbor_data = current_node_info.left_lanes[0]
            
    #         next_id, next_lane_obj = parse_neighbor_info(neighbor_data)
            
    #         if next_id in visited:
    #             break
            
    #         left_lanes.append(next_lane_obj)
    #         visited.add(next_id)
            
    #         # 【关键一步】：将当前节点切换为刚才找到的左车道，继续向左找
    #         current_node_info = self.graph[next_id]

    #     # --- 3. 向右链式搜索 (Iterative Search Right) ---
    #     right_lanes = []
    #     current_node_info = center_info # 重置回中间
    #     # visited 不需要重置，防止左右互联形成闭环
        
    #     while True:
    #         if not current_node_info.right_lanes or len(current_node_info.right_lanes) == 0:
    #             break
                
    #         neighbor_data = current_node_info.right_lanes[0]
    #         next_id, next_lane_obj = parse_neighbor_info(neighbor_data)
            
    #         if next_id in visited:
    #             break
                
    #         right_lanes.append(next_lane_obj)
    #         visited.add(next_id)
            
    #         # 【关键一步】：切换到右车道，继续向右找
    #         current_node_info = self.graph[next_id]

    #     # --- 4. 组装返回 ---
    #     # [当前, 左1, 左2..., 右1, 右2...]
    #     ret = [center_lane]
    #     ret.extend(left_lanes)
    #     ret.extend(right_lanes)
        
    #     return ret

    def destroy(self):
        """
        Destroy all lanes in this road network
        Returns: None

        """
        super(EdgeRoadNetworkWithPlanning, self).destroy()
        self._pathfinding_neighbor_cache = None
        if self.graph is not None:
            for k, v in self.graph.items():
                v.lane.destroy()
                self.graph[k]: lane_info = None
            self.graph = None

    def __del__(self):
        logging.debug("{} is released".format(self.__class__.__name__))

    def get_map_features(self, interval=2):

        ret = {}
        for id, lane_info in self.graph.items():
            assert id == lane_info.lane.index
            ret[id] = {
                SD.POLYLINE: lane_info.lane.get_polyline(interval),
                SD.POLYGON: lane_info.lane.polygon,
                SD.TYPE: lane_info.lane.metadrive_type,
                SD.ENTRY: lane_info.entry_lanes,
                SD.EXIT: lane_info.exit_lanes,
                SD.LEFT_NEIGHBORS: lane_info.left_lanes,
                SD.RIGHT_NEIGHBORS: lane_info.right_lanes,
                "speed_limit_kmh": lane_info.lane.speed_limit
            }
        return ret

    def get_all_lanes(self):
        """
        This function will return all lanes in the road network
        :return: list of lanes
        """
        ret = []
        for id, lane_info in self.graph.items():
            ret.append(lane_info.lane)
        return ret
    

    # def get_closest_lane_index(self, position, heading, ego_vehicle,use_heading_filter=False,return_all=False,draw_polygon=False):
    #     '''
    #     判断给定位置是否在车道内
    #     '''
    #     if len(position) == 3:# 确保position是2D坐标
    #         position = position[:2]
    #     elif len(position) != 2:
    #         raise ValueError(f"Position should be 2D or 3D, got {len(position)}D")
        
    #     from matplotlib.path import Path
    #     import numpy as np
    #     possible_lanes = []

    #     for id, lane_info in self.graph.items():
    #         try:
    #             polygon_array = np.array(lane_info.lane.polygon)
    #             if len(polygon_array) >= 3:  # 至少需要3个点形成多边形
    #                 polygon_path = Path(polygon_array)
    #                 if polygon_path.contains_point(position):
    #                     possible_lanes.append(id)
    #         except Exception as e:
    #             pass
    #     if ego_vehicle.id == '7':
    #         pass
    #     if draw_polygon:
    #         self.debug_plot_all_lane_polygons_and_heading(
    #                 position,
    #                 heading,
    #                 10.0,
    #                 True,
    #                 None
    #                 )
    #     if len(possible_lanes) == 0:
    #         return None, False
    #         # raise ValueError(f"No lane found for position {position}")

    #     if return_all:
    #         return possible_lanes,True
    #     else:
    #         from metadrive.utils import get_np_random
    #         rng = get_np_random()
    #         index = rng.choice(possible_lanes)

    #         return index,True
    def get_closest_lane_index(
        self,
        position,
        heading,
        ego_vehicle,
        use_heading_filter=True,
        return_all=False,
        draw_polygon=False,
        excluded_lane_types=None
    ):
        '''
        判断给定位置是否在车道内，并可选地利用航向角进行筛选
        '''
        import numpy as np
        import math
        from matplotlib.path import Path

        if len(position) == 3:  # 确保position是2D坐标
            position = position[:2]
        elif len(position) != 2:
            raise ValueError(f"Position should be 2D or 3D, got {len(position)}D")

        possible_lanes = []
        excluded_lane_types = self._normalize_lane_type_filter(excluded_lane_types)

        # 1. 第一步：几何筛选（Polygon Contains）
        for id, lane_info in self.graph.items():
            try:
                if self.is_lane_excluded(lane_info.lane, excluded_lane_types):
                    continue
                polygon_array = np.array(lane_info.lane.polygon)
                if len(polygon_array) >= 3:
                    polygon_path = Path(polygon_array)
                    if polygon_path.contains_point(position):
                        possible_lanes.append(id)
            except Exception as e:
                pass
        
        # 调试绘图逻辑
        if ego_vehicle is not None and  ego_vehicle.id == '7':
            pass
        if draw_polygon:
            self.debug_plot_all_lane_polygons_and_heading(
                position,
                heading,
                10.0,
                True,
                None
            )

        # 2. 如果没有找到车道，直接返回
        if len(possible_lanes) == 0:
            return None, False

        # 3. 如果不需要返回所有，且开启了角度过滤，且确实存在多条备选车道
        if not return_all and use_heading_filter and len(possible_lanes) > 1:
            best_lane_id = None
            min_angle_diff = float('inf')

            for lane_id in possible_lanes:
                lane = self.graph[lane_id].lane
                
                # 获取该车道在当前位置的切线方向（Heading）
                # 先将全局坐标投影到车道的 Frenet 坐标 (s, d)
                long_s, _ = lane.local_coordinates(position)
                # 获取该 s 处的航向角 (radians)
                lane_heading = lane.heading_theta_at(long_s)

                # 计算角度差 (处理 -pi 到 pi 的循环)
                # 公式: 差值 = (目标角 - 当前角 + pi) % 2pi - pi
                diff = (lane_heading - heading + np.pi) % (2 * np.pi) - np.pi
                abs_diff = abs(diff)

                # 记录最小偏差的车道
                if abs_diff < min_angle_diff:
                    min_angle_diff = abs_diff
                    best_lane_id = lane_id
            
            # 如果找到了角度最匹配的车道，直接返回它
            if best_lane_id is not None:
                return best_lane_id, True

        # 4. 常规返回逻辑
        if return_all:
            return possible_lanes, True
        else:
            # 如果没开 heading filter 或者 只有一条车道，或者角度计算失败，随机取一条
            # (通常只有一条时，这就是那一条)
            from metadrive.utils import get_np_random
            rng = get_np_random()
            index = rng.choice(possible_lanes)
            return index, True

    # ==================== 新增: 可视化 polygon_array + position + heading ====================
    def debug_plot_lane_polygon_and_heading(
        self,
        lane_id,
        position,
        heading,
        arrow_len: float = 10.0,
        show: bool = True,
        save_path: str = None
    ):
        """
        调试用：画出指定车道的 polygon、多边形边界，给定的 position 点，以及 heading 向量。

        参数:
            lane_id: 车道索引, 必须在 self.graph 中
            position: [x, y] 或 [x, y, z]
            heading: 弧度制朝向, 与 np.arctan2 一致
            arrow_len: heading 箭头长度
            show: 是否调用 plt.show()
            save_path: 若不为 None, 保存到该路径 (如 'debug_lane.png')
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import math

        if lane_id not in self.graph:
            raise ValueError(f"lane_id {lane_id} not in road network")

        if len(position) == 3:
            position = position[:2]
        elif len(position) != 2:
            raise ValueError(f"Position should be 2D or 3D, got {len(position)}D")

        pos = np.asarray(position, dtype=float)
        lane = self.graph[lane_id].lane

        polygon_array = np.asarray(lane.polygon, dtype=float)
        if polygon_array.shape[0] < 3:
            raise ValueError(f"lane {lane_id} polygon has < 3 points")

        fig, ax = plt.subplots()

        # 画 polygon
        closed_poly = np.vstack([polygon_array[:, :2], polygon_array[0, :2]])
        ax.plot(closed_poly[:, 0], closed_poly[:, 1], '-b', label='lane polygon')

        # 画 position 点
        ax.scatter([pos[0]], [pos[1]], c='r', s=50, label='position')

        # 画 heading 箭头
        hx = pos[0] + arrow_len * math.cos(heading)
        hy = pos[1] + arrow_len * math.sin(heading)
        ax.arrow(
            pos[0], pos[1],
            hx - pos[0], hy - pos[1],
            head_width=1.0, head_length=2.0,
            fc='r', ec='r', label='heading'
        )

        ax.set_aspect('equal', 'box')
        ax.legend()
        ax.set_title(f"lane_id={lane_id}, pos=({pos[0]:.2f},{pos[1]:.2f}), heading={heading:.3f} rad")

        if save_path is not None:
            plt.savefig(save_path, dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)

    # ==================== 新增: 画出所有车道 polygon + 一个位置与方向 ====================
    def debug_plot_all_lane_polygons_and_heading(
        self,
        position,
        heading,
        arrow_len: float = 10.0,
        show: bool = True,
        save_path: str = None
    ):
        """
        调试用：画出 self.graph 中所有车道的 polygon，多边形边界，以及给定的 position 点和 heading 向量。

        参数:
            position: [x, y] 或 [x, y, z]
            heading: 弧度制朝向, 与 np.arctan2 一致
            arrow_len: heading 箭头长度
            show: 是否调用 plt.show()
            save_path: 若不为 None, 保存到该路径 (如 'debug_all_lanes.png')
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import math

        if len(position) == 3:
            position = position[:2]
        elif len(position) != 2:
            raise ValueError(f"Position should be 2D or 3D, got {len(position)}D")

        pos = np.asarray(position, dtype=float)

        fig, ax = plt.subplots()

        # 画所有 lane 的 polygon
        for lane_id, lane_info in self.graph.items():
            try:
                polygon_array = np.asarray(lane_info.lane.polygon, dtype=float)
                if polygon_array.shape[0] < 3:
                    continue
                closed_poly = np.vstack([polygon_array[:, :2], polygon_array[0, :2]])
                ax.plot(closed_poly[:, 0], closed_poly[:, 1], '-b', alpha=0.4)
            except Exception:
                # 某些 lane 可能没有合法 polygon，直接跳过
                continue

        # 画 position 点
        ax.scatter([pos[0]], [pos[1]], c='r', s=60, label='position')

        # 画 heading 箭头
        hx = pos[0] + arrow_len * math.cos(heading)
        hy = pos[1] + arrow_len * math.sin(heading)
        ax.arrow(
            pos[0], pos[1],
            hx - pos[0], hy - pos[1],
            head_width=1.0, head_length=2.0,
            fc='r', ec='r', label='heading'
        )

        ax.set_aspect('equal', 'box')
        ax.legend()
        ax.set_title(f"All lane polygons, pos=({pos[0]:.2f},{pos[1]:.2f}), heading={heading:.3f} rad")

        if save_path is not None:
            plt.savefig(save_path, dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)
