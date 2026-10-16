import numpy as np
from typing import Optional, Union


class HeadTopDownRenderer:
    """
    This class is used to render the top-down view of the environment.
    It is a pseudo-render function, only used to update onscreen message when using panda3d backend.
    """

    def __init__(self, env):
        self.engine = env.engine
        self.head_top_down_renderer = None
        self.reference_lines = None
        self.cbv_reference_lines = []  # list of (ref_lines, color)
        self.av2_reference_lines = None

    def set_reference_lines(self, ref_lines):
        """Set ego reference lines to be drawn on the next render call."""
        self.reference_lines = ref_lines

    def add_cbv_reference_lines(self, ref_lines, color=None):
        """Add a CBV's reference lines with a given color."""
        if ref_lines is not None:
            self.cbv_reference_lines.append((ref_lines, color))

    def set_av2_reference_lines(self, ref_lines):
        """Set the nuPlan/AV2 target reference-line overlay."""
        self.av2_reference_lines = ref_lines

    def clear_overlays(self):
        """Clear per-episode planning overlays."""
        self.reference_lines = None
        self.cbv_reference_lines = []
        self.av2_reference_lines = None
        if self.head_top_down_renderer is not None:
            self.head_top_down_renderer.cbv_id = []
            self.head_top_down_renderer.max_score_cbv_id = []

    def _get_nearest_centerline_tangent(self, ego_xy):
        if self.reference_lines is None:
            return None

        best = None
        best_dist = float("inf")

        if isinstance(self.reference_lines, dict):
            positions = self.reference_lines.get("position", [])
            orientations = self.reference_lines.get("orientation", None)
            valid_masks = self.reference_lines.get("valid_mask", None)
            num_lines = len(positions)

            for i in range(num_lines):
                pos = np.asarray(positions[i])
                if pos.size == 0:
                    continue
                pos_xy = pos[..., :2]

                mask = None
                if valid_masks is not None and len(valid_masks) > i:
                    mask = np.asarray(valid_masks[i]).astype(bool)
                    if mask.shape[0] == pos_xy.shape[0]:
                        pos_xy = pos_xy[mask]
                    else:
                        mask = None

                if pos_xy.shape[0] == 0:
                    continue

                dists = np.sum((pos_xy - ego_xy) ** 2, axis=1)
                idx = int(np.argmin(dists))

                heading = None
                if orientations is not None and len(orientations) > i:
                    ori = np.asarray(orientations[i])
                    if mask is not None and ori.shape[0] == mask.shape[0]:
                        ori = ori[mask]
                    if ori.shape[0] > idx:
                        heading = float(ori[idx])

                if heading is None:
                    prev_idx = max(idx - 1, 0)
                    next_idx = min(idx + 1, pos_xy.shape[0] - 1)
                    vec = pos_xy[next_idx] - pos_xy[prev_idx]
                    if np.linalg.norm(vec) < 1e-6:
                        heading = 0.0
                    else:
                        heading = float(np.arctan2(vec[1], vec[0]))

                if dists[idx] < best_dist:
                    best_dist = dists[idx]
                    best = (pos_xy[idx], heading)
        else:
            for ref_line in self.reference_lines:
                if len(ref_line) == 0:
                    continue
                line = np.asarray(ref_line)
                pos_xy = line[:, :2]
                dists = np.sum((pos_xy - ego_xy) ** 2, axis=1)
                idx = int(np.argmin(dists))
                if line.shape[1] > 2:
                    heading = float(line[idx, 2])
                else:
                    prev_idx = max(idx - 1, 0)
                    next_idx = min(idx + 1, pos_xy.shape[0] - 1)
                    vec = pos_xy[next_idx] - pos_xy[prev_idx]
                    heading = float(np.arctan2(vec[1], vec[0])) if np.linalg.norm(vec) > 1e-6 else 0.0

                if dists[idx] < best_dist:
                    best_dist = dists[idx]
                    best = (pos_xy[idx], heading)

        return best

    def _draw_reference_lines_on_canvas(self, ref_lines=None, color=(255, 150, 255), line_width=1, stride=25):
        """Draws the reference lines on the top down canvas imitation nuplan_scenario_render."""
        ref_lines = self.reference_lines if ref_lines is None else ref_lines
        if self.head_top_down_renderer is None or ref_lines is None:
            return

        try:
            import pygame
            import math
            canvas = self.head_top_down_renderer._frame_canvas
        except Exception:
            return

        # Handle dict format (Pluto features) or list/array format (Nuplan)
        if isinstance(ref_lines, dict):
            positions = ref_lines.get("position", [])
            headings = ref_lines.get("orientation", [])
            valid_masks = ref_lines.get("valid_mask", [])

            # Use PyTorch or NumPy structure length
            num_lines = len(positions)
            for i in range(num_lines):
                num_points = len(positions[i])
                for j in range(0, num_points, stride):
                    if len(valid_masks) > i and len(valid_masks[i]) > j and not valid_masks[i][j]:
                        continue

                    x, y = positions[i][j][0], positions[i][j][1]
                    heading = headings[i][j]

                    dx = math.cos(heading) * 1.5
                    dy = math.sin(heading) * 1.5

                    start_pix = canvas.pos2pix(x, y)
                    end_pix = canvas.pos2pix(x + dx, y + dy)

                    pygame.draw.line(canvas, color, start_pix, end_pix, line_width)

                    head_width = 0.35 if line_width <= 1 else 0.5
                    perp_heading = heading + math.pi / 2
                    px1 = x + dx - math.cos(perp_heading) * head_width
                    py1 = y + dy - math.sin(perp_heading) * head_width
                    px2 = x + dx + math.cos(perp_heading) * head_width
                    py2 = y + dy + math.sin(perp_heading) * head_width

                    p1_pix = canvas.pos2pix(px1, py1)
                    p2_pix = canvas.pos2pix(px2, py2)

                    pygame.draw.polygon(canvas, color, [end_pix, p1_pix, p2_pix])

        else:
            # Original list/array format
            for ref_line in ref_lines:
                # Using the stride of 20 from nuplan
                for i in range(0, len(ref_line), stride):
                    p = ref_line[i]
                    x, y = p[0], p[1]
                    # If turning angle is represented differently, adjust it.
                    # MetaDrive generally uses math.cos(heading) and math.sin(heading)
                    heading = p[2]

                    # Nuplan vector length: 1.5
                    dx = math.cos(heading) * 1.5
                    dy = math.sin(heading) * 1.5

                    start_pix = canvas.pos2pix(x, y)
                    end_pix = canvas.pos2pix(x + dx, y + dy)

                    pygame.draw.line(canvas, color, start_pix, end_pix, line_width)

                    # Nuplan arrowhead width: 0.8 => 0.4 each side
                    head_width = 0.35 if line_width <= 1 else 0.4
                    perp_heading = heading + math.pi / 2
                    px1 = x + dx - math.cos(perp_heading) * head_width
                    py1 = y + dy - math.sin(perp_heading) * head_width
                    px2 = x + dx + math.cos(perp_heading) * head_width
                    py2 = y + dy + math.sin(perp_heading) * head_width

                    p1_pix = canvas.pos2pix(px1, py1)
                    p2_pix = canvas.pos2pix(px2, py2)

                    pygame.draw.polygon(canvas, color, [end_pix, p1_pix, p2_pix])

    def _render_topdown(self, text: Optional[Union[dict, str]] = None, *args, **kwargs) -> Optional[np.ndarray]:
        """
        Render the top-down view of the environment.
        :param text: text to show
        :return: top_down image
        """
        if self.head_top_down_renderer is None:
            from unitraj.renderer.top_down_renderer import TopDownRenderer
            self.head_top_down_renderer = TopDownRenderer(*args, **kwargs)

            # Monkey patch _draw to include our reference lines
            original_draw = self.head_top_down_renderer._draw

            def custom_draw(*a, **kw):
                original_draw(*a, **kw)

                # Draw our stuff on _frame_canvas (which was just blitted to screen, we need to draw and blit again)
                self._draw_reference_lines_on_canvas(
                    self.reference_lines,
                    color=(245, 170, 245),
                    line_width=3,
                    stride=10,
                )
                for ref_lines, color in self.cbv_reference_lines:
                    self._draw_reference_lines_on_canvas(
                        ref_lines,
                        color=color or (0, 200, 255),
                        line_width=3,
                        stride=10,
                    )
                if self.av2_reference_lines is not None:
                    self._draw_reference_lines_on_canvas(
                        self.av2_reference_lines,
                        color=(0, 200, 255),
                        line_width=4,
                        stride=10,
                    )

                # RE-BLIT to _screen_canvas so our drawings are visible!
                v = self.head_top_down_renderer.current_track_agent
                canvas = self.head_top_down_renderer._frame_canvas
                screen_canvas = self.head_top_down_renderer._screen_canvas
                field = screen_canvas.get_size()

                if not self.head_top_down_renderer.target_agent_heading_up:
                    if self.head_top_down_renderer.position is not None or v is not None:
                        cam_pos = (self.head_top_down_renderer.position or v.position)
                        position = canvas.pos2pix(*cam_pos)
                    else:
                        position = (field[0] / 2, field[1] / 2)
                    off = (position[0] - field[0] / 2, position[1] - field[1] / 2)
                    screen_canvas.blit(source=canvas, dest=(0, 0), area=(off[0], off[1], field[0], field[1]))
                else:
                    color_white = (255, 255, 255)
                    position = canvas.pos2pix(*v.position)
                    area = (
                        position[0] - self.head_top_down_renderer.canvas_rotate.get_size()[0] / 2,
                        position[1] - self.head_top_down_renderer.canvas_rotate.get_size()[1] / 2,
                        self.head_top_down_renderer.canvas_rotate.get_size()[0],
                        self.head_top_down_renderer.canvas_rotate.get_size()[1]
                    )
                    self.head_top_down_renderer.canvas_rotate.fill(color_white)
                    self.head_top_down_renderer.canvas_rotate.blit(source=canvas, dest=(0, 0), area=area)

                    import pygame
                    rotation = -np.rad2deg(v.heading_theta) + 90
                    new_canvas = pygame.transform.rotozoom(self.head_top_down_renderer.canvas_rotate, rotation, 1)

                    size = screen_canvas.get_size()
                    screen_canvas.blit(
                        new_canvas,
                        (0, 0),
                        (
                            new_canvas.get_size()[0] / 2 - size[0] / 2,
                            new_canvas.get_size()[1] / 2 - size[1] / 2,
                            size[0],
                            size[1]
                        )
                    )

            self.head_top_down_renderer._draw = custom_draw

        return self.head_top_down_renderer.render(text, *args, **kwargs)

    def render(self, text: Optional[Union[dict, str]] = None, mode=None, *args, **kwargs) -> Optional[np.ndarray]:
        """
        This is a pseudo-render function, only used to update onscreen message when using panda3d backend
        :param text: text to show
        :param mode: start_top_down rendering candidate parameter is ["top_down", "topdown", "bev", "birdview"]
        :return: None or top_down image
        """

        if mode in ["top_down", "topdown", "bev", "birdview"]:
            ret = self._render_topdown(text=text, *args, **kwargs)
            return ret
        return None

    def reset(self):
        """
        Reset the renderer.
        This is a placeholder function for compatibility.
        """
        self.clear_overlays()
        if self.head_top_down_renderer is not None:
            self.head_top_down_renderer.close()
            self.head_top_down_renderer = None