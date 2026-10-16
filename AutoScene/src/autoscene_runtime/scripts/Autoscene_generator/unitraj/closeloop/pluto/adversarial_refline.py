import numpy as np


def _target_indices(pluto_feature, target_agent_ids):
    tokens = [str(x) for x in pluto_feature.get("agent_tokens", [])]
    if not tokens:
        return []
    out = []
    for target_id in target_agent_ids or []:
        sid = str(target_id)
        if sid in tokens:
            out.append(tokens.index(sid))
            continue
        suffix = sid[-5:]
        matches = [i for i, token in enumerate(tokens) if token.endswith(suffix)]
        if len(matches) == 1:
            out.append(matches[0])
    return out


def _wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _hermite_line_to_target(target_pos, target_heading, n_points, target_lookahead=24.0):
    p0 = np.zeros(2, dtype=np.float64)
    d0 = np.array([1.0, 0.0], dtype=np.float64)
    d1 = np.array([np.cos(target_heading), np.sin(target_heading)], dtype=np.float64)

    target_pos = np.asarray(target_pos, dtype=np.float64)[:2]
    p1 = target_pos + float(target_lookahead) * d1
    chord = float(np.linalg.norm(p1 - p0))
    tangent_scale = max(8.0, min(35.0, chord * 0.65))
    m0 = tangent_scale * d0
    m1 = tangent_scale * d1

    s = np.linspace(0.0, 1.0, int(n_points) + 1, dtype=np.float64)
    h00 = 2.0 * s ** 3 - 3.0 * s ** 2 + 1.0
    h10 = s ** 3 - 2.0 * s ** 2 + s
    h01 = -2.0 * s ** 3 + 3.0 * s ** 2
    h11 = s ** 3 - s ** 2
    pts = (
        h00[:, None] * p0[None]
        + h10[:, None] * m0[None]
        + h01[:, None] * p1[None]
        + h11[:, None] * m1[None]
    )
    vec = np.diff(pts, axis=0)
    pos = pts[:-1]
    ori = np.arctan2(vec[:, 1], vec[:, 0])
    return pos.astype(np.float32), vec.astype(np.float32), ori.astype(np.float32)


def _target_refline_heading(target_pos, target_heading, opposite_heading_threshold=2.2):
    """Choose terminal tangent for AV2 target refline in AV2 local coordinates."""
    target_pos = np.asarray(target_pos, dtype=np.float64)[:2]
    target_heading = _wrap_to_pi(float(target_heading))
    is_opposite = abs(target_heading) >= float(opposite_heading_threshold)

    if is_opposite:
        # For oncoming targets, matching the target lane heading would create a U-turn.
        # Keep AV2's forward tangent and let the line pass toward the target position.
        return 0.0, True

    return target_heading, False


def _target_forward_goal(agent, idx, hist_idx, forward_length=18.0, fallback_extend=10.0):
    """Pick a target point ahead on the target agent future trajectory."""
    pos_all = np.asarray(agent["position"][idx], dtype=np.float64)
    valid_all = np.asarray(agent["valid_mask"][idx]).astype(bool)
    heading_all = np.asarray(agent["heading"][idx], dtype=np.float64)

    current_pos = pos_all[hist_idx, :2]
    current_heading = float(heading_all[hist_idx])
    if not np.isfinite(current_pos).all() or not np.isfinite(current_heading):
        return None, None

    goal_pos = current_pos.copy()
    goal_heading = current_heading
    acc = 0.0
    last_pos = current_pos
    found = False

    for step in range(hist_idx + 1, len(pos_all)):
        if step >= len(valid_all) or not bool(valid_all[step]):
            break
        next_pos = pos_all[step, :2]
        if not np.isfinite(next_pos).all():
            break
        seg = float(np.linalg.norm(next_pos - last_pos))
        if seg < 1e-4:
            last_pos = next_pos
            continue
        if acc + seg >= float(forward_length):
            ratio = (float(forward_length) - acc) / max(seg, 1e-6)
            goal_pos = last_pos + ratio * (next_pos - last_pos)
            goal_heading = float(np.arctan2(next_pos[1] - last_pos[1], next_pos[0] - last_pos[0]))
            found = True
            break
        acc += seg
        goal_pos = next_pos
        goal_heading = float(heading_all[step]) if np.isfinite(heading_all[step]) else float(
            np.arctan2(next_pos[1] - last_pos[1], next_pos[0] - last_pos[0])
        )
        last_pos = next_pos

    if not found:
        remain = max(0.0, float(forward_length) - acc)
        extend = max(float(fallback_extend), remain)
        direction = np.array([np.cos(goal_heading), np.sin(goal_heading)], dtype=np.float64)
        goal_pos = goal_pos + extend * direction

    return goal_pos, _wrap_to_pi(goal_heading)


def append_target_reference_lines(
        pluto_feature,
        target_agent_ids,
        hist_steps=21,
        max_target_lines=2,
        target_lookahead=24.0,
        target_forward_length=18.0,
        fallback_extend=10.0,
        opposite_heading_threshold=2.2,
        keep_original_reference_lines=True,
):
    """Prepend smooth AV2->target synthetic reference lines to a Pluto feature dict.

    The feature is already in the controlled-agent local frame, so AV2 is at the origin
    with heading 0. The synthetic line starts from AV2 and smoothly bends toward the
    target agent's future forward point. For same-direction targets it continues along the
    target heading; for oncoming targets it keeps AV2's forward heading to avoid
    inducing a U-turn.
    """
    ref = pluto_feature.get("reference_line", None)
    agent = pluto_feature.get("agent", None)
    if ref is None or agent is None:
        return pluto_feature
    if "position" not in ref or "vector" not in ref or "orientation" not in ref:
        return pluto_feature

    n_points = int(ref["position"].shape[1])
    if n_points <= 1:
        return pluto_feature

    target_lines = []
    for idx in _target_indices(pluto_feature, target_agent_ids)[:max_target_lines]:
        if idx == 1:
            continue
        valid = agent["valid_mask"][idx]
        if len(valid) <= hist_steps - 1 or not bool(valid[hist_steps - 1]):
            continue
        target_pos, target_heading = _target_forward_goal(
            agent,
            idx,
            hist_steps - 1,
            forward_length=target_forward_length,
            fallback_extend=fallback_extend,
        )
        if target_pos is None or target_heading is None:
            continue
        if not np.isfinite(target_pos).all() or not np.isfinite(target_heading):
            continue
        if float(np.linalg.norm(target_pos)) < 1.0:
            continue
        refline_heading, _ = _target_refline_heading(
            target_pos,
            target_heading,
            opposite_heading_threshold=opposite_heading_threshold,
        )
        pos, vec, ori = _hermite_line_to_target(
            target_pos,
            refline_heading,
            n_points,
            target_lookahead=target_lookahead,
        )
        target_lines.append((pos, vec, ori))

    if not target_lines:
        return pluto_feature

    k = len(target_lines)
    pos = np.stack([x[0] for x in target_lines], axis=0).astype(ref["position"].dtype, copy=False)
    vec = np.stack([x[1] for x in target_lines], axis=0).astype(ref["vector"].dtype, copy=False)
    ori = np.stack([x[2] for x in target_lines], axis=0).astype(ref["orientation"].dtype, copy=False)
    valid = np.ones((k, n_points), dtype=ref["valid_mask"].dtype)
    route_id = -1000 - np.arange(k, dtype=ref.get("route_id", np.zeros((0,), dtype=np.int64)).dtype)

    if keep_original_reference_lines:
        ref["position"] = np.concatenate([pos, ref["position"]], axis=0)
        ref["vector"] = np.concatenate([vec, ref["vector"]], axis=0)
        ref["orientation"] = np.concatenate([ori, ref["orientation"]], axis=0)
        ref["valid_mask"] = np.concatenate([valid, ref["valid_mask"]], axis=0)
    else:
        ref["position"] = pos
        ref["vector"] = vec
        ref["orientation"] = ori
        ref["valid_mask"] = valid

    if "future_projection" in ref:
        future_projection = np.zeros(
            (k,) + tuple(ref["future_projection"].shape[1:]),
            dtype=ref["future_projection"].dtype,
        )
        if keep_original_reference_lines:
            ref["future_projection"] = np.concatenate([future_projection, ref["future_projection"]], axis=0)
        else:
            ref["future_projection"] = future_projection
    if "route_id" in ref:
        if keep_original_reference_lines:
            ref["route_id"] = np.concatenate([route_id, ref["route_id"]], axis=0)
        else:
            ref["route_id"] = route_id

    return pluto_feature
