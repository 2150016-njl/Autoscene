import os
import torch
import numpy as np
from scipy.special import softmax
from unitraj.datasets.Pluto_dataset.Pluto_test_dataset import PlutoTestDataset
from unitraj.models.pluto.pluto_model import PlanningModel
from unitraj.rl_finetuning.cbv_adversarial_reward import CBVAdversarialReward
from unitraj.closeloop.pluto.fast_trajectory_evaluator import FastTrajectoryEvaluator
from unitraj.closeloop.pluto.trajectory_evaluator import TrajectoryEvaluator

def safe_cuda_available():
    try:
        return bool(torch.cuda.is_available())
    except Exception as exc:
        print(f"CUDA availability check failed; PlutoInference will use CPU: {exc}")
        return False

class PlutoInference:
    def __init__(self, cfg):
        self.device = "cuda" if safe_cuda_available() else "cpu"
        self.cfg = cfg
        self.dataset = PlutoTestDataset(cfg, is_validation=True)
        self._osm_brake_hold_until = {}
        self._adversarial_cutin_lock_until = {}
        self._ego_lane_cutin_lock_until = {}

        # Optional MLP scorer (replaces rule-based evaluator)
        self.score_net = None
        if str(cfg.get("evaluator_type", "rule")).lower() == "mlp":
            ckpt = cfg.get("score_net_ckpt", None)
            if ckpt and os.path.isfile(ckpt):
                from unitraj.rl_finetuning.score_net.infer import ScoreNetInfer
                self.score_net = ScoreNetInfer(ckpt, device=self.device)
                print(f"[PlutoInference] MLP scorer loaded from {ckpt}")

    def _adv_refline_debug_enabled(self):
        return os.environ.get("UNITRAJ_ADV_REF_LINE_DEBUG", "0") in ("1", "true", "True")

    def _brake_debug_enabled(self):
        return os.environ.get("UNITRAJ_BRAKE_DEBUG", "0") not in ("0", "false", "False")

    def _brake_agent_key(self, controlled_agent_id):
        return "ego" if controlled_agent_id is None else str(controlled_agent_id)

    def _agent_key(self, controlled_agent_id):
        return "ego" if controlled_agent_id is None else str(controlled_agent_id)

    def _tjwest_ref_audit_enabled(self):
        return (
            os.environ.get("AUTOSCENE_DATASET_TYPE", "tjwest").strip().lower()
            == "tjwest"
            and os.environ.get(
                "UNITRAJ_TJWEST_REF_AUDIT",
                "1",
            ) not in ("0", "false", "False")
        )

    def _selected_reference_geometry_audit(self, ref_lines, selected_ref_idx):
        gap_threshold = float(
            self.cfg.get("tjwest_ref_audit_point_gap_m", 2.5)
        )
        heading_jump_threshold = float(
            self.cfg.get("tjwest_ref_audit_heading_jump_rad", 1.0)
        )
        result = {
            "selected_ref_idx": selected_ref_idx,
            "selected_ref_route_id": None,
            "valid_point_count": 0,
            "valid_mask_hole_count": 0,
            "max_point_gap_m": None,
            "point_gap_count": 0,
            "max_heading_jump_rad": None,
            "heading_jump_count": 0,
            "geometry_is_continuous": False,
        }
        if (
            not isinstance(ref_lines, dict)
            or selected_ref_idx is None
            or int(selected_ref_idx) < 0
        ):
            return result

        index = int(selected_ref_idx)
        route_ids = np.asarray(ref_lines.get("route_id", []))
        if index < len(route_ids):
            try:
                result["selected_ref_route_id"] = int(route_ids[index])
            except (TypeError, ValueError):
                result["selected_ref_route_id"] = None

        positions = np.asarray(ref_lines.get("position", []), dtype=np.float64)
        if positions.ndim != 3 or index >= positions.shape[0]:
            return result

        mask_raw = np.asarray(ref_lines.get("valid_mask", []))
        if mask_raw.ndim >= 2 and index < mask_raw.shape[0]:
            mask = np.asarray(mask_raw[index], dtype=bool).reshape(-1)
            mask = mask[:positions.shape[1]]
        else:
            mask = np.ones((positions.shape[1],), dtype=bool)
        finite = np.isfinite(positions[index, :, :2]).all(axis=1)
        mask = mask & finite
        valid_indices = np.flatnonzero(mask)
        result["valid_point_count"] = int(len(valid_indices))
        if len(valid_indices) < 2:
            return result

        result["valid_mask_hole_count"] = int(
            np.sum(np.diff(valid_indices) > 1)
        )
        xy = positions[index, valid_indices, :2]
        segments = np.diff(xy, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        if len(segment_lengths) > 0:
            result["max_point_gap_m"] = float(np.max(segment_lengths))
            result["point_gap_count"] = int(
                np.sum(segment_lengths > gap_threshold)
            )

        usable = segment_lengths > 1e-6
        headings = np.arctan2(
            segments[usable, 1],
            segments[usable, 0],
        )
        if len(headings) >= 2:
            heading_delta = np.abs(
                (np.diff(headings) + np.pi) % (2.0 * np.pi) - np.pi
            )
            result["max_heading_jump_rad"] = float(np.max(heading_delta))
            result["heading_jump_count"] = int(
                np.sum(heading_delta > heading_jump_threshold)
            )
        elif len(headings) == 1:
            result["max_heading_jump_rad"] = 0.0

        result["geometry_is_continuous"] = bool(
            result["valid_mask_hole_count"] == 0
            and result["point_gap_count"] == 0
            and result["heading_jump_count"] == 0
        )
        result["point_gap_threshold_m"] = gap_threshold
        result["heading_jump_threshold_rad"] = heading_jump_threshold
        result["_selected_xy"] = xy
        return result

    def _emit_tjwest_reference_audit(
            self,
            timestep,
            controlled_agent_id,
            ref_lines,
            selected_ref_idx,
    ):
        geometry = self._selected_reference_geometry_audit(
            ref_lines,
            selected_ref_idx,
        )
        selected_xy = geometry.pop("_selected_xy", None)

        builder = getattr(self.dataset, "_builder", None)
        builder_records = getattr(
            builder,
            "_last_refline_audit_by_agent",
            {},
        )
        if controlled_agent_id is None:
            pkl_audit = getattr(
                builder,
                "_last_refline_audit_latest",
                None,
            )
        else:
            pkl_audit = builder_records.get(str(controlled_agent_id))
        if not isinstance(pkl_audit, dict):
            pkl_audit = {}

        route_id = geometry.get("selected_ref_route_id")
        pkl_route_ids = [
            int(value)
            for value in pkl_audit.get("pkl_route_lane_ids", [])
        ]
        actor_key = self._agent_key(controlled_agent_id)
        previous_records = getattr(
            self,
            "_previous_tjwest_ref_audit",
            {},
        )
        previous = previous_records.get(actor_key, {})
        previous_route_id = previous.get("selected_ref_route_id")
        previous_xy = previous.get("_selected_xy")

        start_to_previous_line_m = None
        if (
            selected_xy is not None
            and len(selected_xy) > 0
            and previous_xy is not None
            and len(previous_xy) > 0
        ):
            start_to_previous_line_m = float(
                np.min(
                    np.linalg.norm(
                        np.asarray(previous_xy, dtype=np.float64)
                        - np.asarray(selected_xy[0], dtype=np.float64)[None, :],
                        axis=1,
                    )
                )
            )

        audit = {
            "timestep": int(timestep),
            "vehicle": actor_key,
            **geometry,
            **pkl_audit,
            "selected_ref_id_in_pkl_route": (
                route_id in pkl_route_ids
                if route_id is not None
                else False
            ),
            "selected_ref_matches_pkl_current_lane": (
                route_id is not None
                and route_id == pkl_audit.get("pkl_current_lane_id")
            ),
            "previous_selected_ref_route_id": previous_route_id,
            "selected_ref_route_changed": (
                previous_route_id is not None
                and route_id != previous_route_id
            ),
            "selected_ref_start_to_previous_line_m": start_to_previous_line_m,
        }
        previous_records[actor_key] = {
            "selected_ref_route_id": route_id,
            "_selected_xy": (
                None
                if selected_xy is None
                else np.asarray(selected_xy, dtype=np.float64).copy()
            ),
        }
        self._previous_tjwest_ref_audit = previous_records
        # print("[TJWestRefAudit]", audit)
        return audit


    def _is_target_cutin_label(self, label):
        return label is not None and str(label).startswith("target_cutin")

    def _is_ego_lane_cutin_label(self, label):
        return label is not None and (
            str(label).startswith("ego_lane_cutin")
            or str(label).startswith("ego_lane_keep")
        )

    def _select_with_cutin_lock(
        self,
        timestep,
        controlled_agent_id,
        best_idx,
        score_selection_key,
        sorted_osm_labels,
        adversarial_scores,
    ):
        if score_selection_key != "adversarial" or adversarial_scores is None:
            return best_idx, False

        agent_key = self._agent_key(controlled_agent_id)
        hold_until = int(self._adversarial_cutin_lock_until.get(agent_key, -1))
        hold_active = int(timestep) <= hold_until
        cutin_indices = np.asarray(
            [
                idx
                for idx, label in enumerate(sorted_osm_labels)
                if self._is_target_cutin_label(label)
            ],
            dtype=np.int64,
        )
        if len(cutin_indices) == 0:
            self._adversarial_cutin_lock_until.pop(agent_key, None)
            return best_idx, False

        scores = np.asarray(adversarial_scores, dtype=np.float32)
        valid_cutin_indices = cutin_indices[cutin_indices < len(scores)]
        if (
            bool(self.cfg.get("osm_adversarial_force_target_cutin", True))
            and len(valid_cutin_indices) > 0
        ):
            locked_idx = int(valid_cutin_indices[int(np.argmax(scores[valid_cutin_indices]))])
            hold_steps = int(self.cfg.get("osm_adversarial_cutin_lock_steps", 15))
            self._adversarial_cutin_lock_until[agent_key] = int(timestep) + max(0, hold_steps)
            reason = "force_target_cutin_hold" if hold_active else "force_target_cutin"
            return locked_idx, reason

        selected_label = sorted_osm_labels[best_idx] if 0 <= int(best_idx) < len(sorted_osm_labels) else None
        if hold_active:
            if len(valid_cutin_indices) > 0:
                locked_idx = int(valid_cutin_indices[int(np.argmax(scores[valid_cutin_indices]))])
                return locked_idx, "hold"

        if self._is_target_cutin_label(selected_label):
            hold_steps = int(self.cfg.get("osm_adversarial_cutin_lock_steps", 15))
            self._adversarial_cutin_lock_until[agent_key] = int(timestep) + max(0, hold_steps)
            return best_idx, "selected"

        if len(valid_cutin_indices) > 0 and 0 <= int(best_idx) < len(scores):
            margin = float(self.cfg.get("osm_adversarial_cutin_start_margin", 0.75))
            best_cutin_idx = int(valid_cutin_indices[int(np.argmax(scores[valid_cutin_indices]))])
            if margin > 0.0 and scores[best_cutin_idx] >= scores[int(best_idx)] - margin:
                hold_steps = int(self.cfg.get("osm_adversarial_cutin_lock_steps", 15))
                self._adversarial_cutin_lock_until[agent_key] = int(timestep) + max(0, hold_steps)
                return best_cutin_idx, "near_best"

        self._adversarial_cutin_lock_until.pop(agent_key, None)
        return best_idx, False

    def _select_with_ego_lane_cutin_lock(
        self,
        timestep,
        controlled_agent_id,
        best_idx,
        score_selection_key,
        sorted_osm_labels,
        final_scores,
        safe_ego_lane_cutin_indices,
    ):
        agent_id = str(self.cfg.get("av1_blocked_ego_lane_cutin_agent_id", "adv_1"))
        if (
            score_selection_key == "adversarial"
            or controlled_agent_id is None
            or str(controlled_agent_id) != agent_id
            or final_scores is None
        ):
            return best_idx, False

        agent_key = self._agent_key(controlled_agent_id)
        cutin_indices = np.asarray(safe_ego_lane_cutin_indices, dtype=np.int64)
        cutin_indices = cutin_indices[cutin_indices < len(final_scores)]
        if len(cutin_indices) == 0:
            self._ego_lane_cutin_lock_until.pop(agent_key, None)
            return best_idx, False

        scores = np.asarray(final_scores, dtype=np.float64)
        finite_cutin_indices = cutin_indices[np.isfinite(scores[cutin_indices])]
        if len(finite_cutin_indices) == 0:
            self._ego_lane_cutin_lock_until.pop(agent_key, None)
            return best_idx, False

        hold_until = int(self._ego_lane_cutin_lock_until.get(agent_key, -1))
        if int(timestep) <= hold_until:
            locked_idx = int(finite_cutin_indices[int(np.argmax(scores[finite_cutin_indices]))])
            hold_steps = int(self.cfg.get("osm_ego_lane_cutin_lock_steps", 20))
            self._ego_lane_cutin_lock_until[agent_key] = int(timestep) + max(0, hold_steps)
            return locked_idx, "hold"

        selected_label = sorted_osm_labels[best_idx] if 0 <= int(best_idx) < len(sorted_osm_labels) else None
        if self._is_ego_lane_cutin_label(selected_label):
            hold_steps = int(self.cfg.get("osm_ego_lane_cutin_lock_steps", 20))
            self._ego_lane_cutin_lock_until[agent_key] = int(timestep) + max(0, hold_steps)
            return best_idx, "selected"

        self._ego_lane_cutin_lock_until.pop(agent_key, None)
        return best_idx, False

    def _extract_predicted_agent_tokens(self, pluto_feature, pred, controlled_agent_id=None):
        agent_tokens = pluto_feature.get("agent_tokens", None)
        if agent_tokens is None:
            raise RuntimeError("Pluto feature is missing its agent-token mapping")
        tokens = [str(token) for token in agent_tokens]
        if not tokens or len(tokens) != len(set(tokens)):
            raise RuntimeError("Pluto feature has empty or duplicate agent tokens")
        if controlled_agent_id is not None and tokens[0] != str(controlled_agent_id):
            raise RuntimeError(
                "Pluto primary token %s does not match requested controlled agent %s"
                % (tokens[0], controlled_agent_id)
            )
        # Builders prepend the controlled actor. PlanningModel.forward predicts
        # x[:, 1:A], never that actor. A batched output may contain padding up to
        # another actor's neighbor count; array-length equality cannot identify
        # the token offset (it used to shift every identity by one in that case).
        background_tokens = tokens[1:]
        if pred is not None and len(pred) < len(background_tokens):
            raise RuntimeError("Pluto prediction has fewer rows than its background tokens")
        return background_tokens

    def _trim_background_prediction(self, pluto_feature, pred, controlled_agent_id=None):
        tokens = self._extract_predicted_agent_tokens(
            pluto_feature, pred, controlled_agent_id=controlled_agent_id,
        )
        return pred[:len(tokens)]

    def _extract_cbv_ego_future(self, current_state, pluto_feature, pred, controlled_agent_id=None):
        if pred is None or len(pred) == 0:
            return None
        metadata = current_state.get("metadata", {}) if isinstance(current_state, dict) else {}
        sdc_id = metadata.get("sdc_id", None)
        pred_tokens = self._extract_predicted_agent_tokens(pluto_feature, pred,
                                                           controlled_agent_id=controlled_agent_id)
        if sdc_id is not None:
            for idx, token in enumerate(pred_tokens):
                if str(token) == str(sdc_id) and idx < len(pred):
                    return np.asarray(pred[idx], dtype=np.float32)
        # Missing ego is not the first/nearest background actor. Treat the target
        # as unavailable instead of feeding another car's future to ego scoring.
        return None

    def _extract_target_future(self, pluto_feature, pred, target_agent_id, controlled_agent_id=None):
        if pred is None or len(pred) == 0 or target_agent_id is None:
            return None
        pred_tokens = self._extract_predicted_agent_tokens(
            pluto_feature,
            pred,
            controlled_agent_id=controlled_agent_id,
        )
        target_id = str(target_agent_id)
        # Complete IDs must win even if an earlier neighbor shares the suffix.
        if target_id in pred_tokens:
            return np.asarray(pred[pred_tokens.index(target_id)], dtype=np.float32)
        if len(target_id) >= 5:
            matches = [idx for idx, token in enumerate(pred_tokens)
                       if len(token) >= 5 and token[-5:] == target_id[-5:]]
            if len(matches) == 1:
                return np.asarray(pred[matches[0]], dtype=np.float32)
        return None

    def _build_adversarial_eval_candidates(self, control_candidates, planner_candidates):
        source = control_candidates if control_candidates is not None else planner_candidates
        if source is None:
            return None

        candidates = np.asarray(source, dtype=np.float32)
        if candidates.shape[-1] >= 6:
            return candidates[..., :6]
        if candidates.shape[-1] >= 4:
            vel_xy = np.gradient(candidates[..., :2], 0.1, axis=-2)
            return np.concatenate([candidates[..., :4], vel_xy[..., :2]], axis=-1).astype(np.float32)

        xy = candidates[..., :2]
        if candidates.shape[-1] >= 3:
            heading = candidates[..., 2]
        else:
            diff = np.gradient(xy, axis=-2)
            heading = np.arctan2(diff[..., 1], diff[..., 0])
        vel_xy = np.gradient(xy, 0.1, axis=-2)
        return np.concatenate(
            [
                xy,
                np.cos(heading)[..., None],
                np.sin(heading)[..., None],
                vel_xy[..., :2],
            ],
            axis=-1,
        ).astype(np.float32)

    def _filter_adversarial_reference_lines(self, ref_lines):
        if not isinstance(ref_lines, dict) or "route_id" not in ref_lines:
            if self._adv_refline_debug_enabled():
                print(
                    "[AdvRefLineDebug] no route_id in ref_lines for adversarial filter",
                    {"has_ref_lines": isinstance(ref_lines, dict), "keys": list(ref_lines.keys()) if isinstance(ref_lines, dict) else None},
                )
            return None
        route_id = np.asarray(ref_lines.get("route_id"))
        # Synthetic adversarial target reference lines are assigned ids
        # -1000, -1001, ... by adversarial_refline.py. Real map lane ids may
        # also be negative, so do not treat every negative route_id as synthetic.
        mask = (route_id <= -1000) & (route_id > -2000)
        if not mask.any():
            if self._adv_refline_debug_enabled():
                valid_mask = np.asarray(ref_lines.get("valid_mask", []))
                valid_counts = (
                    valid_mask.sum(axis=1).astype(int).tolist()
                    if valid_mask.ndim >= 2 else []
                )
                print(
                    "[AdvRefLineDebug] no synthetic adversarial reference lines",
                    {
                        "num_ref_lines": int(len(route_id)),
                        "route_ids": route_id.astype(int).tolist()[:12],
                        "valid_counts": valid_counts[:12],
                        "hint": "synthetic adversarial reference line ids are expected in [-1999, -1000]",
                    },
                )
            return None
        filtered = {}
        for key, value in ref_lines.items():
            arr = np.asarray(value)
            if arr.shape[:1] == route_id.shape[:1]:
                filtered[key] = arr[mask]
        return filtered if filtered else None

    def _score_adversarial_candidates(
            self,
            current_state,
            pluto_feature,
            input_data_single,
            pred_local,
            eval_candidates,
            candidate_ref_idx,
            controlled_agent_id,
            adversarial_target_agent_ids,
            actual_target_futures_local=None,
            feasibility_predictions_local=None,
            candidate_labels=None,
    ):
        if eval_candidates is None or len(eval_candidates) == 0:
            return None, {}

        metadata = current_state.get("metadata", {}) if isinstance(current_state, dict) else {}
        target_ids = []
        sdc_id = metadata.get("sdc_id", None)
        for target_id in adversarial_target_agent_ids or []:
            target_id = str(target_id)
            if target_id != str(controlled_agent_id) and target_id not in target_ids:
                target_ids.append(target_id)

        reward = CBVAdversarialReward(self.cfg)
        total = np.zeros((len(eval_candidates),), dtype=np.float32)
        component_keys = (
            "distance_risk",
            "ttc_reward",
            "interaction",
            "approach",
            "feasibility",
            "behavior_disruption",
        )
        component_sums = {
            key: np.zeros((len(eval_candidates),), dtype=np.float32)
            for key in component_keys
        }
        per_target = {}
        used_targets = 0
        for target_id in target_ids:
            target_future = None
            target_keys = [str(target_id)]
            if str(target_id) == "ego" or (sdc_id is not None and str(target_id) == str(sdc_id)):
                target_keys.append("ego")
            for key in target_keys:
                if actual_target_futures_local and key in actual_target_futures_local:
                    target_future = actual_target_futures_local[key]
                    break
            if target_future is None:
                target_future = self._extract_target_future(
                    pluto_feature,
                    pred_local,
                    target_id,
                    controlled_agent_id=controlled_agent_id,
                )
            if target_future is None and sdc_id is not None and str(target_id) == str(sdc_id):
                target_future = self._extract_cbv_ego_future(
                    current_state,
                    pluto_feature,
                    pred_local,
                    controlled_agent_id=controlled_agent_id,
                )
            if target_future is None:
                continue
            reward_out = reward.evaluate(
                eval_candidates,
                input_data_single,
                target_future,
                predictions=feasibility_predictions_local if feasibility_predictions_local is not None else pred_local,
                candidate_ref_idx=candidate_ref_idx,
                candidate_labels=candidate_labels,
            )
            target_total = np.asarray(reward_out.get("total", []), dtype=np.float32)
            if len(target_total) != len(eval_candidates):
                continue
            total += target_total
            for key in component_keys:
                value = np.asarray(
                    reward_out.get(key, np.zeros_like(target_total)),
                    dtype=np.float32,
                )
                if len(value) == len(eval_candidates):
                    component_sums[key] += value
            per_target[str(target_id)] = target_total
            used_targets += 1

        if used_targets == 0:
            return None, {}
        self._print_adversarial_score_debug(
            controlled_agent_id=controlled_agent_id,
            target_ids=target_ids,
            total=total,
            components=component_sums,
            candidate_labels=candidate_labels,
        )
        return total, per_target

    def _print_adversarial_score_debug(self, controlled_agent_id, target_ids, total, components, candidate_labels=None):
        if os.environ.get("UNITRAJ_ADV_SCORE_DEBUG", str(self.cfg.get("adv_score_debug", False))) not in (
            "1",
            "true",
            "True",
        ):
            return
        top_n = int(self.cfg.get("adv_score_debug_top_n", 5))
        if top_n <= 0 or total is None or len(total) == 0:
            return

        order = np.argsort(-np.asarray(total, dtype=np.float32))[:top_n]

        def _value(name, idx):
            values = np.asarray(components.get(name, []), dtype=np.float32)
            if idx >= len(values):
                return 0.0
            return float(values[idx])

        for idx in order:
            idx = int(idx)
            label = candidate_labels[idx] if candidate_labels is not None and idx < len(candidate_labels) else None
            print(
                "[ADV_SCORE]\n"
                f"agent={controlled_agent_id}\n"
                f"targets={[str(x) for x in target_ids]}\n"
                f"idx={idx}\n"
                f"label={label}\n"
                f"distance={_value('distance_risk', idx):.4f}\n"
                f"ttc={_value('ttc_reward', idx):.4f}\n"
                f"interaction={_value('interaction', idx):.4f}\n"
                f"approach={_value('approach', idx):.4f}\n"
                f"feasible={_value('feasibility', idx):.4f}\n"
                f"behavior={_value('behavior_disruption', idx):.4f}\n"
                f"total={float(total[idx]):.4f}"
            )

    def _build_cbv_rollout_sample(
            self,
            current_state,
            pluto_feature,
            out,
            pred_local,
            timestep,
            controlled_agent_id,
            batch_idx=0,
            selected_future_traj=None,
            score_selection="max",
    ):
        if controlled_agent_id is None:
            return None
        raw_logits = out.get("probability", None)
        raw_traj = out.get("trajectory", None)
        if raw_logits is None:
            return None
        if raw_traj is None:
            raw_traj = out.get("candidate_trajectories", None)
        if raw_traj is None:
            return None

        raw_logits = raw_logits[batch_idx].detach().cpu().numpy().astype(np.float32)
        raw_traj = raw_traj[batch_idx].detach().cpu().numpy().astype(np.float32)
        if raw_logits.ndim != 2 or raw_traj.ndim != 4:
            return None
        latent = None
        hidden = out.get("hidden", None)
        if hidden is not None:
            latent = hidden[batch_idx].detach().cpu().numpy().astype(np.float32)

        n_ref, n_mode = raw_logits.shape
        candidate_ref_idx = np.repeat(np.arange(n_ref, dtype=np.int64)[:, None], n_mode, axis=1)
        ego_future = self._extract_cbv_ego_future(
            current_state,
            pluto_feature,
            pred_local,
            controlled_agent_id=controlled_agent_id,
        )
        if ego_future is None:
            return None

        metadata = current_state.get("metadata", {}) if isinstance(current_state, dict) else {}
        return {
            "pluto_feature": {"data": pluto_feature},
            "old_logits": raw_logits,
            "candidate_trajectories": raw_traj,
            "ego_future": np.asarray(ego_future, dtype=np.float32),
            "candidate_ref_idx": candidate_ref_idx,
            "predictions": np.asarray(pred_local, dtype=np.float32),
            "latent": latent,
            "selected_future_traj": (
                None if selected_future_traj is None
                else np.asarray(selected_future_traj, dtype=np.float32)
            ),
            "info": {
                "frame_idx": int(timestep),
                "cbv_id": str(controlled_agent_id),
                "controlled_agent_id": str(controlled_agent_id),
                "score_selection": str(score_selection).lower(),
                "scenario_token": metadata.get("scenario_id", metadata.get("token", None)),
            },
        }

    def initialize_model(self):
        """Initialize the model for Pluto."""
        self.model = PlanningModel(config=self.cfg)
        model_ckpt = self.cfg.get('ckpt_path', None)
        if model_ckpt is not None and model_ckpt != "null":
            if not os.path.isabs(model_ckpt):
                model_ckpt = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")), model_ckpt)
            ckpt = torch.load(model_ckpt, map_location=self.device, weights_only=False)
            if 'state_dict' in ckpt:
                state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
                self.model.load_state_dict(state_dict, strict=False)
            else:
                self.model.load_state_dict(ckpt, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def _to_device(self, data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        if isinstance(data, dict):
            return {k: self._to_device(v, device) for k, v in data.items()}
        if isinstance(data, list):
            return [self._to_device(v, device) for v in data]
        return data

    def _slice_batch(self, data, batch_idx):
        if isinstance(data, torch.Tensor):
            if data.ndim > 0 and data.shape[0] > batch_idx:
                return data[batch_idx: batch_idx + 1]
            return data
        if isinstance(data, dict):
            return {k: self._slice_batch(v, batch_idx) for k, v in data.items()}
        if isinstance(data, list):
            return [self._slice_batch(v, batch_idx) for v in data]
        return data

    def _forward_features(self, pluto_features, role):
        from unitraj.datasets.Pluto_dataset.pluto_utils import collate_pluto_dicts, to_feature_tensor_dict

        input_dict = collate_pluto_dicts([to_feature_tensor_dict(feature) for feature in pluto_features])
        input_dict = self._to_device(input_dict, self.device)

        out = self.model.forward(input_dict)
        return input_dict, out

    def _rule_evaluator_config(self, score_selection="max", controlled_agent_id=None):
        eval_cfg = dict(self.cfg or {})
        if "num_frames_eval" not in eval_cfg and "eval_num_frames" in eval_cfg:
            eval_cfg["num_frames_eval"] = eval_cfg["eval_num_frames"]
        if "collision_num_frames_eval" not in eval_cfg:
            eval_cfg["collision_num_frames_eval"] = int(
                eval_cfg.get(
                    "prediction_collision_num_frames_eval",
                    eval_cfg.get("short_horizon_num_frames_eval", 20),
                )
            )
        is_av1_branch = controlled_agent_id is not None and str(score_selection).lower() == "max"
        if not is_av1_branch or not bool(eval_cfg.get("av1_rule_override_enabled", True)):
            return eval_cfg

        overrides = {
            "weight_progress": "av1_rule_weight_progress",
            "weight_comfort": "av1_rule_weight_comfort",
            "weight_ref_tracking": "av1_rule_weight_ref_tracking",
            "collision_penalty": "av1_rule_collision_penalty",
            "wrong_way_penalty": "av1_rule_wrong_way_penalty",
            "rule_based_feasible_margin": "av1_rule_based_feasible_margin",
        }
        defaults = {
            "weight_progress": 1.0,
            "weight_comfort": 2.0,
            "weight_ref_tracking": 0.5,
            "collision_penalty": 100.0,
            "wrong_way_penalty": 30.0,
        }
        for target_key, source_key in overrides.items():
            if source_key in eval_cfg:
                eval_cfg[target_key] = eval_cfg[source_key]
            elif target_key in defaults:
                eval_cfg[target_key] = defaults[target_key]
        return eval_cfg

    def _world_traj_to_prediction_local(self, traj, origin, angle, target_dim=5):
        arr = np.asarray(traj, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim != 3 or arr.shape[-1] < 2 or arr.shape[1] == 0:
            return None

        rot_to_local = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        out_dim = max(int(target_dim), 3)
        local = np.zeros((arr.shape[0], arr.shape[1], out_dim), dtype=np.float32)
        local[..., :2] = np.matmul(arr[..., :2] - origin, rot_to_local.T)

        if arr.shape[-1] >= 4:
            heading_vec = np.matmul(arr[..., 2:4], rot_to_local.T)
            local[..., 2] = np.arctan2(heading_vec[..., 1], heading_vec[..., 0])
        elif arr.shape[-1] >= 3:
            local[..., 2] = arr[..., 2] - float(angle)
        else:
            delta = np.diff(local[..., :2], axis=1, prepend=local[..., :1, :2])
            local[..., 2] = np.arctan2(delta[..., 1], delta[..., 0] + 1e-8)

        if out_dim >= 5 and arr.shape[-1] >= 6:
            local[..., 3:5] = np.matmul(arr[..., 4:6], rot_to_local.T)
        return local

    def _extra_collision_predictions_local(self, extra_collision_trajs, origin, angle, target_dim=5):
        extras = []
        for traj in extra_collision_trajs or []:
            local = self._world_traj_to_prediction_local(
                traj,
                origin,
                angle,
                target_dim=target_dim,
            )
            if local is not None:
                extras.append(local)
        if not extras:
            return None
        # Cross-model selected plans have a 50-point export window; background
        # replay may still have 80 points. Compare only their genuine common
        # horizon instead of padding or failing the mixed-length concatenation.
        horizon = min(item.shape[1] for item in extras)
        return np.concatenate([item[:, :horizon] for item in extras], axis=0).astype(np.float32)

    def _target_futures_to_local(self, target_futures_by_id, origin, angle, target_dim=5):
        local = {}
        for target_id, traj in (target_futures_by_id or {}).items():
            converted = self._world_traj_to_prediction_local(
                traj,
                origin,
                angle,
                target_dim=target_dim,
            )
            if converted is not None and len(converted) > 0:
                local[str(target_id)] = converted[0]
        return local

    def _agent_id_matches(self, lhs, rhs):
        lhs = str(lhs)
        rhs = str(rhs)
        return lhs == rhs or lhs.endswith(rhs[-5:]) or rhs.endswith(lhs[-5:])

    def _lookup_future_entry(self, futures_by_id, sources_by_id, target_id):
        for agent_id, traj in (futures_by_id or {}).items():
            if self._agent_id_matches(agent_id, target_id):
                return str(agent_id), traj, str((sources_by_id or {}).get(str(agent_id), "unknown"))
        return None, None, "missing"

    def _scenario_agent_state(self, current_state, timestep, agent_id):
        tracks = current_state.get("tracks", {}) if isinstance(current_state, dict) else {}
        track = tracks.get(str(agent_id))
        if track is None:
            track = tracks.get(agent_id)
        if not isinstance(track, dict):
            return None

        state = track.get("state", {})
        if not isinstance(state, dict):
            return None
        position = np.asarray(state.get("position", []), dtype=np.float32)
        if position.ndim != 2 or position.shape[-1] < 2 or len(position) == 0:
            return None
        idx = int(np.clip(timestep, 0, len(position) - 1))
        pos = position[idx, :2]
        if not np.isfinite(pos).all():
            return None

        heading_arr = np.asarray(state.get("heading", []), dtype=np.float32).reshape(-1)
        heading = float(heading_arr[idx]) if len(heading_arr) > idx else 0.0
        velocity = np.asarray(state.get("velocity", []), dtype=np.float32)
        vel = np.zeros((2,), dtype=np.float32)
        if velocity.ndim >= 2 and len(velocity) > idx and velocity.shape[-1] >= 2:
            value = velocity[idx, :2]
            if np.isfinite(value).all():
                vel = value.astype(np.float32)
        return pos.astype(np.float32), heading, vel

    def _ego_lane_cutin_rear_threat(self, current_state, timestep, controlled_agent_id, score_cfg):
        agent_id = str(score_cfg.get("av1_blocked_ego_lane_cutin_agent_id", "adv_1"))
        if controlled_agent_id is None or str(controlled_agent_id) != agent_id:
            return {"active": False}

        threat_id = str(score_cfg.get("av1_blocked_ego_lane_cutin_threat_agent_id", "adv_2"))
        agent_state = self._scenario_agent_state(current_state, timestep, controlled_agent_id)
        threat_state = self._scenario_agent_state(current_state, timestep, threat_id)
        if agent_state is None or threat_state is None:
            return {"active": False, "threat_agent": threat_id, "reason": "missing_state"}

        agent_pos, _, agent_vel = agent_state
        threat_pos, _, threat_vel = threat_state
        rel = threat_pos - agent_pos
        distance = float(np.linalg.norm(rel))
        rel_dir = rel / max(distance, 1e-3)
        closing_speed = float(-np.dot(threat_vel - agent_vel, rel_dir))
        min_dist = float(score_cfg.get("av1_blocked_ego_lane_cutin_threat_min_distance_m", 25.0))
        max_dist = float(score_cfg.get("av1_blocked_ego_lane_cutin_threat_max_distance_m", 40.0))
        min_closing = float(score_cfg.get("av1_blocked_ego_lane_cutin_threat_min_closing_mps", 0.0))
        active = min_dist <= distance <= max_dist and closing_speed >= min_closing
        return {
            "active": bool(active),
            "threat_agent": threat_id,
            "distance": round(distance, 3),
            "closing_speed": round(closing_speed, 3),
            "min_distance": round(min_dist, 3),
            "max_distance": round(max_dist, 3),
            "min_closing": round(min_closing, 3),
        }

    def _append_unique_future(self, futures, seen_traj_ids, traj):
        if traj is None:
            return
        traj_key = id(traj)
        if traj_key in seen_traj_ids:
            return
        seen_traj_ids.add(traj_key)
        futures.append(traj)

    def _filter_prediction_agents(self, pluto_feature, pred_local, controlled_agent_id=None, exclude_agent_ids=None):
        pred = np.asarray(pred_local, dtype=np.float32)
        if pred.ndim != 3 or len(pred) == 0:
            return pred
        exclude = [str(agent_id) for agent_id in (exclude_agent_ids or [])]
        if controlled_agent_id is not None:
            exclude.append(str(controlled_agent_id))
        if not exclude:
            return pred

        tokens = self._extract_predicted_agent_tokens(
            pluto_feature,
            pred,
            controlled_agent_id=controlled_agent_id,
        )
        if not tokens:
            return pred

        keep = []
        for idx in range(len(pred)):
            token = str(tokens[idx]) if idx < len(tokens) else None
            if token is None:
                keep.append(idx)
                continue
            if any(self._agent_id_matches(token, agent_id) for agent_id in exclude):
                continue
            keep.append(idx)
        return pred[keep] if keep else np.zeros((0, pred.shape[1], pred.shape[2]), dtype=np.float32)

    def _extract_replay_futures_by_id(self, scenario, timestep, exclude_agent_ids=None, horizon=None):
        if not isinstance(scenario, dict):
            return {}
        tracks = scenario.get("tracks", {})
        if not isinstance(tracks, dict):
            return {}

        horizon = int(horizon or self.cfg.get("future_steps", 80))
        start = int(max(0, timestep))
        exclude = [str(agent_id) for agent_id in (exclude_agent_ids or [])]
        replay_futures = {}

        for agent_id, track in tracks.items():
            agent_id = str(agent_id)
            if any(self._agent_id_matches(agent_id, excluded) for excluded in exclude):
                continue
            if not isinstance(track, dict):
                continue
            state = track.get("state", {})
            if not isinstance(state, dict):
                continue

            pos = np.asarray(state.get("position", []), dtype=np.float32)
            if pos.ndim != 2 or pos.shape[-1] < 2 or start >= len(pos):
                continue
            valid = np.asarray(state.get("valid", np.ones((len(pos),), dtype=np.bool_)), dtype=np.bool_)
            if len(valid) < len(pos):
                valid = np.pad(valid, (0, len(pos) - len(valid)), constant_values=False)
            if start >= len(valid) or not bool(valid[start]):
                continue

            end = min(len(pos), start + horizon)
            frame_count = max(0, end - start)
            if frame_count == 0:
                continue

            traj = np.full((horizon, 6), np.nan, dtype=np.float32)
            frame_valid = valid[start:end].astype(bool) & np.isfinite(pos[start:end, :2]).all(axis=-1)
            if not np.any(frame_valid):
                continue
            traj[:frame_count, :2] = pos[start:end, :2]

            heading = np.asarray(state.get("heading", []), dtype=np.float32)
            if heading.ndim > 1:
                heading = heading.reshape(-1)
            if len(heading) >= end:
                heading_slice = heading[start:end]
            else:
                delta = np.diff(pos[start:end, :2], axis=0, prepend=pos[start:start + 1, :2])
                heading_slice = np.arctan2(delta[:, 1], delta[:, 0] + 1e-8)
            traj[:frame_count, 2] = np.cos(heading_slice)
            traj[:frame_count, 3] = np.sin(heading_slice)

            vel = np.asarray(state.get("velocity", []), dtype=np.float32)
            if vel.ndim == 2 and vel.shape[-1] >= 2 and len(vel) >= end:
                traj[:frame_count, 4:6] = vel[start:end, :2]
            elif frame_count >= 2:
                traj[:frame_count, 4:6] = np.gradient(traj[:frame_count, :2], float(self.cfg.get("dt", 0.1)), axis=0)
            else:
                traj[:frame_count, 4:6] = 0.0

            invalid_idx = np.where(~frame_valid)[0]
            if len(invalid_idx) > 0:
                traj[invalid_idx, :] = np.nan
            replay_futures[agent_id] = traj
        return replay_futures

    def _circle_centers(self, xy, yaw, length, width):
        radius = float(width) * 0.5
        offset = max(0.0, float(length) * 0.5 - radius)
        offsets = np.array([-offset, 0.0, offset], dtype=np.float32)
        direction = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1).astype(np.float32)
        centers = xy[..., None, :] + direction[..., None, :] * offsets[None, :, None]
        return centers, radius

    def _wrap_yaw(self, yaw):
        return (np.asarray(yaw, dtype=np.float32) + np.pi) % (2.0 * np.pi) - np.pi

    def _trajectory_yaw_from_state(self, state, xy):
        arr = np.asarray(state, dtype=np.float32)
        xy = np.asarray(xy, dtype=np.float32)
        if arr.shape[-1] >= 4:
            norm = np.linalg.norm(arr[..., 2:4], axis=-1)
            if np.nanmedian(norm) < 1.5:
                return np.arctan2(arr[..., 3], arr[..., 2]).astype(np.float32)
        if arr.shape[-1] >= 3:
            return self._wrap_yaw(arr[..., 2])
        dxy = np.diff(xy, axis=-2, prepend=xy[..., :1, :])
        return np.arctan2(dxy[..., 1], dxy[..., 0] + 1e-8).astype(np.float32)


    def _candidate_kinematic_feasibility(self, candidates_local, evaluator_cfg, current_speed=None):
        """Hard physical-consistency gate for executable candidate trajectories.

        Scoring can prefer a learned mode that is numerically finite yet physically
        impossible (for example a 0.1 s tens-of-metres jump or >100 m/s velocity).
        Such candidates must be removed before score selection, not clipped after
        selection, because post-hoc clipping would make XY and velocity inconsistent.
        """
        candidates = np.asarray(candidates_local, dtype=np.float64)
        count = len(candidates) if candidates.ndim >= 1 else 0
        if candidates.ndim != 3 or count == 0:
            return np.zeros((count,), dtype=bool), []
        if not bool(evaluator_cfg.get("candidate_kinematic_gate_enabled", True)):
            return np.ones((count,), dtype=bool), []

        dt = max(float(evaluator_cfg.get("candidate_dt_sec", 0.1)), 1e-3)
        max_speed = float(evaluator_cfg.get("candidate_max_speed_mps", 16.7))
        max_accel = float(evaluator_cfg.get("candidate_max_accel_mps2", 6.0))
        max_decel = float(evaluator_cfg.get("candidate_max_decel_mps2", 8.0))
        max_jerk = float(evaluator_cfg.get("candidate_max_jerk_mps3", 30.0))
        speed_geom_tol = float(
            evaluator_cfg.get("candidate_speed_geometry_tolerance_mps", 6.0)
        )
        first_step_extra = float(evaluator_cfg.get("candidate_first_step_extra_m", 1.0))
        first_step_floor = float(evaluator_cfg.get("candidate_first_step_max_m", 2.0))

        feasible = np.ones((count,), dtype=bool)
        metrics = []
        for idx in range(count):
            traj = candidates[idx]
            reason = []
            finite = bool(np.isfinite(traj).all())
            if not finite or traj.shape[0] < 2 or traj.shape[1] < 2:
                feasible[idx] = False
                metrics.append({"idx": int(idx), "reason": ["non_finite_or_short"]})
                continue

            xy = traj[:, :2]
            geom_speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
            first_step_distance = float(np.linalg.norm(xy[0]))
            if current_speed is not None and np.isfinite(float(current_speed)):
                allowed_first_step = max(
                    first_step_floor,
                    max(float(current_speed), 0.0) * dt + first_step_extra,
                )
                if first_step_distance > allowed_first_step:
                    reason.append("first_step_jump")
            else:
                allowed_first_step = first_step_floor
                if first_step_distance > first_step_floor:
                    reason.append("first_step_jump")

            supplied_speed = None
            if traj.shape[1] >= 6:
                supplied_speed = np.linalg.norm(traj[:, 4:6], axis=1)
                if current_speed is not None and np.isfinite(float(current_speed)):
                    speed_series = np.concatenate(
                        [[max(float(current_speed), 0.0)], supplied_speed]
                    )
                else:
                    speed_series = supplied_speed
            else:
                speed_series = np.concatenate([geom_speed[:1], geom_speed])

            first_interval_speed = first_step_distance / dt
            geom_speed_full = np.concatenate([[first_interval_speed], geom_speed])
            observed_speed = np.concatenate(
                [geom_speed_full, supplied_speed] if supplied_speed is not None else [geom_speed_full]
            )
            observed_speed = observed_speed[np.isfinite(observed_speed)]
            max_observed_speed = float(np.max(observed_speed)) if len(observed_speed) else float("inf")

            # A hard speed ceiling must distinguish a newly-planned overspeed from
            # recovery when the physical vehicle is ALREADY above the configured
            # ceiling.  Requiring every t+0.1 point to be <= max_speed in that state
            # would demand an impossible instantaneous deceleration and would even
            # reject our deterministic safety-brake trajectory.
            current_speed_value = None
            if current_speed is not None and np.isfinite(float(current_speed)):
                current_speed_value = max(float(current_speed), 0.0)
            overspeed_recovery_active = bool(
                current_speed_value is not None
                and current_speed_value > max_speed
            )
            recovery_peak_tolerance = max(
                float(evaluator_cfg.get(
                    "candidate_overspeed_recovery_tolerance_mps",
                    0.25,
                )),
                0.0,
            )
            recovery_min_drop = max(
                float(evaluator_cfg.get(
                    "candidate_overspeed_min_recovery_drop_mps",
                    0.5,
                )),
                0.0,
            )
            allowed_peak_speed = float(max_speed)
            required_terminal_speed = None
            if supplied_speed is not None and len(supplied_speed) > 0:
                terminal_observed_speed = float(supplied_speed[-1])
            elif len(geom_speed) > 0:
                terminal_observed_speed = float(geom_speed[-1])
            else:
                terminal_observed_speed = float(first_interval_speed)

            if overspeed_recovery_active:
                allowed_peak_speed = current_speed_value + recovery_peak_tolerance
                required_terminal_speed = max(
                    float(max_speed),
                    current_speed_value - recovery_min_drop,
                )
                if max_observed_speed > allowed_peak_speed:
                    reason.append("max_speed")
                if terminal_observed_speed > required_terminal_speed:
                    reason.append("overspeed_no_recovery")
            elif max_observed_speed > max_speed:
                reason.append("max_speed")

            accel = np.diff(speed_series) / dt if len(speed_series) >= 2 else np.zeros((0,))
            max_observed_accel = float(np.max(accel)) if len(accel) else 0.0
            max_observed_decel = float(max(0.0, -np.min(accel))) if len(accel) else 0.0
            if max_observed_accel > max_accel:
                reason.append("max_accel")
            if max_observed_decel > max_decel:
                reason.append("max_decel")

            jerk = np.diff(accel) / dt if len(accel) >= 2 else np.zeros((0,))
            max_observed_jerk = float(np.max(np.abs(jerk))) if len(jerk) else 0.0
            diagnostic_reason = []
            if max_observed_jerk > max_jerk:
                if bool(evaluator_cfg.get("candidate_hard_jerk_gate_enabled", False)):
                    reason.append("max_jerk")
                else:
                    diagnostic_reason.append("max_jerk")

            speed_geom_error = 0.0
            if supplied_speed is not None and len(geom_speed) > 0:
                mean_state_speed = 0.5 * (supplied_speed[:-1] + supplied_speed[1:])
                speed_geom_error = float(np.max(np.abs(mean_state_speed - geom_speed)))
                if speed_geom_error > speed_geom_tol:
                    reason.append("speed_geometry_mismatch")

            feasible[idx] = len(reason) == 0
            if reason or diagnostic_reason:
                metrics.append({
                    "idx": int(idx),
                    "reason": reason,
                    "diagnostic": diagnostic_reason,
                    "max_speed": round(max_observed_speed, 3),
                    "configured_max_speed": round(float(max_speed), 3),
                    "allowed_peak_speed": round(float(allowed_peak_speed), 3),
                    "current_speed": (
                        None if current_speed_value is None
                        else round(float(current_speed_value), 3)
                    ),
                    "terminal_speed": round(float(terminal_observed_speed), 3),
                    "required_terminal_speed": (
                        None if required_terminal_speed is None
                        else round(float(required_terminal_speed), 3)
                    ),
                    "overspeed_recovery": bool(overspeed_recovery_active),
                    "max_accel": round(max_observed_accel, 3),
                    "max_decel": round(max_observed_decel, 3),
                    "max_jerk": round(max_observed_jerk, 3),
                    "speed_geometry_error": round(speed_geom_error, 3),
                    "first_step_distance": round(first_step_distance, 3),
                    "allowed_first_step": round(allowed_first_step, 3),
                })
        return feasible, metrics


    def _build_local_failsafe_brake_candidate(
            self,
            current_speed,
            horizon,
            planner_channels,
            control_channels,
            evaluator_cfg,
    ):
        """Build a map-independent, physically consistent emergency brake candidate.

        This is only used when every model/OSM candidate fails the hard kinematic
        gate.  It keeps the scenario alive without executing an invalid learned
        trajectory.  The local model frame is actor-aligned at the current step,
        so +x is the current forward direction.
        """
        dt = max(float(evaluator_cfg.get("candidate_dt_sec", 0.1)), 1e-3)
        decel = max(float(evaluator_cfg.get("candidate_failsafe_brake_decel_mps2", 3.0)), 1e-3)
        speed0 = max(float(current_speed or 0.0), 0.0)
        horizon = max(int(horizon), 1)
        times = (np.arange(horizon, dtype=np.float64) + 1.0) * dt
        stop_time = speed0 / decel if speed0 > 0.0 else 0.0
        effective_t = np.minimum(times, stop_time)
        distance = speed0 * effective_t - 0.5 * decel * effective_t * effective_t
        distance = np.maximum(distance, 0.0)
        speed = np.maximum(speed0 - decel * times, 0.0)

        planner = np.zeros((horizon, int(planner_channels)), dtype=np.float32)
        planner[:, 0] = distance.astype(np.float32)
        if planner.shape[1] >= 3:
            # Pluto planner candidate format is local [x, y, heading].
            planner[:, 2] = 0.0

        control = None
        if control_channels is not None:
            control = np.zeros((horizon, int(control_channels)), dtype=np.float32)
            control[:, 0] = distance.astype(np.float32)
            if control.shape[1] >= 4:
                # Pluto control format is [x, y, cos(h), sin(h), vx, vy].
                control[:, 2] = 1.0
                control[:, 3] = 0.0
            elif control.shape[1] >= 3:
                control[:, 2] = 0.0
            if control.shape[1] >= 6:
                control[:, 4] = speed.astype(np.float32)
                control[:, 5] = 0.0
        return planner, control

    def _candidate_collision_flags_local(self, candidates_local, predictions_local, evaluator_cfg):
        candidates = np.asarray(candidates_local, dtype=np.float32)
        pred = np.asarray(predictions_local, dtype=np.float32)
        if candidates.ndim != 3 or len(candidates) == 0 or pred.ndim != 3 or len(pred) == 0:
            return np.zeros((len(candidates) if candidates.ndim >= 1 else 0,), dtype=bool)

        collision_frames = int(
            evaluator_cfg.get(
                "collision_num_frames_eval",
                evaluator_cfg.get("num_frames_eval", candidates.shape[1]),
            )
        )
        t_eval = candidates.shape[1] if collision_frames <= 0 else min(candidates.shape[1], collision_frames)
        if t_eval <= 0:
            return np.zeros((len(candidates),), dtype=bool)

        pred = pred[:, :t_eval]
        if pred.shape[1] < t_eval and pred.shape[1] > 0:
            pred = np.concatenate(
                [pred, np.repeat(pred[:, -1:], t_eval - pred.shape[1], axis=1)],
                axis=1,
            )
        if pred.shape[1] == 0:
            return np.zeros((len(candidates),), dtype=bool)

        traj_xy = candidates[:, :t_eval, :2]
        traj_yaw = self._trajectory_yaw_from_state(candidates[:, :t_eval], traj_xy)
        other_xy = pred[..., :2]
        other_yaw = self._trajectory_yaw_from_state(pred, other_xy)
        ego_centers, ego_radius = self._circle_centers(
            traj_xy,
            traj_yaw,
            evaluator_cfg.get("ego_length", 4.8),
            evaluator_cfg.get("ego_width", 2.0),
        )
        other_centers, other_radius = self._circle_centers(
            other_xy,
            other_yaw,
            evaluator_cfg.get("agent_length", 4.8),
            evaluator_cfg.get("agent_width", 2.0),
        )
        threshold_sq = float((ego_radius + other_radius) ** 2)
        collided = np.zeros((len(candidates),), dtype=bool)
        horizon = min(t_eval, other_centers.shape[1])
        other_centers_t = np.transpose(other_centers[:, :horizon], (1, 0, 2, 3))
        for i in range(len(candidates)):
            diff = ego_centers[i, :horizon, :, None, None, :] - other_centers_t[:, None, :, :, :]
            collided[i] = bool(np.any(np.sum(diff * diff, axis=-1) < threshold_sq))
        return collided

    def _candidate_collision_debug_local(
            self,
            candidates_local,
            predictions_local,
            evaluator_cfg,
            candidate_labels=None,
            prediction_labels=None,
            max_entries=8,
    ):
        candidates = np.asarray(candidates_local, dtype=np.float32)
        pred = np.asarray(predictions_local, dtype=np.float32)
        if candidates.ndim != 3 or len(candidates) == 0 or pred.ndim != 3 or len(pred) == 0:
            return []

        collision_frames = int(
            evaluator_cfg.get(
                "collision_num_frames_eval",
                evaluator_cfg.get("num_frames_eval", candidates.shape[1]),
            )
        )
        t_eval = candidates.shape[1] if collision_frames <= 0 else min(candidates.shape[1], collision_frames)
        if t_eval <= 0:
            return []

        pred = pred[:, :t_eval]
        if pred.shape[1] < t_eval and pred.shape[1] > 0:
            pred = np.concatenate(
                [pred, np.repeat(pred[:, -1:], t_eval - pred.shape[1], axis=1)],
                axis=1,
            )
        if pred.shape[1] == 0:
            return []

        traj_xy = candidates[:, :t_eval, :2]
        traj_yaw = self._trajectory_yaw_from_state(candidates[:, :t_eval], traj_xy)
        other_xy = pred[..., :2]
        other_yaw = self._trajectory_yaw_from_state(pred, other_xy)
        ego_centers, ego_radius = self._circle_centers(
            traj_xy,
            traj_yaw,
            evaluator_cfg.get("ego_length", 4.8),
            evaluator_cfg.get("ego_width", 2.0),
        )
        other_centers, other_radius = self._circle_centers(
            other_xy,
            other_yaw,
            evaluator_cfg.get("agent_length", 4.8),
            evaluator_cfg.get("agent_width", 2.0),
        )
        threshold = float(ego_radius + other_radius)
        threshold_sq = threshold ** 2
        horizon = min(t_eval, other_centers.shape[1])
        other_centers_t = np.transpose(other_centers[:, :horizon], (1, 0, 2, 3))
        labels = list(candidate_labels or [])
        pred_labels = list(prediction_labels or [])
        dt = float(evaluator_cfg.get("dt", evaluator_cfg.get("eval_dt", 0.1)))

        entries = []
        for i in range(len(candidates)):
            diff = ego_centers[i, :horizon, :, None, None, :] - other_centers_t[:, None, :, :, :]
            dist_sq = np.sum(diff * diff, axis=-1)
            hit = dist_sq < threshold_sq
            if not hit.any():
                continue
            hit_indices = np.argwhere(hit)
            first = hit_indices[np.argmin(hit_indices[:, 0])]
            t_idx, ego_circle_idx, pred_idx, pred_circle_idx = [int(x) for x in first]
            first_dist = float(np.sqrt(dist_sq[t_idx, ego_circle_idx, pred_idx, pred_circle_idx]))
            min_flat = np.unravel_index(int(np.argmin(dist_sq)), dist_sq.shape)
            min_t, min_ego_circle, min_pred_idx, min_pred_circle = [int(x) for x in min_flat]
            min_dist = float(np.sqrt(dist_sq[min_flat]))
            current_xy_dist = float(np.linalg.norm(traj_xy[i, 0] - other_xy[pred_idx, 0]))
            collision_xy_dist = float(np.linalg.norm(traj_xy[i, t_idx] - other_xy[pred_idx, t_idx]))
            entries.append({
                "candidate_idx": int(i),
                "candidate_label": labels[i] if i < len(labels) else None,
                "prediction_idx": int(pred_idx),
                "prediction_label": pred_labels[pred_idx] if pred_idx < len(pred_labels) else None,
                "first_collision_step": int(t_idx),
                "first_collision_time_s": round(float(t_idx * dt), 3),
                "first_circle_distance": round(first_dist, 3),
                "circle_threshold": round(threshold, 3),
                "first_clearance": round(first_dist - threshold, 3),
                "current_xy_distance": round(current_xy_dist, 3),
                "collision_xy_distance": round(collision_xy_dist, 3),
                "ego_circle": int(ego_circle_idx),
                "other_circle": int(pred_circle_idx),
                "min_circle_distance": round(min_dist, 3),
                "min_distance_step": int(min_t),
                "min_prediction_idx": int(min_pred_idx),
                "min_prediction_label": pred_labels[min_pred_idx] if min_pred_idx < len(pred_labels) else None,
                "min_ego_circle": int(min_ego_circle),
                "min_other_circle": int(min_pred_circle),
            })
            if len(entries) >= int(max_entries):
                break
        return entries

    def _candidate_score_debug_enabled(self):
        return (
            bool(self.cfg.get("candidate_score_debug", False))
            or os.environ.get("UNITRAJ_CANDIDATE_SCORE_DEBUG", "0") == "1"
        )

    def _print_candidate_score_debug(
        self,
        *,
        timestep,
        controlled_agent_id,
        score_selection_key,
        best_idx,
        selected_original_idx,
        rule_based_scores,
        final_scores,
        sorted_probability,
        prob_norm_scores,
        feasible_mask,
        collision_flags,
        osm_labels,
        danger_brake_idx=None,
        danger_brake_reason=None,
        score_components=None,
    ):
        if len(final_scores) == 0:
            return

        def _num(arr, idx, default=0.0):
            if arr is None or idx < 0 or idx >= len(arr):
                return default
            value = arr[idx]
            if not np.isfinite(value):
                return str(value)
            return round(float(value), 6)

        def _component_item(idx):
            if not isinstance(score_components, dict):
                return None
            fields = (
                "displacement",
                "path_length",
                "mean_speed",
                "terminal_speed",
                "max_speed",
                "progress",
                "progress_term",
                "comfort",
                "comfort_term",
                "ref_tracking",
                "ref_tracking_term",
                "wrong_way",
                "wrong_way_term",
                "collision_term",
            )
            item = {}
            for key in fields:
                arr = score_components.get(key)
                if arr is None or idx < 0 or idx >= len(arr):
                    continue
                value = arr[idx]
                if isinstance(value, (bool, np.bool_)):
                    item[key] = bool(value)
                elif np.issubdtype(np.asarray(value).dtype, np.bool_):
                    item[key] = bool(value)
                elif np.isfinite(value):
                    item[key] = round(float(value), 6)
                else:
                    item[key] = str(value)
            return item or None

        def _item(idx):
            label = osm_labels[idx] if 0 <= idx < len(osm_labels) else None
            item = {
                "idx": int(idx),
                "orig_idx": int(selected_original_idx[idx]) if 0 <= idx < len(selected_original_idx) else int(idx),
                "selected": bool(idx == best_idx),
                "rule": _num(rule_based_scores, idx),
                "final": _num(final_scores, idx),
                "prob": _num(sorted_probability, idx),
                "prob_norm": _num(prob_norm_scores, idx),
                "feasible": bool(feasible_mask[idx]) if 0 <= idx < len(feasible_mask) else True,
                "collision": bool(collision_flags[idx]) if 0 <= idx < len(collision_flags) else False,
                "osm_label": label,
            }
            components = _component_item(idx)
            if components is not None:
                item["components"] = components
            return item

        top_n = int(self.cfg.get("candidate_score_debug_top_n", 8))
        if score_selection_key == "min":
            ranked = np.argsort(final_scores)[:max(1, top_n)]
        else:
            ranked = np.argsort(-final_scores)[:max(1, top_n)]
        brake_indices = [i for i, label in enumerate(osm_labels) if label == "brake"]
        ego_lane_cutin_indices = [i for i, label in enumerate(osm_labels) if self._is_ego_lane_cutin_label(label)]
        selected = _item(int(best_idx)) if 0 <= int(best_idx) < len(final_scores) else None
        print(
            "[CandidateScore]",
            {
                "t": int(timestep),
                "agent": "ego" if controlled_agent_id is None else str(controlled_agent_id),
                "score_selection": str(score_selection_key),
                "selected": selected,
                "danger_brake_idx": None if danger_brake_idx is None else int(danger_brake_idx),
                "danger_brake_reason": danger_brake_reason,
                "top": [_item(int(i)) for i in ranked],
                "osm_brake": [_item(int(i)) for i in brake_indices],
                "osm_ego_lane_cutin": [_item(int(i)) for i in ego_lane_cutin_indices],
                "collision_count": int(np.sum(collision_flags)) if len(collision_flags) > 0 else 0,
                "candidate_count": int(len(final_scores)),
            },
        )

    def _extract_candidates(self, out, batch_idx):
        """Extract topk candidates and probs for one agent (6ch output)."""
        prob = out["probability"][batch_idx].cpu().numpy()
        n_ref, n_mode = prob.shape
        T = out["candidate_trajectories"].shape[-2]
        traj = out.get("trajectory", None)
        if traj is not None and traj.shape[-1] == 6:
            cands = traj[batch_idx].detach().cpu().numpy()
        else:
            # Convert 3ch (xy, heading) → 6ch (xy, cos, sin, vx, vy)
            raw = out["candidate_trajectories"][batch_idx].detach().cpu().numpy()
            xy = raw[..., :2]
            h = raw[..., 2] if raw.shape[-1] >= 3 else np.zeros_like(xy[..., 0])
            vel = np.gradient(xy, 0.1, axis=-2)
            cands = np.concatenate([xy, np.cos(h)[..., None], np.sin(h)[..., None], vel], axis=-1).astype(np.float32)
        C = cands.shape[-1]
        cands = cands.reshape(-1, T, C)
        prob = prob.reshape(-1)
        from scipy.special import softmax
        prob = softmax(prob)
        topk = self.cfg.get("candidate_max_num", 20)
        sorted_idx = np.argsort(-prob)
        return cands[sorted_idx][:topk], prob[sorted_idx][:topk], sorted_idx[:topk], n_mode

    def _candidates_local_to_world(self, candidates, input_dict, batch_idx):
        """Match the coordinate system used by rollout ScoreNet training data.

        Rollout records use ``selected_future_traj`` after
        ``_postprocess_single_output`` has transformed it to world coordinates.
        Raw model candidates, however, are expressed in each controlled CBV's
        local frame.  A joint AV1×AV2 scorer must therefore transform every
        candidate into world coordinates before its forward pass.
        """
        world = np.array(candidates, dtype=np.float32, copy=True)
        if world.ndim != 3 or world.shape[-1] < 2:
            return world
        data = input_dict.data if hasattr(input_dict, "data") else input_dict
        origin = data["origin"][batch_idx].detach().cpu().numpy()[:2]
        angle = float(data["angle"][batch_idx].detach().cpu().numpy())
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        world[..., :2] = np.matmul(world[..., :2], rot_mat) + origin
        if world.shape[-1] >= 4:
            world[..., 2:4] = np.matmul(world[..., 2:4], rot_mat)
        elif world.shape[-1] >= 3:
            world[..., 2] += angle
        if world.shape[-1] >= 6:
            world[..., 4:6] = np.matmul(world[..., 4:6], rot_mat)
        return world

    def _score_full_matrix(self, cands1, cands2, probs1, probs2, ego_latent, ego_fut, mlp_weight=0.7):
        """Batch score all (i,j) pairs, return scores for joint CBV selection."""
        import torch
        N1, N2 = len(cands1), len(cands2)
        av1_all = np.tile(cands1[:, None], (1, N2, 1, 1)).reshape(-1, *cands1.shape[1:])
        av2_all = np.tile(cands2[None, :], (N1, 1, 1, 1)).reshape(-1, *cands2.shape[1:])
        lat_all = np.tile(ego_latent[None], (N1 * N2, 1))
        lat_t = torch.from_numpy(lat_all).float().to(self.device)
        a1_t = torch.from_numpy(av1_all).float().to(self.device)
        a2_t = torch.from_numpy(av2_all).float().to(self.device)
        kwargs = {}
        if ego_fut is not None:
            kwargs["ego_future"] = torch.from_numpy(np.tile(ego_fut[None], (N1*N2, 1, 1))).float().to(self.device)
        with torch.no_grad():
            mlp_scores = self.score_net.model(lat_t, a1_t, a2_t, **kwargs).cpu().numpy().reshape(N1, N2)
        prob_weight = 1.0 - mlp_weight
        prob_matrix = probs1[:, None] * probs2[None, :]
        mlp_min = float(np.min(mlp_scores))
        mlp_max = float(np.max(mlp_scores))
        if mlp_max - mlp_min > 1e-6:
            mlp_norm = (mlp_scores - mlp_min) / (mlp_max - mlp_min)
        else:
            mlp_norm = np.full_like(mlp_scores, 0.5, dtype=np.float64)
        prob_max = float(np.max(prob_matrix))
        if prob_max > 1e-12:
            prob_norm = prob_matrix / prob_max
        else:
            prob_norm = np.full_like(prob_matrix, 0.5, dtype=np.float64)
        combined = mlp_weight * mlp_norm + prob_weight * prob_norm
        best_i, best_j = np.unravel_index(combined.argmax(), (N1, N2))
        return int(best_i), int(best_j), mlp_scores, combined

    def _select_full_matrix_pair(self, combined_scores):
        policy = str(self.cfg.get("score_net_pair_selection", "topk_sample")).lower()
        flat_scores = np.asarray(combined_scores, dtype=np.float64).reshape(-1)
        if len(flat_scores) == 0:
            return 0, 0

        if policy in ("argmax", "max", "stress"):
            chosen_flat = int(np.argmax(flat_scores))
        else:
            topk = int(self.cfg.get("score_net_pair_topk", 8))
            topk = int(np.clip(topk, 1, len(flat_scores)))
            top_indices = np.argsort(-flat_scores)[:topk]
            temperature = float(self.cfg.get("score_net_pair_temperature", 0.35))
            if temperature <= 1e-6 or topk == 1:
                chosen_flat = int(top_indices[0])
            else:
                logits = (flat_scores[top_indices] - float(np.max(flat_scores[top_indices]))) / temperature
                probs = softmax(logits)
                if not np.isfinite(probs).all() or float(np.sum(probs)) <= 0.0:
                    chosen_flat = int(top_indices[0])
                else:
                    chosen_flat = int(np.random.choice(top_indices, p=probs))

        return tuple(int(x) for x in np.unravel_index(chosen_flat, combined_scores.shape))

    def _postprocess_single_output(
            self,
            current_state,
            timestep,
            controlled_agent_id,
            pluto_feature,
            input_dict,
            out,
            batch_idx=0,
            score_selection="max",
            adversarial_target_agent_ids=None,
            forced_best_idx=None,
            per_candidate_mlp_scores=None,
            extra_collision_trajs=None,
            actual_target_futures_by_id=None,
            use_prediction_background=None,
    ):
        ref_lines = pluto_feature['reference_line']

        planner_candidates = out["candidate_trajectories"][batch_idx].detach().cpu().numpy()
        control_candidates = (
            out["trajectory"][batch_idx].detach().cpu().numpy()
            if out["trajectory"] is not None else None
        )
        pred = out['output_prediction'][batch_idx].detach().cpu().numpy()
        pred = self._trim_background_prediction(
            pluto_feature, pred, controlled_agent_id=controlled_agent_id,
        )

        probability = out["probability"][batch_idx].cpu().numpy()

        n_ref, n_mode, T, C = planner_candidates.shape
        planner_candidates = planner_candidates.reshape(-1, T, C)
        probability = probability.reshape(-1)
        if control_candidates is not None and len(control_candidates.shape) == 4:
            control_candidates = control_candidates.reshape(
                n_ref * n_mode, control_candidates.shape[-2], control_candidates.shape[-1]
            )

        input_data = input_dict.data if hasattr(input_dict, 'data') else input_dict
        input_data_single = self._slice_batch(input_data, batch_idx)
        flat_ref_idx = np.repeat(np.arange(n_ref, dtype=np.int64), n_mode)
        score_selection_key = str(score_selection).lower()
        topk = int(self.cfg.get('candidate_max_num', 20))
        sorted_idx = np.argsort(-probability)
        use_all_for_adversarial = score_selection_key == "adversarial"
        if use_all_for_adversarial:
            selected_idx = sorted_idx
        else:
            selected_idx = sorted_idx[:topk]
        sorted_candidate_trajectories = planner_candidates[selected_idx]
        sorted_osm_labels = [None] * len(selected_idx)
        sorted_control_candidates = None
        if control_candidates is not None:
            sorted_control_candidates = control_candidates[selected_idx]
        sorted_probability = softmax(probability[selected_idx])
        sorted_ref_idx = None # 每个候选轨迹对应的参考线索引
        if len(flat_ref_idx) == len(probability):
            sorted_ref_idx = flat_ref_idx[selected_idx].astype(np.int64)

        sorted_candidate_trajectories_local = np.array(sorted_candidate_trajectories, copy=True)
        sorted_control_candidates_local = (
            None if sorted_control_candidates is None else np.array(sorted_control_candidates, copy=True)
        )
        pred_local = np.array(pred, copy=True)
        origin = input_data['origin'][batch_idx].detach().cpu().numpy()[:2]
        angle = float(input_data['angle'][batch_idx].detach().cpu().numpy())
        rot_mat = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        extra_pred_local = self._extra_collision_predictions_local(
            extra_collision_trajs,
            origin,
            angle,
            target_dim=pred_local.shape[-1] if pred_local.ndim == 3 else 5,
        )
        if use_prediction_background is None:
            use_prediction_background = controlled_agent_id is None
        use_background_prediction = bool(use_prediction_background)
        actual_ids = list((actual_target_futures_by_id or {}).keys())
        background_pred_local = (
            self._filter_prediction_agents(
                pluto_feature,
                pred_local,
                controlled_agent_id=controlled_agent_id,
                exclude_agent_ids=actual_ids,
            )
            if use_background_prediction
            else None
        )
        rule_pred_parts = []
        rule_pred_labels = []
        if extra_pred_local is not None:
            rule_pred_parts.append(extra_pred_local)
            rule_pred_labels.extend([f"extra_collision_{i}" for i in range(len(extra_pred_local))])
        if background_pred_local is not None and len(background_pred_local) > 0:
            rule_pred_parts.append(background_pred_local.astype(np.float32))
            background_tokens = self._extract_predicted_agent_tokens(
                pluto_feature,
                pred_local,
                controlled_agent_id=controlled_agent_id,
            )
            exclude_for_labels = [str(agent_id) for agent_id in actual_ids]
            if controlled_agent_id is not None:
                exclude_for_labels.append(str(controlled_agent_id))
            background_labels = []
            for i in range(len(pred_local)):
                token = str(background_tokens[i]) if i < len(background_tokens) else str(i)
                if any(self._agent_id_matches(token, excluded) for excluded in exclude_for_labels):
                    continue
                background_labels.append(f"background_{token}")
            if len(background_labels) != len(background_pred_local):
                background_labels = [f"background_{i}" for i in range(len(background_pred_local))]
            rule_pred_labels.extend(background_labels)
        if rule_pred_parts:
            common_horizon = min(part.shape[1] for part in rule_pred_parts)
            rule_pred_local = np.concatenate([part[:, :common_horizon] for part in rule_pred_parts], axis=0)
        else:
            pred_dim = pred_local.shape[-1] if pred_local.ndim == 3 else 5
            pred_t = pred_local.shape[1] if pred_local.ndim == 3 else T
            rule_pred_local = np.zeros((0, pred_t, pred_dim), dtype=np.float32)
        if len(rule_pred_labels) != len(rule_pred_local):
            rule_pred_labels = [f"prediction_{i}" for i in range(len(rule_pred_local))]
        actual_target_futures_local = self._target_futures_to_local(
            actual_target_futures_by_id,
            origin,
            angle,
            target_dim=pred_local.shape[-1] if pred_local.ndim == 3 else 5,
        )

        # Normal CBV collision feasibility must see the planned Ego future and
        # already-resolved CBV futures, not only model background prediction.
        # Skip the controlled agent itself and deduplicate aliases such as
        # {"ego", sdc_id} that intentionally carry the same trajectory.
        actual_collision_parts = []
        actual_collision_labels = []
        actual_collision_xy = []
        for target_id, target_future in (actual_target_futures_local or {}).items():
            if controlled_agent_id is not None and self._agent_id_matches(
                    str(target_id), str(controlled_agent_id)):
                continue
            arr = np.asarray(target_future, dtype=np.float32)
            if arr.ndim != 2 or len(arr) == 0:
                continue
            duplicate = False
            for previous_xy in actual_collision_xy:
                horizon_cmp = min(len(arr), len(previous_xy))
                if horizon_cmp > 0 and np.allclose(
                        arr[:horizon_cmp, :2],
                        previous_xy[:horizon_cmp, :2],
                        atol=1e-4,
                        rtol=0.0,
                ):
                    duplicate = True
                    break
            if duplicate:
                continue
            actual_collision_parts.append(arr[None, ...])
            actual_collision_xy.append(arr)
            actual_collision_labels.append("actual_%s" % str(target_id))

        if actual_collision_parts:
            collision_parts = []
            collision_labels = []
            if rule_pred_local is not None and len(rule_pred_local) > 0:
                collision_parts.append(np.asarray(rule_pred_local, dtype=np.float32))
                collision_labels.extend(list(rule_pred_labels))
            collision_parts.extend(actual_collision_parts)
            collision_labels.extend(actual_collision_labels)
            common_horizon = min(part.shape[1] for part in collision_parts)
            common_dim = min(part.shape[2] for part in collision_parts)
            rule_pred_local = np.concatenate(
                [part[:, :common_horizon, :common_dim] for part in collision_parts],
                axis=0,
            )
            rule_pred_labels = collision_labels

        planner_candidates_for_eval_local = sorted_candidate_trajectories_local
        if sorted_candidate_trajectories_local.shape[1] > 0:
            planner_candidates_for_eval_local = np.concatenate(
                [
                    sorted_candidate_trajectories_local[..., 0:1, :],
                    sorted_candidate_trajectories_local,
                ],
                axis=-2,
            )

        debug_eval_cfg = self._rule_evaluator_config(
            score_selection=score_selection_key,
            controlled_agent_id=controlled_agent_id,
        )
        # Collision futures and candidate futures both start at t+0.1 s.
        # Do not prepend a duplicate candidate point here; that shifts collision
        # timing by one frame and can miss fast Ego/CBV interactions.
        candidate_collision_flags = self._candidate_collision_flags_local(
            sorted_candidate_trajectories_local,
            rule_pred_local,
            debug_eval_cfg,
        )
        kinematic_eval_candidates = (
            sorted_control_candidates_local
            if sorted_control_candidates_local is not None
            else sorted_candidate_trajectories_local
        )
        current_motion_state = self._scenario_agent_state(
            current_state,
            timestep,
            (
                controlled_agent_id
                if controlled_agent_id is not None
                else (current_state.get("metadata", {}) or {}).get("sdc_id", "ego")
            ),
        )
        current_speed_for_gate = (
            None if current_motion_state is None
            else float(np.linalg.norm(np.asarray(current_motion_state[2], dtype=np.float64)[:2]))
        )
        candidate_kinematic_mask, candidate_kinematic_metrics = self._candidate_kinematic_feasibility(
            kinematic_eval_candidates,
            debug_eval_cfg,
            current_speed=current_speed_for_gate,
        )

        # A safety filter must never terminate the entire AutoScene episode.  If
        # every learned/OSM mode is physically invalid, replace one candidate with
        # a deterministic map-independent brake trajectory and continue planning.
        if len(candidate_kinematic_mask) > 0 and not bool(np.any(candidate_kinematic_mask)):
            invalid_snapshot = list(candidate_kinematic_metrics[:8])
            replace_idx = int(len(candidate_kinematic_mask) - 1)
            planner_channels = int(sorted_candidate_trajectories_local.shape[-1])
            control_channels = (
                None if sorted_control_candidates_local is None
                else int(sorted_control_candidates_local.shape[-1])
            )
            fallback_planner, fallback_control = self._build_local_failsafe_brake_candidate(
                current_speed_for_gate,
                sorted_candidate_trajectories_local.shape[1],
                planner_channels,
                control_channels,
                debug_eval_cfg,
            )
            sorted_candidate_trajectories_local[replace_idx] = fallback_planner
            sorted_candidate_trajectories[replace_idx] = fallback_planner
            if sorted_control_candidates_local is not None and fallback_control is not None:
                sorted_control_candidates_local[replace_idx] = fallback_control
                sorted_control_candidates[replace_idx] = fallback_control
            if replace_idx < len(sorted_osm_labels):
                sorted_osm_labels[replace_idx] = "safety_brake"
            if replace_idx < len(selected_idx):
                selected_idx[replace_idx] = -1
            if sorted_ref_idx is not None and replace_idx < len(sorted_ref_idx):
                sorted_ref_idx[replace_idx] = -1
            if replace_idx < len(sorted_probability):
                sorted_probability[replace_idx] = min(
                    float(sorted_probability[replace_idx]),
                    1e-9,
                )
                prob_sum_after_fallback = float(np.sum(sorted_probability))
                if prob_sum_after_fallback > 1e-12:
                    sorted_probability = sorted_probability / prob_sum_after_fallback
            if per_candidate_mlp_scores is not None and replace_idx < len(per_candidate_mlp_scores):
                per_candidate_mlp_scores = np.asarray(per_candidate_mlp_scores, dtype=np.float64).copy()
                per_candidate_mlp_scores[replace_idx] = float(np.min(per_candidate_mlp_scores))

            planner_candidates_for_eval_local = sorted_candidate_trajectories_local
            if sorted_candidate_trajectories_local.shape[1] > 0:
                planner_candidates_for_eval_local = np.concatenate(
                    [
                        sorted_candidate_trajectories_local[..., 0:1, :],
                        sorted_candidate_trajectories_local,
                    ],
                    axis=-2,
                )
            candidate_collision_flags = self._candidate_collision_flags_local(
                sorted_candidate_trajectories_local,
                rule_pred_local,
                debug_eval_cfg,
            )
            kinematic_eval_candidates = (
                sorted_control_candidates_local
                if sorted_control_candidates_local is not None
                else sorted_candidate_trajectories_local
            )
            candidate_kinematic_mask, candidate_kinematic_metrics = self._candidate_kinematic_feasibility(
                kinematic_eval_candidates,
                debug_eval_cfg,
                current_speed=current_speed_for_gate,
            )
            print(
                "[CandidateKinematicsFallback]",
                {
                    "t": int(timestep),
                    "agent": "ego" if controlled_agent_id is None else str(controlled_agent_id),
                    "current_speed": None if current_speed_for_gate is None else round(float(current_speed_for_gate), 3),
                    "replaced_idx": int(replace_idx),
                    "fallback_valid": bool(candidate_kinematic_mask[replace_idx]),
                    "invalid_examples_before_fallback": invalid_snapshot,
                },
            )
            if not bool(candidate_kinematic_mask[replace_idx]):
                fallback_metric = next(
                    (
                        item for item in candidate_kinematic_metrics
                        if int(item.get("idx", -1)) == int(replace_idx)
                    ),
                    None,
                )
                raise RuntimeError(
                    "internal safety-brake candidate failed kinematic feasibility at t=%d agent=%s fallback_metric=%s"
                    % (
                        int(timestep),
                        "ego" if controlled_agent_id is None else str(controlled_agent_id),
                        fallback_metric,
                    )
                )

        # IMPORTANT: keep the pure kinematic mask independent from the hard
        # collision mask.  np.asarray(bool_array) may alias the original array;
        # the in-place &= below must not overwrite candidate_kinematic_mask.
        hard_feasible_mask = np.asarray(candidate_kinematic_mask, dtype=bool).copy()
        if (
            score_selection_key != "adversarial"
            and bool(debug_eval_cfg.get("candidate_hard_collision_gate_enabled", True))
        ):
            hard_feasible_mask &= ~np.asarray(candidate_collision_flags, dtype=bool)
        if self._candidate_score_debug_enabled() and candidate_kinematic_metrics:
            print(
                "[CandidateKinematics]",
                {
                    "t": int(timestep),
                    "agent": "ego" if controlled_agent_id is None else str(controlled_agent_id),
                    "valid": int(np.sum(candidate_kinematic_mask)),
                    "count": int(len(candidate_kinematic_mask)),
                    "invalid_examples": candidate_kinematic_metrics[:8],
                },
            )

        if forced_best_idx is not None:
            # Matrix search pre-selected best pair — skip rule evaluator
            rule_based_scores = np.zeros((len(sorted_candidate_trajectories_local),), dtype=np.float32)
            rule_score_components = None
        elif per_candidate_mlp_scores is not None and score_selection_key != "adversarial":
            # CBV mode: MLP+probability scoring, skip rule evaluator
            rule_based_scores = np.zeros((len(sorted_candidate_trajectories_local),), dtype=np.float32)
            rule_score_components = None
        elif score_selection_key == "adversarial" and not bool(self.cfg.get("av2_adversarial_use_rule_feasible_mask", False)) and bool(self.cfg.get("av2_adversarial_skip_rule_eval", True)):
            rule_based_scores = np.zeros((len(sorted_candidate_trajectories_local),), dtype=np.float32)
            rule_score_components = None
        else:
            if bool(self.cfg.get("use_fast_rule_evaluator", True)):
                evaluator = FastTrajectoryEvaluator(debug_eval_cfg)
            else:
                evaluator = TrajectoryEvaluator(debug_eval_cfg)
            if self._candidate_score_debug_enabled() and hasattr(evaluator, "score_components"):
                rule_score_components = evaluator.score_components(
                    planner_candidates_for_eval_local,
                    input_data_single,
                    rule_pred_local,
                    candidate_ref_idx=sorted_ref_idx,
                )
                rule_based_scores = np.asarray(rule_score_components.get("score", []), dtype=np.float32)
            else:
                rule_score_components = None
                rule_based_scores = evaluator.evaluate(
                    planner_candidates_for_eval_local,
                    input_data_single,
                    rule_pred_local,
                    candidate_ref_idx=sorted_ref_idx,
                )

        sorted_candidate_trajectories[..., :2] = (
                np.matmul(sorted_candidate_trajectories[..., :2], rot_mat) + origin
        )
        sorted_candidate_trajectories[..., 2] += angle

        if sorted_control_candidates is not None:
            sorted_control_candidates = np.array(sorted_control_candidates, copy=True)
            sorted_control_candidates[..., :2] = (
                    np.matmul(sorted_control_candidates[..., :2], rot_mat) + origin
            )
            if sorted_control_candidates.shape[-1] >= 4:
                sorted_control_candidates[..., 2:4] = np.matmul(
                    sorted_control_candidates[..., 2:4], rot_mat
                )
            elif sorted_control_candidates.shape[-1] >= 3:
                heading = sorted_control_candidates[..., 2] + angle
                sorted_control_candidates[..., 2] = heading
            if sorted_control_candidates.shape[-1] >= 6:
                sorted_control_candidates[..., 4:6] = np.matmul(
                    sorted_control_candidates[..., 4:6], rot_mat
                )

        pred[..., :2] = np.matmul(pred[..., :2], rot_mat) + origin
        pred[..., 2] += angle
        pred[..., 3:5] = np.matmul(pred[..., 3:5], rot_mat)

        if ref_lines is not None:
            import copy
            ref_lines = copy.deepcopy(ref_lines)
            ref_lines['position'] = np.matmul(ref_lines['position'], rot_mat) + origin
            if 'orientation' in ref_lines:
                ref_lines['orientation'] += angle

        use_mlp_scoring = (per_candidate_mlp_scores is not None
                           and forced_best_idx is None
                           and score_selection_key != "adversarial")
        danger_brake_idx = None
        danger_brake_reason = None
        safe_ego_lane_cutin_indices = np.asarray([], dtype=np.int64)
        target_ego_lane_cutin_indices = np.asarray([], dtype=np.int64)

        if use_mlp_scoring:
            # ===== CBV: MLP + probability =====
            mlp_weight = float(self.cfg.get('mlp_based_score_weight', 0.7))
            mlp_s = per_candidate_mlp_scores.astype(np.float64)
            mlp_min = float(np.min(mlp_s))
            mlp_max = float(np.max(mlp_s))
            if mlp_max - mlp_min > 1e-6:
                mlp_norm = (mlp_s - mlp_min) / (mlp_max - mlp_min)
            else:
                mlp_norm = np.ones_like(mlp_s, dtype=np.float64)

            prob_sum = float(np.sum(sorted_probability))
            if prob_sum > 1e-8:
                prob_norm = sorted_probability.astype(np.float64) / prob_sum
            else:
                prob_norm = np.full_like(sorted_probability, 1.0 / max(len(sorted_probability), 1), dtype=np.float64)

            prob_norm_scores = prob_norm
            prob_weight = 1.0 - mlp_weight
            raw_scores = mlp_weight * mlp_norm + prob_weight * prob_norm
            feasible_mask = np.asarray(hard_feasible_mask, dtype=bool).copy()
            final_scores = np.where(feasible_mask, raw_scores, -1e9).astype(np.float64)

        else:
            # ===== Ego: rule + probability =====
            learning_weight = float(self.cfg.get('learning_based_score_weight', 0.25))
            rule_weight = float(self.cfg.get('rule_based_score_weight', 0.9))
            score_cfg = debug_eval_cfg
            feasible_margin = float(score_cfg.get('rule_based_feasible_margin', 25.0))
            min_keep = int(score_cfg.get(
                'rule_based_min_keep',
                min(8, len(rule_based_scores)) if len(rule_based_scores) > 0 else 0,
            ))

            feasible_mask = np.ones_like(rule_based_scores, dtype=bool)
            if len(rule_based_scores) > 0:
                best_rule = float(np.max(rule_based_scores))
                feasible_mask = rule_based_scores >= (best_rule - feasible_margin)
                if feasible_mask.sum() < max(1, min_keep):
                    top_rule_idx = np.argsort(-rule_based_scores)[:max(1, min_keep)]
                    feasible_mask = np.zeros_like(rule_based_scores, dtype=bool)
                    feasible_mask[top_rule_idx] = True

            brake_indices = np.asarray(
                [i for i, label in enumerate(sorted_osm_labels) if label == "brake"],
                dtype=np.int64,
            )
            safe_brake_indices = brake_indices[
                brake_indices < len(candidate_collision_flags)
            ]
            safe_brake_indices = safe_brake_indices[
                ~candidate_collision_flags[safe_brake_indices]
            ]
            ego_lane_cutin_indices = np.asarray(
                [
                    i for i, label in enumerate(sorted_osm_labels)
                    if self._is_ego_lane_cutin_label(label)
                ],
                dtype=np.int64,
            )
            valid_ego_lane_cutin_indices = ego_lane_cutin_indices[
                ego_lane_cutin_indices < len(candidate_collision_flags)
            ]
            safe_ego_lane_cutin_indices = ego_lane_cutin_indices[
                ego_lane_cutin_indices < len(candidate_collision_flags)
            ]
            safe_ego_lane_cutin_indices = safe_ego_lane_cutin_indices[
                ~candidate_collision_flags[safe_ego_lane_cutin_indices]
            ]
            allow_unsafe_ego_lane_cutin = (
                bool(score_cfg.get("av1_blocked_ego_lane_cutin_allow_unsafe", True))
                and str(controlled_agent_id) == str(score_cfg.get("av1_blocked_ego_lane_cutin_agent_id", "adv_1"))
            )
            if (
                bool(score_cfg.get("osm_brake_keep_if_no_collision", True))
                and len(safe_brake_indices) > 0
                and len(feasible_mask) == len(rule_based_scores)
            ):
                feasible_mask[safe_brake_indices] = True
            if len(safe_ego_lane_cutin_indices) > 0 and len(feasible_mask) == len(rule_based_scores):
                feasible_mask[safe_ego_lane_cutin_indices] = True
            if (
                allow_unsafe_ego_lane_cutin
                and len(valid_ego_lane_cutin_indices) > 0
                and len(feasible_mask) == len(rule_based_scores)
            ):
                feasible_mask[valid_ego_lane_cutin_indices] = True

            # Soft rule/OSM keep logic may widen the set, but hard kinematic
            # and normal-mode collision feasibility always wins.
            feasible_mask &= np.asarray(hard_feasible_mask, dtype=bool)

            final_scores = np.full_like(sorted_probability, -1e9, dtype=np.float64)
            prob_norm_scores = np.zeros_like(sorted_probability, dtype=np.float64)

            if feasible_mask.any():
                feasible_rule = rule_based_scores[feasible_mask].astype(np.float64)
                feasible_prob = sorted_probability[feasible_mask].astype(np.float64)

                prob_norm_scores[feasible_mask] = feasible_prob
                final_scores[feasible_mask] = rule_weight * feasible_rule + learning_weight * feasible_prob
            else:
                # Keep all scores suppressed; the explicit safety fallback below
                # chooses brake/lowest-motion among kinematically valid options.
                prob_norm_scores = sorted_probability.astype(np.float64)

            collision_ratio = (
                float(np.sum(candidate_collision_flags)) / float(max(len(candidate_collision_flags), 1))
            )
            collision_count = int(np.sum(candidate_collision_flags))
            ego_lane_cutin_bonus_indices = np.asarray([], dtype=np.int64)
            ego_lane_cutin_prefer_indices = np.asarray([], dtype=np.int64)
            ego_lane_cutin_bonus = float(score_cfg.get("av1_blocked_ego_lane_cutin_bonus", 0.0))
            ego_lane_cutin_agent_id = str(score_cfg.get("av1_blocked_ego_lane_cutin_agent_id", "adv_1"))
            ego_lane_cutin_min_collisions = int(score_cfg.get("av1_blocked_ego_lane_cutin_min_collision_count", 1))
            ego_lane_cutin_prefer_margin = float(score_cfg.get("av1_blocked_ego_lane_cutin_prefer_margin", 0.0))
            ego_lane_cutin_rear_threat = self._ego_lane_cutin_rear_threat(
                current_state,
                timestep,
                controlled_agent_id,
                score_cfg,
            )
            target_ego_lane_cutin_indices = np.asarray(
                [
                    int(i) for i in (
                        valid_ego_lane_cutin_indices
                        if allow_unsafe_ego_lane_cutin and len(valid_ego_lane_cutin_indices) > 0
                        else safe_ego_lane_cutin_indices
                    )
                    if 0 <= int(i) < len(sorted_osm_labels)
                ],
                dtype=np.int64,
            )

            if (
                ego_lane_cutin_bonus > 0.0
                and score_selection_key != "adversarial"
                and str(controlled_agent_id) == ego_lane_cutin_agent_id
                and (
                    collision_count >= max(1, ego_lane_cutin_min_collisions)
                    or bool(ego_lane_cutin_rear_threat.get("active", False))
                )
                and len(target_ego_lane_cutin_indices) > 0
            ):
                ego_lane_cutin_bonus_indices = target_ego_lane_cutin_indices
                if len(ego_lane_cutin_bonus_indices) > 0:
                    final_scores[ego_lane_cutin_bonus_indices] += ego_lane_cutin_bonus
                if ego_lane_cutin_prefer_margin > 0.0:
                    valid_target_scores = final_scores[target_ego_lane_cutin_indices]
                    if np.isfinite(valid_target_scores).any():
                        current_best = float(np.max(final_scores))
                        best_target = float(np.max(valid_target_scores))
                        needed = current_best + ego_lane_cutin_prefer_margin - best_target
                        if needed > 0.0:
                            final_scores[target_ego_lane_cutin_indices] += needed
                            ego_lane_cutin_prefer_indices = target_ego_lane_cutin_indices

            danger_brake_ratio = float(score_cfg.get("osm_brake_force_collision_ratio", 0.9))
            danger_brake_min_collisions = int(score_cfg.get("osm_brake_force_min_collision_count", 3))
            brake_agent_key = self._brake_agent_key(controlled_agent_id)
            brake_hold_until = int(self._osm_brake_hold_until.get(brake_agent_key, -1))
            hold_active = int(timestep) <= brake_hold_until
            if (
                bool(score_cfg.get("osm_brake_force_on_collision_risk", True))
                and score_selection_key != "adversarial"
                and len(target_ego_lane_cutin_indices) == 0
                and len(safe_brake_indices) > 0
                and collision_count >= max(1, danger_brake_min_collisions)
                and collision_ratio >= danger_brake_ratio
            ):
                safe_brake_scores = rule_based_scores[safe_brake_indices]
                danger_brake_idx = int(safe_brake_indices[int(np.argmax(safe_brake_scores))])
                danger_brake_reason = "collision_ratio"
                hold_steps = int(score_cfg.get("osm_brake_hold_steps", 5))
                self._osm_brake_hold_until[brake_agent_key] = int(timestep) + max(0, hold_steps)
            elif (
                bool(score_cfg.get("osm_brake_hold_enabled", True))
                and score_selection_key != "adversarial"
                and hold_active
                and len(target_ego_lane_cutin_indices) == 0
                and len(safe_brake_indices) > 0
            ):
                safe_brake_scores = rule_based_scores[safe_brake_indices]
                danger_brake_idx = int(safe_brake_indices[int(np.argmax(safe_brake_scores))])
                danger_brake_reason = "hold"
            elif not hold_active:
                self._osm_brake_hold_until.pop(brake_agent_key, None)

            suppress_brake_without_risk = (
                bool(score_cfg.get("osm_brake_select_only_on_risk", True))
                and danger_brake_idx is None
                and len(brake_indices) > 0
            )
            if suppress_brake_without_risk:
                valid_brake_indices = brake_indices[brake_indices < len(final_scores)]
                final_scores[valid_brake_indices] = -1e9

            if self._brake_debug_enabled() and score_selection_key != "adversarial":
                brake_debug_indices = safe_brake_indices.tolist() if len(safe_brake_indices) > 0 else brake_indices.tolist()

                def _score_item(idx):
                    idx = int(idx)
                    return {
                        "idx": idx,
                        "label": sorted_osm_labels[idx] if 0 <= idx < len(sorted_osm_labels) else None,
                        "safe": bool(idx < len(candidate_collision_flags) and not candidate_collision_flags[idx]),
                        "collision": bool(idx < len(candidate_collision_flags) and candidate_collision_flags[idx]),
                        "rule": None if idx >= len(rule_based_scores) else round(float(rule_based_scores[idx]), 6),
                        "final": None if idx >= len(final_scores) else round(float(final_scores[idx]), 6),
                        "prob": None if idx >= len(sorted_probability) else round(float(sorted_probability[idx]), 6),
                    }

                print(
                    "[BrakeDebug]",
                    {
                        "t": int(timestep),
                        "agent": "ego" if controlled_agent_id is None else str(controlled_agent_id),
                        "reason": danger_brake_reason,
                        "selected_brake_idx": None if danger_brake_idx is None else int(danger_brake_idx),
                        "collision_count": collision_count,
                        "candidate_count": int(len(candidate_collision_flags)),
                        "collision_ratio": round(float(collision_ratio), 3),
                        "ratio_threshold": round(float(danger_brake_ratio), 3),
                        "min_collision_count": int(danger_brake_min_collisions),
                        "safe_brake_count": int(len(safe_brake_indices)),
                        "safe_ego_lane_cutin_count": int(len(safe_ego_lane_cutin_indices)),
                        "valid_ego_lane_cutin_count": int(len(valid_ego_lane_cutin_indices)),
                        "allow_unsafe_ego_lane_cutin": bool(allow_unsafe_ego_lane_cutin),
                        "ego_lane_cutin_bonus": round(float(ego_lane_cutin_bonus), 6),
                        "ego_lane_cutin_bonus_indices": [
                            int(x) for x in ego_lane_cutin_bonus_indices.tolist()
                        ],
                        "ego_lane_cutin_prefer_indices": [
                            int(x) for x in ego_lane_cutin_prefer_indices.tolist()
                        ],
                        "ego_lane_cutin_rear_threat": ego_lane_cutin_rear_threat,
                        "brake_indices": [int(x) for x in brake_indices.tolist()],
                        "suppress_brake_without_risk": bool(suppress_brake_without_risk),
                        "brake_candidates": [_score_item(i) for i in brake_debug_indices[:4]],
                        "hold_active": bool(hold_active),
                        "hold_until": int(brake_hold_until),
                        "hold_remaining": max(0, int(brake_hold_until) - int(timestep)),
                        "hold_steps_cfg": int(score_cfg.get("osm_brake_hold_steps", 5)),
                        "collision_frames": int(
                            score_cfg.get(
                                "collision_num_frames_eval",
                                score_cfg.get("num_frames_eval", planner_candidates_for_eval_local.shape[1]),
                            )
                        ),
                        "circle_radius": {
                            "ego": round(float(score_cfg.get("ego_width", 2.0)) * 0.5, 3),
                            "other": round(float(score_cfg.get("agent_width", 2.0)) * 0.5, 3),
                            "threshold": round(
                                float(score_cfg.get("ego_width", 2.0)) * 0.5
                                + float(score_cfg.get("agent_width", 2.0)) * 0.5,
                                3,
                            ),
                        },
                        "collision_examples": self._candidate_collision_debug_local(
                            planner_candidates_for_eval_local,
                            rule_pred_local,
                            score_cfg,
                            candidate_labels=sorted_osm_labels,
                            prediction_labels=rule_pred_labels,
                            max_entries=int(score_cfg.get("brake_debug_max_examples", 8)),
                        ),
                    },
                )

        adversarial_scores = None
        adversarial_scores_by_target = {}
        cutin_lock_reason = False
        ego_lane_cutin_lock_reason = False
        if forced_best_idx is not None:
            best_idx = int(forced_best_idx)
        elif danger_brake_idx is not None:
            best_idx = int(danger_brake_idx)
        else:
            if score_selection_key == "adversarial":
                adversarial_eval_candidates = self._build_adversarial_eval_candidates(
                    sorted_control_candidates_local,
                    sorted_candidate_trajectories_local,
                )
                adversarial_scores, adversarial_scores_by_target = self._score_adversarial_candidates(
                    current_state=current_state,
                    pluto_feature=pluto_feature,
                    input_data_single=input_data_single,
                    pred_local=pred_local,
                    eval_candidates=adversarial_eval_candidates,
                    candidate_ref_idx=sorted_ref_idx,
                    controlled_agent_id=controlled_agent_id,
                    adversarial_target_agent_ids=adversarial_target_agent_ids,
                    actual_target_futures_local=actual_target_futures_local,
                    feasibility_predictions_local=rule_pred_local,
                    candidate_labels=sorted_osm_labels,
                )
                if adversarial_scores is not None:
                    if bool(self.cfg.get("av2_adversarial_use_rule_feasible_mask", False)):
                        selectable_scores = np.where(feasible_mask, adversarial_scores, -np.inf)
                        if not np.isfinite(selectable_scores).any():
                            selectable_scores = adversarial_scores
                    else:
                        selectable_scores = adversarial_scores
                    adversarial_brake_indices = np.asarray(
                        [i for i, label in enumerate(sorted_osm_labels) if label == "brake"],
                        dtype=np.int64,
                    )
                    valid_adversarial_brake_indices = adversarial_brake_indices[
                        adversarial_brake_indices < len(selectable_scores)
                    ]
                    if len(valid_adversarial_brake_indices) > 0:
                        non_brake_scores = np.array(selectable_scores, copy=True)
                        non_brake_scores[valid_adversarial_brake_indices] = -np.inf
                        if np.isfinite(non_brake_scores).any():
                            selectable_scores = non_brake_scores
                    best_idx = int(np.argmax(selectable_scores))
                else:
                    best_idx = int(np.argmax(final_scores))
            elif score_selection_key == "min":
                # Pure Pluto has no explicit OSM brake candidate.  Treat ``min``
                # as fail-safe low-motion selection instead of argmin(final_score),
                # because very negative scores can be suppression sentinels.
                candidate_xy = np.asarray(
                    sorted_candidate_trajectories_local[..., :2],
                    dtype=np.float64,
                )
                motion = np.sum(
                    np.linalg.norm(np.diff(candidate_xy, axis=1), axis=-1),
                    axis=1,
                )
                selectable_motion = np.where(feasible_mask, motion, np.inf)
                if not np.isfinite(selectable_motion).any():
                    selectable_motion = motion
                best_idx = int(np.argmin(selectable_motion))
            else:
                best_idx = int(np.argmax(final_scores))
                best_idx, ego_lane_cutin_lock_reason = self._select_with_ego_lane_cutin_lock(
                    timestep=timestep,
                    controlled_agent_id=controlled_agent_id,
                    best_idx=int(best_idx),
                    score_selection_key=score_selection_key,
                    sorted_osm_labels=sorted_osm_labels,
                    final_scores=final_scores,
                    safe_ego_lane_cutin_indices=target_ego_lane_cutin_indices,
                )
        best_idx, cutin_lock_reason = self._select_with_cutin_lock(
            timestep=timestep,
            controlled_agent_id=controlled_agent_id,
            best_idx=int(best_idx),
            score_selection_key=score_selection_key,
            sorted_osm_labels=sorted_osm_labels,
            adversarial_scores=adversarial_scores,
        )


        # Final hard safety enforcement.  ScoreNet locks, cut-in locks, or soft
        # rule keep-masks must never resurrect a physically impossible candidate.
        # For normal max/min planning collision is also a hard feasibility rule.
        kinematic_only_mask = np.asarray(candidate_kinematic_mask, dtype=bool)
        hard_selection_mask = np.asarray(hard_feasible_mask, dtype=bool)
        if score_selection_key == "adversarial":
            hard_selection_mask = kinematic_only_mask

        old_best_idx = int(best_idx)
        if not (0 <= int(best_idx) < len(hard_selection_mask)) or not bool(hard_selection_mask[int(best_idx)]):
            preferred_mask = np.asarray(feasible_mask, dtype=bool) & hard_selection_mask
            if not preferred_mask.any():
                preferred_mask = hard_selection_mask.copy()

            if preferred_mask.any():
                if score_selection_key == "min":
                    motion_source = (
                        sorted_control_candidates_local
                        if sorted_control_candidates_local is not None
                        else sorted_candidate_trajectories_local
                    )
                    motion_xy = np.asarray(motion_source[..., :2], dtype=np.float64)
                    motion = np.sum(
                        np.linalg.norm(np.diff(motion_xy, axis=1), axis=-1),
                        axis=1,
                    )
                    best_idx = int(np.argmin(np.where(preferred_mask, motion, np.inf)))
                elif score_selection_key == "adversarial" and adversarial_scores is not None:
                    best_idx = int(np.argmax(np.where(preferred_mask, adversarial_scores, -np.inf)))
                else:
                    best_idx = int(np.argmax(np.where(preferred_mask, final_scores, -np.inf)))
            else:
                # If every normal candidate is predicted to collide, prefer a
                # kinematically valid brake/lowest-motion trajectory instead of
                # falling back to an arbitrary high-probability collision mode.
                brake_kin = np.asarray(
                    [
                        i for i, label in enumerate(sorted_osm_labels)
                        if label in ("brake", "safety_brake") and i < len(kinematic_only_mask) and kinematic_only_mask[i]
                    ],
                    dtype=np.int64,
                )
                if len(brake_kin) > 0:
                    best_idx = int(brake_kin[0])
                elif kinematic_only_mask.any():
                    motion_source = (
                        sorted_control_candidates_local
                        if sorted_control_candidates_local is not None
                        else sorted_candidate_trajectories_local
                    )
                    motion_xy = np.asarray(motion_source[..., :2], dtype=np.float64)
                    motion = np.sum(
                        np.linalg.norm(np.diff(motion_xy, axis=1), axis=-1),
                        axis=1,
                    )
                    best_idx = int(np.argmin(np.where(kinematic_only_mask, motion, np.inf)))
                else:
                    raise RuntimeError(
                        "all Pluto candidates failed kinematic feasibility at t=%d agent=%s"
                        % (int(timestep), "ego" if controlled_agent_id is None else str(controlled_agent_id))
                    )

        if old_best_idx != int(best_idx) and self._candidate_score_debug_enabled():
            print(
                "[CandidateSafetyOverride]",
                {
                    "t": int(timestep),
                    "agent": "ego" if controlled_agent_id is None else str(controlled_agent_id),
                    "score_selection": str(score_selection_key),
                    "old_idx": int(old_best_idx),
                    "new_idx": int(best_idx),
                    "old_collision": bool(
                        0 <= old_best_idx < len(candidate_collision_flags)
                        and candidate_collision_flags[old_best_idx]
                    ),
                    "old_kinematic": bool(
                        0 <= old_best_idx < len(kinematic_only_mask)
                        and kinematic_only_mask[old_best_idx]
                    ),
                },
            )

        if self._candidate_score_debug_enabled():
            self._print_candidate_score_debug(
                timestep=timestep,
                controlled_agent_id=controlled_agent_id,
                score_selection_key=score_selection_key,
                best_idx=best_idx,
                selected_original_idx=selected_idx,
                rule_based_scores=rule_based_scores,
                final_scores=final_scores,
                sorted_probability=sorted_probability,
                prob_norm_scores=prob_norm_scores,
                feasible_mask=feasible_mask,
                collision_flags=candidate_collision_flags,
                osm_labels=sorted_osm_labels,
                danger_brake_idx=danger_brake_idx,
                danger_brake_reason=danger_brake_reason,
                score_components=rule_score_components,
            )
        if cutin_lock_reason and score_selection_key == "adversarial":
            selected_label = sorted_osm_labels[best_idx] if 0 <= int(best_idx) < len(sorted_osm_labels) else None
            # print(
            #     "[CutinLock]",
            #     {
            #         "t": int(timestep),
            #         "agent": None if controlled_agent_id is None else str(controlled_agent_id),
            #         "best_idx": int(best_idx),
            #         "label": selected_label,
            #         "reason": str(cutin_lock_reason),
            #     },
            # )
        if ego_lane_cutin_lock_reason:
            selected_label = sorted_osm_labels[best_idx] if 0 <= int(best_idx) < len(sorted_osm_labels) else None
            # print(
            #     "[EgoLaneCutinLock]",
            #     {
            #         "t": int(timestep),
            #         "agent": None if controlled_agent_id is None else str(controlled_agent_id),
            #         "best_idx": int(best_idx),
            #         "label": selected_label,
            #         "reason": str(ego_lane_cutin_lock_reason),
            #     },
            # )
        selected_control_traj = None
        if sorted_control_candidates is not None:
            selected_control_traj = sorted_control_candidates[best_idx]
            if selected_control_traj.shape[-1] == 3:
                heading = selected_control_traj[..., 2]
                vx = np.gradient(selected_control_traj[..., 0], 0.1)
                vy = np.gradient(selected_control_traj[..., 1], 0.1)
                selected_control_traj = np.concatenate(
                    [
                        selected_control_traj[..., :2],
                        np.cos(heading)[..., None],
                        np.sin(heading)[..., None],
                        vx[..., None],
                        vy[..., None],
                    ],
                    axis=-1,
                )
            elif selected_control_traj.shape[-1] >= 4 and selected_control_traj.shape[-1] < 6:
                vel_xy = np.gradient(selected_control_traj[..., :2], 0.1, axis=0)
                selected_control_traj = np.concatenate(
                    [selected_control_traj[..., :4], vel_xy[..., :2]], axis=-1
                )

        if selected_control_traj is None:
            planner_selected = sorted_candidate_trajectories[best_idx]
            planner_xy = planner_selected
            if planner_xy.shape[-1] >= 3:
                heading = planner_xy[..., 2]
            else:
                diff = np.gradient(planner_xy[..., :2], axis=0)
                heading = np.arctan2(diff[..., 1], diff[..., 0])
            vel_xy = np.gradient(planner_xy[..., :2], 0.1, axis=0)
            selected_control_traj = np.concatenate(
                [
                    planner_xy[..., :2],
                    np.cos(heading)[..., None],
                    np.sin(heading)[..., None],
                    vel_xy[..., :2],
                ],
                axis=-1,
            )
        T_out = selected_control_traj.shape[0]
        C_out = min(selected_control_traj.shape[-1], 6)
        pred_traj = np.zeros((16, min(T_out, 80), 6), dtype=np.float32)
        pred_len = min(80, T_out)
        pred_traj[0, :pred_len, :C_out] = selected_control_traj[:pred_len, :C_out]

        info = {}
        if 'agent_tokens' in pluto_feature:
            info['agent_tokens'] = pluto_feature['agent_tokens']
            info['feature_primary_agent_id'] = str(pluto_feature['agent_tokens'][0])
        if controlled_agent_id is not None:
            info['controlled_agent_id'] = str(controlled_agent_id)
            if self.cfg["rl_finetuning_mode"]:
                cbv_rollout_sample = self._build_cbv_rollout_sample(
                    current_state=current_state,
                    pluto_feature=pluto_feature,
                    out=out,
                    pred_local=pred_local,
                    timestep=timestep,
                    controlled_agent_id=controlled_agent_id,
                    batch_idx=batch_idx,
                    selected_future_traj=selected_control_traj[:pred_len, :C_out],
                    score_selection=score_selection,
                )
                if cbv_rollout_sample is not None:
                    info['cbv_rollout_samples'] = [cbv_rollout_sample]
        info['prediction'] = pred
        info['prediction_agent_tokens'] = self._extract_predicted_agent_tokens(
            pluto_feature,
            pred,
            controlled_agent_id=controlled_agent_id,
        )
        info['best_idx'] = best_idx
        info['selected_original_idx'] = int(selected_idx[int(best_idx)]) if 0 <= int(best_idx) < len(selected_idx) else int(best_idx)
        info['selected_ref_idx'] = (
            None if sorted_ref_idx is None or not (0 <= int(best_idx) < len(sorted_ref_idx))
            else int(sorted_ref_idx[int(best_idx)])
        )
        if self._tjwest_ref_audit_enabled():
            info['tjwest_reference_audit'] = self._emit_tjwest_reference_audit(
                timestep=timestep,
                controlled_agent_id=controlled_agent_id,
                ref_lines=ref_lines,
                selected_ref_idx=info['selected_ref_idx'],
            )
        info['score_selection'] = str(score_selection).lower()
        info['selected_osm_label'] = sorted_osm_labels[best_idx] if 0 <= int(best_idx) < len(sorted_osm_labels) else None
        info['cutin_lock_active'] = bool(cutin_lock_reason or ego_lane_cutin_lock_reason)
        info['cutin_lock_reason'] = (
            str(cutin_lock_reason or ego_lane_cutin_lock_reason)
            if (cutin_lock_reason or ego_lane_cutin_lock_reason)
            else None
        )
        hidden = out.get("hidden", None)
        if hidden is not None:
            info['latent'] = hidden[batch_idx].detach().cpu().numpy().astype(np.float32)
        info['selected_future_traj'] = selected_control_traj[:pred_len, :C_out].astype(np.float32)
        if adversarial_scores is not None:
            info['adversarial_scores'] = adversarial_scores
            info['adversarial_scores_by_target'] = adversarial_scores_by_target
        if score_selection_key == "adversarial":
            adversarial_reference_lines = self._filter_adversarial_reference_lines(ref_lines)
            if adversarial_reference_lines is not None:
                info['adversarial_reference_lines'] = adversarial_reference_lines

        return pred_traj, ref_lines, sorted_candidate_trajectories[:, :80], info

    def run_inference_batch(
            self,
            current_state,
            timestep,
            controlled_agent_ids,
            score_selections=None,
            adversarial_target_agent_ids_by_id=None,
            ego_latent=None,
            ego_future_traj=None,
            previous_agent_futures_by_id=None,
    ):
        controlled_agent_ids = [str(agent_id) for agent_id in controlled_agent_ids]
        if not controlled_agent_ids:
            return {}
        score_selections = score_selections or {}
        adversarial_target_agent_ids_by_id = adversarial_target_agent_ids_by_id or {}

        with torch.no_grad():
            pluto_features = []
            valid_ids = []
            for controlled_agent_id in controlled_agent_ids:
                try:
                    [pluto_feature] = self.dataset.process_scenario(
                        current_state,
                        timestep,
                        controlled_agent_id=controlled_agent_id,
                    )
                except Exception as exc:
                    if self._adv_refline_debug_enabled():
                        print(
                            "[AdvRefLineDebug] failed to build CBV Pluto feature",
                            {
                                "controlled_agent_id": str(controlled_agent_id),
                                "score_selection": str(score_selections.get(controlled_agent_id, "")).lower() or None,
                                "error": repr(exc),
                            },
                        )
                    continue
                if self._adv_refline_debug_enabled():
                    ref = pluto_feature.get("reference_line", {}) if isinstance(pluto_feature, dict) else {}
                    route_ids = np.asarray(ref.get("route_id", [])) if isinstance(ref, dict) else np.asarray([])
                    valid_mask = np.asarray(ref.get("valid_mask", [])) if isinstance(ref, dict) else np.asarray([])
                    selection = str(score_selections.get(controlled_agent_id, "")).lower()
                    if selection == "adversarial" or len(route_ids) == 0:
                        valid_counts = (
                            valid_mask.sum(axis=1).astype(int).tolist()
                            if valid_mask.ndim >= 2 else []
                        )
                        print(
                            "[AdvRefLineDebug] built CBV Pluto feature",
                            {
                                "controlled_agent_id": str(controlled_agent_id),
                                "score_selection": selection or None,
                                "adversarial_targets": [
                                    str(x) for x in adversarial_target_agent_ids_by_id.get(controlled_agent_id, []) or []
                                ],
                                "num_ref_lines": int(len(route_ids)),
                                "route_ids": route_ids.astype(int).tolist()[:12] if len(route_ids) else [],
                                "synthetic_ref_lines": int(np.sum((route_ids <= -1000) & (route_ids > -2000))) if len(route_ids) else 0,
                                "valid_counts": valid_counts[:12],
                                "agent_tokens_head": [str(x) for x in pluto_feature.get("agent_tokens", [])[:8]],
                            },
                        )
                pluto_features.append(pluto_feature)
                valid_ids.append(controlled_agent_id)

            if not pluto_features:
                return {}

            input_dict, out = self._forward_features(pluto_features, role="cbv")
            results = {}
            actual_futures_by_id = {}
            for agent_id, traj in (previous_agent_futures_by_id or {}).items():
                actual_futures_by_id[str(agent_id)] = traj
            metadata = current_state.get("metadata", {}) if isinstance(current_state, dict) else {}
            sdc_id = metadata.get("sdc_id", None)
            if ego_future_traj is not None:
                actual_futures_by_id["ego"] = ego_future_traj
                if sdc_id is not None:
                    actual_futures_by_id[str(sdc_id)] = ego_future_traj
            replay_exclude_ids = ["ego"]
            if sdc_id is not None:
                replay_exclude_ids.append(str(sdc_id))
            replay_futures_by_id = self._extract_replay_futures_by_id(
                current_state,
                timestep,
                exclude_agent_ids=replay_exclude_ids,
                horizon=int(self.cfg.get("replay_collision_horizon", self.cfg.get("future_steps", 80))),
            )
            for agent_id, traj in replay_futures_by_id.items():
                agent_id = str(agent_id)
                if agent_id not in actual_futures_by_id:
                    actual_futures_by_id[agent_id] = traj

            # Full matrix search via ScoreNet (both agents jointly)
            per_agent_mlp_scores = {}
            forced_best_indices_by_id = {}
            score_net_debug = os.environ.get("UNITRAJ_SCORENET_DEBUG", "0") in ("1", "true", "True")
            if score_net_debug:
                print(
                    "[MatrixSearch] input",
                    {
                        "loaded": self.score_net is not None,
                        "valid_ids": valid_ids,
                        "ego_latent_shape": None if ego_latent is None else list(np.asarray(ego_latent).shape),
                        "mode": str(self.cfg.get("cbv_score_mode", "normal")).lower(),
                    },
                )
            if self.score_net is not None and len(valid_ids) >= 2 and ego_latent is not None:
                try:
                    cands1, probs1, _, _ = self._extract_candidates(out, 0)
                    cands2, probs2, _, _ = self._extract_candidates(out, 1)
                    if cands1 is None or cands2 is None:
                        raise ValueError("Candidate extraction failed")
                    # Training records contain world-frame selected trajectories;
                    # score candidates in that identical frame at deployment.
                    cands1 = self._candidates_local_to_world(cands1, input_dict, 0)
                    cands2 = self._candidates_local_to_world(cands2, input_dict, 1)
                    mlp_w = float(self.cfg.get("score_net_mlp_weight", 0.7))
                    best_i, best_j, mlp_scores, combined_scores = self._score_full_matrix(
                        cands1, cands2, probs1, probs2, ego_latent, ego_future_traj, mlp_weight=mlp_w
                    )
                    score_selection_keys = {
                        str(score_selections.get(agent_id, "")).lower()
                        for agent_id in valid_ids[:2]
                    }
                    exact_full_matrix = (
                        str(self.cfg.get("cbv_score_mode", "normal")).lower()
                        in ("normal", "mlp", "scorenet", "score_net")
                        and "adversarial" not in score_selection_keys
                    )
                    if exact_full_matrix:
                        if str(self.cfg.get("score_net_pair_selection", "topk_sample")).lower() not in ("argmax", "max", "stress"):
                            best_i, best_j = self._select_full_matrix_pair(combined_scores)
                        forced_best_indices_by_id[valid_ids[0]] = int(best_i)
                        forced_best_indices_by_id[valid_ids[1]] = int(best_j)
                    else:
                        # Marginalize: for each candidate, best possible pairing score.
                        per_agent_mlp_scores[valid_ids[0]] = mlp_scores.max(axis=1)
                        per_agent_mlp_scores[valid_ids[1]] = mlp_scores.max(axis=0)
                    if score_net_debug:
                        chain_debug = None
                        if ego_future_traj is not None:
                            ego_world = np.asarray(ego_future_traj, dtype=np.float32)
                            horizon = min(cands1.shape[1], cands2.shape[1], len(ego_world))
                            if horizon > 0:
                                d_av1_ego = np.linalg.norm(
                                    cands1[:, :horizon, :2] - ego_world[None, :horizon, :2], axis=-1
                                ).min(axis=-1)
                                d_av2_av1 = np.linalg.norm(
                                    cands2[:, None, :horizon, :2] - cands1[None, :, :horizon, :2], axis=-1
                                ).min(axis=-1)
                                chain_mask = (d_av1_ego[:, None] <= 4.0) & (d_av2_av1.T <= 4.0)
                                if chain_mask.any():
                                    flat_order = np.argsort(-combined_scores.reshape(-1))
                                    best_chain_flat = int(np.flatnonzero(chain_mask.reshape(-1))[np.argmax(
                                        combined_scores.reshape(-1)[chain_mask.reshape(-1)]
                                    )])
                                    best_chain_i, best_chain_j = np.unravel_index(best_chain_flat, combined_scores.shape)
                                    chain_debug = {
                                        "count": int(chain_mask.sum()),
                                        "best_pair": [int(best_chain_i), int(best_chain_j)],
                                        "best_score": float(combined_scores[best_chain_i, best_chain_j]),
                                        "rank": int(np.flatnonzero(flat_order == best_chain_flat)[0]) + 1,
                                    }
                        print(
                            "[MatrixSearch] selected",
                            {
                                "candidate_shape": [int(len(cands1)), int(len(cands2))],
                                "mlp_range": [float(mlp_scores.min()), float(mlp_scores.max())],
                                "combined_range": [float(combined_scores.min()), float(combined_scores.max())],
                                "pair": [int(best_i), int(best_j)],
                                "exact_full_matrix": bool(exact_full_matrix),
                                "best_chain": chain_debug,
                            },
                        )
                except Exception as exc:
                    print(f"[MatrixSearch] fallback to per-agent: {exc}")

            for batch_idx, controlled_agent_id in enumerate(valid_ids):
                score_selection = score_selections.get(
                    controlled_agent_id,
                    "min",
                )
                extra_collision_trajs = []
                seen_traj_ids = set()
                score_selection_key = str(score_selection).lower()
                if score_selection_key == "adversarial":
                    for target_id in adversarial_target_agent_ids_by_id.get(controlled_agent_id) or []:
                        _, target_traj, _ = self._lookup_future_entry(
                            actual_futures_by_id,
                            None,
                            target_id,
                        )
                        self._append_unique_future(
                            extra_collision_trajs,
                            seen_traj_ids,
                            target_traj,
                        )
                else:
                    for other_agent_id, target_traj in actual_futures_by_id.items():
                        if self._agent_id_matches(other_agent_id, controlled_agent_id):
                            continue
                        self._append_unique_future(
                            extra_collision_trajs,
                            seen_traj_ids,
                            target_traj,
                        )
                # score_selection = 'max'
                results[controlled_agent_id] = self._postprocess_single_output(
                    current_state=current_state,
                    timestep=timestep,
                    controlled_agent_id=controlled_agent_id,
                    pluto_feature=pluto_features[batch_idx],
                    input_dict=input_dict,
                    out=out,
                    batch_idx=batch_idx,
                    score_selection=score_selection,
                    adversarial_target_agent_ids=adversarial_target_agent_ids_by_id.get(
                        controlled_agent_id
                    ),
                    forced_best_idx=forced_best_indices_by_id.get(controlled_agent_id),
                    per_candidate_mlp_scores=per_agent_mlp_scores.get(controlled_agent_id),
                    extra_collision_trajs=extra_collision_trajs,
                    actual_target_futures_by_id=dict(actual_futures_by_id),
                    use_prediction_background=False,
                )
                info = results[controlled_agent_id][3] if results.get(controlled_agent_id) else {}
                if isinstance(info, dict) and info.get("selected_future_traj", None) is not None:
                    actual_futures_by_id[str(controlled_agent_id)] = info["selected_future_traj"]
            return results

    def run_inference(
            self,
            current_state,
            timestep,
            controlled_agent_id=None,
            score_selection="max",
            adversarial_target_agent_ids=None,
            extra_collision_trajs=None,
            actual_target_futures_by_id=None,
    ):
        with torch.no_grad():
            role = "ego" if controlled_agent_id is None else "cbv"
            [pluto_feature] = self.dataset.process_scenario(
                current_state,
                timestep,
                controlled_agent_id=controlled_agent_id,
            )
            input_dict, out = self._forward_features([pluto_feature], role=role)
            result = self._postprocess_single_output(
                current_state=current_state,
                timestep=timestep,
                controlled_agent_id=controlled_agent_id,
                pluto_feature=pluto_feature,
                input_dict=input_dict,
                out=out,
                batch_idx=0,
                score_selection=score_selection,
                adversarial_target_agent_ids=adversarial_target_agent_ids,
                extra_collision_trajs=extra_collision_trajs,
                actual_target_futures_by_id=actual_target_futures_by_id,
                use_prediction_background=True,
            )
            return result