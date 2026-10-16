import os
import time

from torch.utils.data import Dataset

dataset_type = os.environ.get("AUTOSCENE_DATASET_TYPE", "tjwest").strip().lower()
if dataset_type == "nuplan":
    from unitraj.datasets.Pluto_dataset.Pluto_dataset import PlutoDataset
elif dataset_type == "tjwest":
    from unitraj.datasets.Pluto_dataset.Pluto_dataset_tjwest import PlutoDataset
else:
    raise ValueError(
        "AUTOSCENE_DATASET_TYPE must be nuplan or tjwest, got %s" % dataset_type
    )


class PlutoTestDataset(Dataset):
    """Select the nuPlan or TJ-West builder without changing the public API."""

    def __init__(self, config=None, is_validation=True):
        if config is None:
            config = {}
        self.dataset_type = dataset_type
        self._builder = PlutoDataset(config=config, is_validation=is_validation)

    def process_scenario(self, scenario, current_step, controlled_agent_id=None):
        return self.process_scenario_data(
            scenario,
            current_step,
            controlled_agent_id=controlled_agent_id,
        )

    def process_scenario_data(self, scenario, current_step, controlled_agent_id=None):
        tracks = scenario.get("tracks", {})
        requested_id = str(controlled_agent_id) if controlled_agent_id is not None else str(
            scenario.get("metadata", {}).get("sdc_id", "ego")
        )
        matching_ids = [key for key in tracks if str(key) == requested_id]
        if len(matching_ids) != 1:
            raise RuntimeError(
                "Pluto requested agent %s must resolve to exactly one scenario track; got %d"
                % (requested_id, len(matching_ids))
            )
        # Some builders silently return 'ego' for an unknown BV id. Never accept
        # that fallback and later label an ego trajectory as the requested BV.
        resolved_id = self._builder._resolve_controlled_agent_id(scenario, controlled_agent_id)
        if str(resolved_id) != requested_id:
            raise RuntimeError(
                "Pluto builder resolved requested agent %s to different track %s"
                % (requested_id, resolved_id)
            )
        preprocess_start = time.perf_counter()
        intermediate = self._builder.preprocess(
            scenario,
            current_step=current_step,
            controlled_agent_id=controlled_agent_id,
        )
        preprocess_seconds = time.perf_counter() - preprocess_start

        process_start = time.perf_counter()
        pluto_feature = self._builder.process(
            intermediate,
            current_step=current_step,
            controlled_agent_id=controlled_agent_id,
        )
        process_seconds = time.perf_counter() - process_start

        postprocess_start = time.perf_counter()
        output = self._builder.postprocess(pluto_feature)
        if len(output) != 1 or not isinstance(output[0], dict):
            raise RuntimeError("Pluto builder must return exactly one feature dictionary")
        tokens = [str(token) for token in output[0].get("agent_tokens", [])]
        if not tokens or tokens[0] != requested_id or len(tokens) != len(set(tokens)):
            raise RuntimeError(
                "Pluto feature token mapping does not match requested primary agent %s: %s"
                % (requested_id, tokens)
            )
        postprocess_seconds = time.perf_counter() - postprocess_start

        timing = dict(getattr(self._builder, "_last_feature_timing", {}))
        timing.update(
            {
                "dataset_type": self.dataset_type,
                "preprocess_ms": preprocess_seconds * 1000.0,
                "process_ms": process_seconds * 1000.0,
                "postprocess_ms": postprocess_seconds * 1000.0,
            }
        )
        self._builder._last_feature_timing = timing
        return output
