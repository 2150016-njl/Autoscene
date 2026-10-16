import os


dataset_type = os.environ.get("AUTOSCENE_DATASET_TYPE", "tjwest").strip().lower()
if dataset_type == "nuplan":
    from unitraj.closeloop.inference_policy import InferencePolicy
elif dataset_type == "tjwest":
    from unitraj.closeloop.inference_policy_for_tjwest import InferencePolicy
else:
    raise ValueError(
        "AUTOSCENE_DATASET_TYPE must be nuplan or tjwest, got %s" % dataset_type
    )


class PlutoHybridPolicy(InferencePolicy):
    """Dataset-aware wrapper for the Pluto + OSM mixed-candidate planner.

    All warmup/history/control/adversarial/platform behavior stays in the current
    dataset-specific InferencePolicy. The only planner difference is selected by
    ``planner_variant=pluto_hybrid`` in the inference-engine factory.
    """

    def __init__(self, obj, seed):
        super(PlutoHybridPolicy, self).__init__(obj, seed)

        self.replan_frequency = int(self.base_cfg.get("replan_frequency", 5))
        if self.replan_frequency <= 0:
            raise ValueError("replan_frequency must be > 0")

        # TJ-West parent deliberately leaves ego Sim unset; nuPlan initializes it
        # in the parent. Do not reload an already initialized model.
        if getattr(self, "Sim", None) is None:
            self.Sim = self._init_inference_engine(self.ego_cfg)
