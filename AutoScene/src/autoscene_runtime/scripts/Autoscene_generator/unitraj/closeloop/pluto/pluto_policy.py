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

from unitraj.closeloop.pluto.pluto_inference import PlutoInference


class PlutoPolicy(InferencePolicy):
    """按 AutoScene 主配置选择 nuPlan/TJ-West 闭环策略。"""

    def __init__(self, obj, seed):
        super(PlutoPolicy, self).__init__(obj, seed)
        # Generic nuPlan InferencePolicy already initializes self.Sim.  TJ-West
        # intentionally leaves it unset, so only create Pluto there.
        if getattr(self, "Sim", None) is None:
            self.Sim = PlutoInference(self.ego_cfg)
            self.Sim.initialize_model()