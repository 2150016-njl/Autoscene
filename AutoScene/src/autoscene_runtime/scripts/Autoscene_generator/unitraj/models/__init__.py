from unitraj.models.vbd.model.vbd import VBD
from unitraj.models.pluto.pluto_model import PlanningModel

__all__ = {
    'VBD': VBD,
    'Pluto': PlanningModel,
}


def build_model(config):
    model = __all__[config.model_name](
        config=config
    )

    return model
