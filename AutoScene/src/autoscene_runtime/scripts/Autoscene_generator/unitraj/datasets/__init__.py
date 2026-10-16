from unitraj.datasets.VBD_dataset.VBD_dataset import VBDDataset
from unitraj.datasets.Pluto_dataset.Pluto_dataset import PlutoDataset

__all__ = {
    'VBD': VBDDataset,
    'Pluto': PlutoDataset,
}


def build_dataset(config, val=False):
    dataset = __all__[config.method.model_name](
        config=config, is_validation=val
    )
    return dataset
