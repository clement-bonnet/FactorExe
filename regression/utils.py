import numpy as np


def flatten(data: dict) -> dict:
    flattened_data = {}
    for key, value in data.items():
        value = np.squeeze(value)
        if value.ndim > 0:
            assert value.ndim == 1
            for idx in range(value.shape[0]):
                flattened_data[f"{key}_{idx}"] = value[idx]
        else:
            flattened_data[key] = value
    return flattened_data
