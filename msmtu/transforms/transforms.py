import torch.nn as nn
import torch

from .utils import get_tasseled_cap, adapt_masks


class NormalizeTS(nn.Module):
    """
    Class to normalize our time series
    """

    def __init__(
            self,
            x_min: float,
            x_max: float,
            feature_range: tuple = (0, 1),
    ):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max

        assert feature_range[0] < feature_range[1]

        self.feature_range = feature_range

    def forward(self, data):
        ts_data = data['ts_data']
        for direction in ['forward', 'backward']:
            ts_data_direction = ts_data[direction]

            for stage in ['values', 'evals']:
                ts_tensor = torch.as_tensor(ts_data_direction[stage])
                x_std = ((ts_tensor - self.x_min) / (self.x_max - self.x_min))
                x_scaled = ((x_std * (self.feature_range[1] - self.feature_range[0])) + self.feature_range[0]).tolist()
                ts_data_direction[stage] = x_scaled

            ts_data[direction] = ts_data_direction

        data['ts_data'] = ts_data

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(feature_range={self.feature_range}, x_min={self.x_min}, x_max={self.x_max})"


class StandardizeTS(nn.Module):
    """
    Class to standardize our time series
    """

    def __init__(
            self,
            mean,
            std,
            dict_ancillary: dict = None,
    ):
        super().__init__()
        self.mean = mean
        self.std = std

        self.dict_ancillary = dict_ancillary

    def forward(self, rec):
        for direction in ['forward', 'backward']:
            ts_data_direction = rec[direction]

            for stage in ['values', 'evals']:
                ts_tensor = torch.as_tensor(ts_data_direction[stage])
                normalization = ((ts_tensor - self.mean) / self.std).tolist()  # * masks_tensor
                ts_data_direction[stage] = normalization

            rec[direction] = ts_data_direction

        if self.dict_ancillary is not None:
            for key, mean_std in self.dict_ancillary.items():
                rec[key] = (rec[key] - mean_std[0]) / mean_std[1]

        return rec

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, " \
               f"dict_ancillary={self.dict_ancillary})"


class TasseledCap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        ts_data = data['ts_data']
        for direction in ['forward', 'backward']:
            ts_data_direction = ts_data[direction]

            for stage, mask in [('values', 'masks'), ('evals', 'eval_masks')]:
                ts_tensor = torch.as_tensor(ts_data_direction[stage])
                # Tasseled cap transformation
                ts_data_direction[stage] = get_tasseled_cap(ts_tensor).tolist()

                # Adapt masks
                if mask == 'masks':
                    masks = torch.as_tensor(ts_data_direction[mask])
                    ts_data_direction[mask] = adapt_masks(masks).tolist()

            ts_data[direction] = ts_data_direction

        data['ts_data'] = ts_data

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(Tasseled Cap transformation)"
