import torch


def get_tasseled_cap(tensor_seq: torch.Tensor) -> torch.Tensor:

    red = tensor_seq[:, 0]
    nir1 = tensor_seq[:, 1]
    blue = tensor_seq[:, 2]
    green = tensor_seq[:, 3]
    nir2 = tensor_seq[:, 4]
    swir1 = tensor_seq[:, 5]
    swir2 = tensor_seq[:, 6]

    brightness = 0.4395 * red + 0.5945 * nir1 + 0.2460 * blue + 0.3918 * green + 0.3506 * nir2 + 0.2136 * swir1 \
                 + 0.2678 * swir2
    greenness = -0.4064 * red + 0.5129 * nir1 - 0.2744 * blue - 0.2893 * green + 0.4882 * nir2 - 0.0036 * swir1 \
                - 0.4169 * swir2
    wetness = 0.1147 * red + 0.2489 * nir1 + 0.2408 * blue + 0.3132 * green - 0.3122 * nir2 - 0.6416 * swir1 \
              - 0.5087 * swir2

    output = torch.cat((brightness.unsqueeze(1), greenness.unsqueeze(1), wetness.unsqueeze(1)), axis=1)

    return output


def adapt_masks(masks: torch.Tensor) -> torch.Tensor:
    conditions = (masks == 0).sum(axis=1) > 3

    new_mask = torch.ones(12, 3)

    for idx, cond in enumerate(conditions):
        if cond:
            new_mask[idx] = torch.zeros(3)

    return new_mask








