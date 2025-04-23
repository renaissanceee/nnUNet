import torch

def check_cropped_outside_is_zero(tensor, bbox):
    mask = torch.zeros_like(tensor, dtype=bool)  # 创建一个全为 False 的掩码
    slices = tuple(slice(low, high) for low, high in bbox)  # 在 bbox内置为 True
    mask[slices] = True
    outside_region = tensor[~mask]  # 检查 bbox 外是否为 0
    return torch.all(outside_region == 0)


def find_nonzero_range(tensor: torch.Tensor, cropped_z: int = 100, cropped_size: int = 190) -> torch.Tensor:
    """
    Crop a [D, H, W] tensor around the first detected non-zero column and row in the [H, W] dimensions.

    Args:
        tensor (torch.Tensor): Input tensor with shape [D, H, W]->[D, cropped_size, cropped_size]
    """
    D, H, W = tensor.shape

    # Find non-zero columns and rows
    col_has_value = (tensor != 0).any(dim=0).any(dim=0)  # [W]
    row_has_value = (tensor != 0).any(dim=0).any(dim=1)  # [H]
    z_has_value = (tensor != 0).any(dim=1).any(dim=1)  # [D]

    # Get first non-zero indices
    first_nonzero_col = torch.where(col_has_value)[0][0].item()
    first_nonzero_row = torch.where(row_has_value)[0][0].item()
    first_nonzero_z = torch.where(z_has_value)[0][0].item()  # z_min

    # Prevent out-of-bounds crop
    first_nonzero_z = D - cropped_z if first_nonzero_z + cropped_z > D else first_nonzero_z
    first_nonzero_col = W - cropped_size if first_nonzero_col + cropped_size > W else first_nonzero_col
    first_nonzero_row = H - cropped_size if first_nonzero_row + cropped_size > H else first_nonzero_row

    if cropped_z == 155:
        first_nonzero_z = 0
    return first_nonzero_z, first_nonzero_col, first_nonzero_row  # W,H


def downsample_3d_tensor(tensor, sampling_factor=2, random_seed=None):
    """
    对3D张量进行多维度降采样 (带随机种子控制)
    Args:
        tensor: 输入张量，形状为 [D, H, W] (此处为 [155, 240, 240])
        sampling_factor: 降采样因子 (默认2)
        random_seed: 随机种子 (None表示不固定)
    Returns:
        降采样后的张量，形状为 [D//f, H//f, W//f]
    """
    assert len(tensor.shape) == 3, "input must be [z,h,w]"
    D, H, W = tensor.shape
    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        import random
        random.seed(random_seed)

    # z-slices: equal-interval
    dim0_indices = torch.linspace(0, D - 1, D // sampling_factor, dtype=torch.long)
    # H,W: 2-step-sampling
    dim1_indices = torch.randint(0, H, (H // sampling_factor,)).sort().values
    dim2_indices = torch.randint(0, W, (W // sampling_factor,)).sort().values
    sampled_tensor = tensor[dim0_indices[:, None, None],
    dim1_indices[None, :, None],
    dim2_indices[None, None, :]]

    return sampled_tensor