import torch

class GridCache:
    """
    A singleton-like cache for reusable grid tensors to avoid re-allocation and re-computation
    during training/inference steps.
    """
    _cache = {}

    @staticmethod
    def get_mesh_grid(height: int, width: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Returns a meshgrid of shape (height, width, 2) where the last dimension is (x, y).
        The grid covers [0, 1] for both dimensions.
        """
        key = ("mesh_01", height, width, device, dtype)
        if key not in GridCache._cache:
            y_range = torch.linspace(0, 1, height, device=device, dtype=dtype)
            x_range = torch.linspace(0, 1, width, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
            grid = torch.stack([grid_x, grid_y], dim=-1) # (H, W, 2) -> (x, y)
            GridCache._cache[key] = grid
        return GridCache._cache[key]

    @staticmethod
    def get_uv_grid(width: int, height: int, aspect_ratio: float, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Returns a UV grid of shape (height, width, 2) where the last dimension is (u, v) or (x, y).
        Coordinates are centered and scaled by aspect ratio.
        """
        # Derive aspect ratio if not explicitly provided
        if aspect_ratio is None:
            aspect_ratio = float(width) / float(height)

        key = ("uv_grid", width, height, aspect_ratio, device, dtype)
        if key not in GridCache._cache:
             # Compute normalized spans for X and Y
            diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
            span_x = aspect_ratio / diag_factor
            span_y = 1.0 / diag_factor

            # Establish the linspace boundaries
            left_x = -span_x * (width - 1) / width
            right_x = span_x * (width - 1) / width
            top_y = -span_y * (height - 1) / height
            bottom_y = span_y * (height - 1) / height

            # Generate 1D coordinates
            x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
            y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

            # Create 2D meshgrid (width x height) and stack into UV
            uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
            uv_grid = torch.stack((uu, vv), dim=-1)
            GridCache._cache[key] = uv_grid

        return GridCache._cache[key]

    @staticmethod
    def clear():
        GridCache._cache.clear()
