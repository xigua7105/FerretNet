import torch
import torch.nn as nn
import torch.nn.functional as F


class MinValues(nn.Module):
    def __init__(
            self,
            kernel_size: int,
            stride: int = None,
            padding: int = None,
    ):
        super().__init__()
        self.k = kernel_size
        self.s = stride if stride is not None else 1
        self.p = padding if padding is not None else kernel_size//2

    def forward(self, x):
        N, C, H, W = x.shape
        center_idx = (self.k*self.k) // 2

        # padding
        input_padded = F.pad(x, (self.p, self.p, self.p, self.p), mode='constant', value=float('inf'))

        # mask center pixel
        unfolded = F.unfold(input_padded, kernel_size=(self.k, self.k), stride=self.s)
        unfolded = unfolded.view(N, C, self.k*self.k, -1)  # (N, C, k*k, n_patches)
        unfolded[:, :, center_idx, :] = float('inf')

        min_vals = unfolded.min(dim=2).values

        # (N, C, H_out, W_out)
        H_out = (H + 2 * self.p - self.k) // self.s + 1
        W_out = (W + 2 * self.p - self.k) // self.s + 1
        return min_vals.view(N, C, H_out, W_out)


class MaxValues(nn.Module):
    def __init__(
            self,
            kernel_size: int,
            stride: int = None,
            padding: int = None,
    ):
        super().__init__()
        self.k = kernel_size
        self.s = stride if stride is not None else 1
        self.p = padding if padding is not None else kernel_size//2

    def forward(self, x):
        N, C, H, W = x.shape
        center_idx = (self.k * self.k) // 2

        # padding
        input_padded = F.pad(x, (self.p, self.p, self.p, self.p), mode='constant', value=float('-inf'))

        # mask center pixel
        unfolded = F.unfold(input_padded, kernel_size=(self.k, self.k), stride=self.s)
        unfolded = unfolded.view(N, C, self.k*self.k, -1)  # (N, C, k*k, n_patches)
        unfolded[:, :, center_idx, :] = float('-inf')

        max_vals = unfolded.max(dim=2).values

        # (N, C, H_out, W_out)
        H_out = (H + 2 * self.p - self.k) // self.s + 1
        W_out = (W + 2 * self.p - self.k) // self.s + 1
        return max_vals.view(N, C, H_out, W_out)


class AverageValues(nn.Module):
    def __init__(
            self,
            kernel_size: int,
            stride: int = None,
            padding: int = None,
    ):
        super().__init__()
        self.k = kernel_size
        self.s = stride if stride is not None else 1
        self.p = padding if padding is not None else kernel_size//2

    def forward(self, x):
        N, C, H, W = x.shape
        center_idx = (self.k*self.k) // 2

        # padding
        input_padded = F.pad(x, (self.p, self.p, self.p, self.p), mode='constant', value=float(0.0))

        # mask center pixel
        unfolded = F.unfold(input_padded, kernel_size=(self.k, self.k), stride=self.s)
        unfolded = unfolded.view(N, C, self.k*self.k, -1)  # (N, C, k*k, n_patches)
        unfolded[:, :, center_idx, :] = float(0.0)

        avg_vals = unfolded.mean(dim=2)

        # (N, C, H_out, W_out)
        H_out = (H + 2 * self.p - self.k) // self.s + 1
        W_out = (W + 2 * self.p - self.k) // self.s + 1
        return avg_vals.view(N, C, H_out, W_out)


class ExclusionMedianValues(nn.Module):
    def __init__(
            self,
            kernel_size: int,
            stride: int = None,
            padding: int = None,
    ):
        super().__init__()
        self.k = kernel_size
        self.s = stride if stride is not None else 1
        self.p = padding if padding is not None else kernel_size//2

    def forward(self, x):
        N, C, H, W = x.shape
        center_idx = (self.k * self.k) // 2

        # padding
        input_padded = F.pad(x, (self.p, self.p, self.p, self.p), mode='constant', value=float(0.0))

        unfolded = F.unfold(input_padded, kernel_size=(self.k , self.k ), stride=self.s)
        unfolded = unfolded.view(N, C, self.k *self.k , -1)  # (N, C, k*k, n_patches)

        # exclude the center pixel
        mask = torch.arange(self.k *self.k ) != center_idx
        unfolded = unfolded[:, :, mask, :]  # (N, C, k*k-1, n_patches)

        exclusion_median_vals = unfolded.median(dim=2).values

        # (N, C, H_out, W_out)
        H_out = (H + 2 * self.p - self.k ) // self.s + 1
        W_out = (W + 2 * self.p - self.k ) // self.s + 1
        return exclusion_median_vals.view(N, C, H_out, W_out)


class MaskMedianValues(nn.Module):
    def __init__(
            self,
            kernel_size: int,
            stride: int = None,
            padding: int = None,
    ):
        super().__init__()
        self.k = kernel_size
        self.s = stride if stride is not None else 1
        self.p = padding if padding is not None else kernel_size//2

    def forward(self, x):
        N, C, H, W = x.shape
        center_idx = (self.k*self.k) // 2

        # padding
        input_padded = F.pad(x, (self.p, self.p, self.p, self.p), mode='constant', value=float(0.0))

        # mask center pixel
        unfolded = F.unfold(input_padded, kernel_size=(self.k, self.k), stride=self.s)
        unfolded = unfolded.view(N, C, self.k*self.k, -1)  # (N, C, k*k, n_patches)
        unfolded[:, :, center_idx, :] = float(0.0)

        median_vals = unfolded.median(dim=2).values

        # (N, C, H_out, W_out)
        H_out = (H + 2 * self.p - self.k) // self.s + 1
        W_out = (W + 2 * self.p - self.k) // self.s + 1
        return median_vals.view(N, C, H_out, W_out)


class RetentionMedianValues(nn.Module):
    def __init__(
            self,
            kernel_size: int,
            stride: int = None,
            padding: int = None,
    ):
        super().__init__()
        self.k = kernel_size
        self.s = stride if stride is not None else 1
        self.p = padding if padding is not None else kernel_size//2

    def forward(self, x):
        N, C, H, W = x.shape

        # padding
        input_padded = F.pad(x, (self.p, self.p, self.p, self.p), mode='constant', value=float(0.0))

        unfolded = F.unfold(input_padded, kernel_size=(self.k, self.k), stride=self.s)
        unfolded = unfolded.view(N, C, self.k * self.k, -1)  # (N, C, k*k, n_patches)

        retention_median_vals = unfolded.median(dim=2).values

        H_out = (H + 2 * self.p - self.k) // self.s + 1
        W_out = (W + 2 * self.p - self.k) // self.s + 1
        return retention_median_vals.view(N, C, H_out, W_out)


class Origin(nn.Module):
    def __init__(
            self,
            kernel_size: int,
            stride: int = None,
            padding: int = None,
    ):
        super().__init__()

    def forward(self, _):
        # do nothing
        return 0.0


def get_lpd_dict():
    return {
        "max": MaxValues,
        "min": MinValues,
        "avg": AverageValues,
        "median": MaskMedianValues,
        "excmed": ExclusionMedianValues,
        "retmed": RetentionMedianValues,
        "origin": Origin,
    }


if __name__ == "__main__":
    x = torch.randn(64, 3, 224, 224).cuda()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    x1 = MaxValues(3)(x)
    end.record()
    torch.cuda.synchronize()
    print(f"max time consumed: {start.elapsed_time(end):.4f} ms")

    start.record()
    x2 = MinValues(3)(x)
    end.record()
    torch.cuda.synchronize()
    print(f"min time consumed: {start.elapsed_time(end):.4f} ms")

    start.record()
    x3 = AverageValues(3)(x)
    end.record()
    torch.cuda.synchronize()
    print(f"avg time consumed: {start.elapsed_time(end):.4f} ms")

    start.record()
    x4 = MaskMedianValues(3)(x)
    end.record()
    torch.cuda.synchronize()
    print(f"mask time consumed: {start.elapsed_time(end):.4f} ms")

    start.record()
    x5 = ExclusionMedianValues(3)(x)
    end.record()
    torch.cuda.synchronize()
    print(f"exclusion time consumed: {start.elapsed_time(end):.4f} ms")

    start.record()
    x6 = RetentionMedianValues(3)(x)
    end.record()
    torch.cuda.synchronize()
    print(f"retention time consumed: {start.elapsed_time(end):.4f} ms")
