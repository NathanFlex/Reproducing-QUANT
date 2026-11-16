import numpy as np
import torch
import torch.nn.functional as F

# == Skewness ===============================================================
def rolling_skew(X, window=5):
    pad = window // 2
    padded = F.pad(X, (pad, pad), "replicate")

    mean = F.avg_pool1d(padded, window, stride=1)
    var = torch.clamp(F.avg_pool1d(padded ** 2, window, stride=1) - mean ** 2, min=1e-6)
    std = torch.sqrt(var)

    # 3rd central moment: E[(X - mu)^3]
    m3 = F.avg_pool1d(padded ** 3, window, stride=1) - 3 * mean * F.avg_pool1d(padded ** 2, window,
                                                                               stride=1) + 2 * mean ** 3
    skew = m3 / (std ** 3 + 1e-6)
    return skew

# == ZCR =====================================================================

def rolling_zcr(X, window=5):
    pad = window // 2
    signs = torch.sign(X)
    # Detect where sign changes (-1 to 1 or 1 to -1)
    sign_diff = torch.abs(signs[..., 1:] - signs[..., :-1]) / 2.0
    padded_diff = F.pad(sign_diff, (pad, window - pad - 1), "replicate")

    # Average frequency of sign crossings over the rolling window
    zcr = F.avg_pool1d(padded_diff, window, stride=1)
    return zcr

def make_intervals_with_depth(input_length, depth):
    exponent = min(depth, int(np.log2(input_length)) + 1)

    intervals = []

    for depth_idx, n in enumerate(2 ** torch.arange(exponent), start=1):
        indices = torch.linspace(0, input_length, n + 1).long()
        intervals_n = torch.stack((indices[:-1], indices[1:]), 1)

        for a, b in intervals_n:
            intervals.append((int(a), int(b), depth_idx))

        if n > 1 and intervals_n.diff().median() > 1:
            shift = int(np.ceil(input_length / n / 2))
            shifted = intervals_n[:-1] + shift
            for a, b in shifted:
                intervals.append((int(a), int(b), depth_idx))

    return intervals


def f_quantile(X, div=4):
    n = X.shape[-1]

    if n == 1:
        return X.view(X.shape[0], 1, X.shape[1] * X.shape[2])

    num_quantiles = 1 + (n - 1) // div

    if num_quantiles == 1:
        quantiles = X.quantile(
            torch.tensor([0.5], device=X.device),
            dim=-1
        ).permute(1, 2, 0)

        return quantiles.view(
            quantiles.shape[0], 1,
            quantiles.shape[1] * quantiles.shape[2]
        )

    quantiles = X.quantile(
        torch.linspace(0, 1, num_quantiles, device=X.device),
        dim=-1
    ).permute(1, 2, 0)

    quantiles[..., 1::2] = (
        quantiles[..., 1::2] - X.mean(-1, keepdims=True)
    )

    return quantiles.view(
        quantiles.shape[0], 1,
        quantiles.shape[1] * quantiles.shape[2]
    )


class ScaleAwareIntervalModel:
    """
    Uses a depth-dependent quantile density.

    Default schedule for depth=6:
        depth 1 -> div 4
        depth 2 -> div 8
        depth 3 -> div 8
        depth 4 -> div 8
        depth 5 -> div 16
        depth 6 -> div 16

    This gives more quantile resolution to coarse intervals and less
    resolution to the finest intervals.

    The reciprocal-divisor average is approximately equal to uniform div=8:
        (1/4 + 1/8 + 1/8 + 1/8 + 1/16 + 1/16) / 6 = 1/8
    so the feature budget is designed to be broadly comparable.
    """

    def __init__(self, input_length, depth=6, div_schedule=None):
        assert depth >= 1

        self.depth = depth
        self.intervals = make_intervals_with_depth(input_length, depth)

        if div_schedule is None:
            base = [4, 8, 8, 8, 16, 16]
            if depth <= len(base):
                div_schedule = base[:depth]
            else:
                div_schedule = base + [16] * (depth - len(base))

        assert len(div_schedule) >= depth
        assert all(d >= 1 for d in div_schedule)

        self.div_schedule = div_schedule

    def fit(self, X, Y):
        pass

    def transform(self, X):
        features = []

        for a, b, depth_idx in self.intervals:
            div = self.div_schedule[depth_idx - 1]
            features.append(
                f_quantile(X[..., a:b], div=div).squeeze(1)
            )

        return torch.cat(features, -1)

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.transform(X)


class ScaleAwareQuant:
    def __init__(self, depth=6, div_schedule=None):
        assert depth >= 1

        self.depth = depth
        self.div_schedule = div_schedule

        self.representation_functions = (
            lambda X: X,
            lambda X: F.avg_pool1d(
                F.pad(X.diff(), (2, 2), "replicate"), 5, 1
            ),
            lambda X: X.diff(n=2),
            lambda X: torch.fft.rfft(X).abs(),
            lambda X: rolling_skew(X, window=5),
            lambda X: rolling_zcr(X, window=5),
        )

        self.models = {}
        self.fitted = False

    def transform(self, X):
        assert self.fitted, "not fitted"

        features = []

        for index, function in enumerate(self.representation_functions):
            Z = function(X)
            features.append(self.models[index].transform(Z))

        return torch.cat(features, -1)

    def fit_transform(self, X, Y):
        features = []

        for index, function in enumerate(self.representation_functions):
            Z = function(X)

            self.models[index] = ScaleAwareIntervalModel(
                input_length=Z.shape[-1],
                depth=self.depth,
                div_schedule=self.div_schedule,
            )

            features.append(
                self.models[index].fit_transform(Z, Y)
            )

        self.fitted = True
        return torch.cat(features, -1)
