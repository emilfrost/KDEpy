#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for the FFTKDE.
"""
import numbers
from typing import Callable, Optional, Union, Literal
import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import convolve
from scipy.special import logsumexp

from KDEpy.BaseKDE import BaseKDE
from KDEpy.binning import linear_binning
from KDEpy.utils import cartesian
from KDEpy.kernel_funcs import log_gaussian, gaussian

# noinspection PyProtectedMember, PyUnresolvedReferences
from KDEpy._cutils import grid_is_sorted


def nice_log(
    x: np.ndarray, eps: float | None = None, check_negative: bool = True
) -> np.ndarray:
    """Compute the logarithm of x, handling zero and negative values."""
    if check_negative and np.any(x < 0):
        raise ValueError("Input contains negative values.")
    mask = x > 0
    log_x = np.full_like(x, fill_value=-np.inf, dtype=np.float64)
    log_x[mask] = np.log(x[mask]) if eps is None else np.log(x[mask] + eps)
    return log_x


def log_convolve(
    array: np.ndarray,
    kernel: np.ndarray,
    memory_mode: Literal["vectorized", "loop"] | int = 1000,
) -> np.ndarray:
    """Calculate the log of an N-dimensional convolution. This is done using
    the log-sum-exp trick to avoid numerical underflow/overflow. The input is
    zero-padded as needed - equivalent to 'same' mode in scipy.signal.convolve.
    Three memory modes are available: "vectorized", "loop", and an integer
    specifying the chunk size of numbers in the output to compute at once.

    The quantity computed is the log of the sum of products over all kernel
    positions:
        out[i1, i2, ..., iN] = log(
            sum_{j1, j2, ..., jN} (
                array[i1 - j1, i2 - j2, ..., iN - jN] *
                kernel[j1, j2, ..., jN]
            )
        )
    which is equivalent to:
        out[i1, i2, ..., iN] = logsumexp_{j1, j2, ..., jN} (
            log(array[i1 + j1, i2 + j2, ..., iN + jN]) +
            log(kernel[-j1, -j2, ..., -jN])
        )
    where j1, j2, ..., jN run over the kernel dimensions.
    """
    if np.any(np.mod(kernel.shape, 2) == 0):
        raise ValueError(f"Kernel dimensions must be odd. Got {kernel.shape}.")

    # Take the log of the input array and kernel.
    log_array = nice_log(array)
    log_kernel = nice_log(kernel)

    # Pad the input array with -inf values (this is log(0)). The amount of
    # padding is half the kernel size in each dimension to match 'same' mode.
    array_padded = np.pad(
        log_array,
        pad_width=[(dim // 2, dim // 2) for dim in kernel.shape],
        mode="constant",
        constant_values=-np.inf,
    )

    # Flip the kernel for convolution.
    log_kernel_flip = np.ascontiguousarray(np.flip(log_kernel))

    # Create views of all sliding windows of the input array with the same
    # shape as the kernel.
    windows = sliding_window_view(array_padded, kernel.shape)
    kernel_axes = tuple(range(-kernel.ndim, 0))

    # Compute the output using the specified memory mode.
    if memory_mode == "vectorized":
        return logsumexp(
            windows + log_kernel_flip, axis=kernel_axes
        )  # type: ignore
    elif memory_mode == "loop" or memory_mode == 1:
        out = np.zeros_like(array, dtype=np.float64)
        tmp = np.zeros_like(log_kernel_flip)
        with np.nditer(
            out, flags=["multi_index"], op_flags=[["writeonly"]]
        ) as it:
            while not it.finished:
                np.add(windows[it.multi_index], log_kernel_flip, out=tmp)
                it[0] = logsumexp(tmp)
                it.iternext()
        return out
    elif isinstance(memory_mode, int):
        chunk = min(memory_mode, array.size)
        out = np.zeros_like(array, dtype=np.float64)
        tmp = np.zeros((chunk,) + kernel.shape, dtype=np.float64)
        for start in range(0, array.size, chunk):
            end = min(start + chunk, array.size)
            idxs = np.unravel_index(np.arange(start, end), array.shape)
            np.add(windows[idxs], log_kernel_flip, out=tmp[: end - start])
            out[idxs] = logsumexp(tmp[: end - start], axis=kernel_axes)
        return out
    else:
        raise ValueError(f"Unknown memory_mode: {memory_mode}")


class FFTKDE(BaseKDE):
    r"""
    This class implements a convolution (FFT) based computation of a KDE.
    While this implementation is very fast, there are some limitations: (1) the
    bandwidth must be constant, (2) the KDE must be evaluated on an
    equidistant grid and (3) the grid must encompass every data point.
    The finer the grid, the smaller the error.

    The evaluation step is split into two phases. First the :math:`N` data
    points are binned using a linear binning routine on an equidistant grid `x`
    with :math:`n` grid points. This runs in :math:`O(N 2^d)` time.
    Then the kernel is evaluated once on :math:`\leq n` points and the result
    of the kernel evaluation and the binned data is convolved. Using the
    convolution theorem, this step runs in :math:`O(n \log n)` time.
    While :math:`N` may be millions, :math:`n` is typically 2**10. The total
    running time of the algorithm is :math:`O(N 2^d + n \log n)`.
    See references for more information.

    The implementation is reminiscent of the one found in statsmodels. However,
    unlike the statsmodels implementation every kernel is available for FFT
    computation, weighted data is available for FFT computation, and no large
    temporary arrays are created.

    Parameters
    ----------
    kernel : str or callable
        Kernel function, or string matching available options. See cls._available_kernels.keys() for choices.
    bw : float or str
        Bandwidth or bandwidth selection method. If a float is passed, it
        is the standard deviation of the kernel. If a string it passed, it
        is the bandwidth selection method, see cls._bw_methods.keys() for
        choices.
    bw_scale: float
        A scaling factor for kernel bandwidth when evaluating the log KDE.
        Increasing bw_scale is useful to increase sensitivity to the tails of
        the kernel and thereby decrease the minimum possible log probability.

    Examples
    --------
    >>> data = np.random.randn(2**10)
    >>> # (1) Automatic bw selection using Improved Sheather Jones (ISJ)
    >>> x, y = FFTKDE(bw='ISJ').fit(data).evaluate()
    >>> # (2) Explicit choice of kernel and bw (standard deviation of kernel)
    >>> x, y = FFTKDE(kernel='triweight', bw=0.5).fit(data).evaluate()
    >>> weights = data + 10
    >>> # (3) Using a grid and weights for the data
    >>> y = FFTKDE(kernel='epa', bw=0.5).fit(data, weights).evaluate(x)
    >>> # (4) If you supply your own grid, it must be equidistant
    >>> y = FFTKDE().fit(data)(np.linspace(-10, 10, num=2**12))

    References
    ----------
    - Wand, M. P., and M. C. Jones. Kernel Smoothing.
      London ; New York: Chapman and Hall/CRC, 1995. Pages 182-192.
    - Statsmodels implementation, at
      ``statsmodels.nonparametric.kde.KDEUnivariate``.

    """

    def __init__(
        self,
        kernel: Union[str, Callable] = "gaussian",
        bw: Union[float, str] = 1,
        norm: int = 2,
        bw_scale: float = 2.0
    ):
        self.norm = norm
        self.bw_scale = bw_scale
        super().__init__(kernel, bw)
        assert isinstance(self.norm, numbers.Number) and self.norm > 0

    def fit(self, data: np.ndarray, weights: Optional[np.ndarray] = None) -> "FFTKDE":
        """
        Fit the KDE to the data. This validates the data and stores it.
        Computations are performed upon evaluation on a specific grid.

        Parameters
        ----------
        data: array-like
            The data points.
        weights: array-like
            One weight per data point. Must have same shape as the data.

        Returns
        -------
        self
            Returns the instance.

        Examples
        --------
        >>> data = [1, 3, 4, 7]
        >>> weights = [3, 4, 2, 1]
        >>> kde = FFTKDE().fit(data, weights=None)
        >>> kde = FFTKDE().fit(data, weights=weights)
        >>> x, y = kde.evaluate()
        """

        # Sets self.data
        super().fit(data, weights)
        return self

    def evaluate(self, grid_points: Optional[Union[np.ndarray, int, tuple]] = None) -> Union[np.ndarray, tuple]:
        """
        Evaluate on equidistant grid points.

        Parameters
        ----------
        grid_points: array-like, int, tuple or None
            A grid (mesh) to evaluate on. High dimensional grids must have
            shape (obs, dims). If an integer is passed, it's the number of grid
            points on an equidistant grid. If a tuple is passed, it's the
            number of grid points in each dimension. If None, a grid will be
            automatically created.

        Returns
        -------
        y: array-like
            If a grid is supplied, `y` is returned. If no grid is supplied,
            a tuple (`x`, `y`) is returned.

        Examples
        --------
        >>> kde = FFTKDE().fit([1, 3, 4, 7])
        >>> # Three ways to evaluate a fitted KDE object:
        >>> x, y = kde.evaluate()  # (1) Auto grid
        >>> x, y = kde.evaluate(256)  # (2) Auto grid with 256 points
        >>> # (3) Use a custom grid (make sure it's wider than the data)
        >>> x_grid = np.linspace(-10, 25, num=2**10)  # <- Must be equidistant
        >>> y = kde.evaluate(x_grid)  # Notice that only y is returned
        """
        if self.kernel is log_gaussian:
            raise TypeError(
                "Use the `evaluate_log` method instead with kernel='gaussian'."
            )
        # This method sets self.grid_points and verifies it
        super().evaluate(grid_points)

        # Extra verification for FFTKDE (checking the sorting property)
        if not grid_is_sorted(self.grid_points):
            raise ValueError("The grid must be sorted.")

        if isinstance(self.bw, numbers.Number) and self.bw > 0:
            bw = self.bw
        else:
            raise ValueError("The bw must be a number.")
        self.bw = bw

        # Step 0 - Make sure data points are inside of the grid
        min_grid = np.min(self.grid_points, axis=0)
        max_grid = np.max(self.grid_points, axis=0)

        min_data = np.min(self.data, axis=0)
        max_data = np.max(self.data, axis=0)
        if not ((min_grid < min_data).all() and (max_grid > max_data).all()):
            raise ValueError("Every data point must be inside of the grid.")

        # Step 1 - Obtaining the grid counts
        # TODO: Consider moving this to the fitting phase instead
        data = linear_binning(self.data, grid_points=self.grid_points, weights=self.weights)

        # Step 2 - Computing kernel weights
        g_shape = self.grid_points.shape[1]
        num_grid_points = np.array(list(len(np.unique(self.grid_points[:, i])) for i in range(g_shape)))

        num_intervals = num_grid_points - 1
        dx = (max_grid - min_grid) / num_intervals

        # Find the real bandwidth, the support times the desired bw factor
        if self.kernel.finite_support:
            real_bw = self.kernel.support * self.bw
        else:
            # The parent class should compute this already. If not, compute
            # it again. This optimization only dominates a little bit with
            # few data points
            try:
                real_bw = self._kernel_practical_support
            except AttributeError:
                real_bw = self.kernel.practical_support(self.bw)

        # Compute L, the number of dx'es to move out from 0 in kernel
        L = np.minimum(np.floor(real_bw / dx), num_intervals + 1)
        assert (dx * L <= real_bw).all()

        # Evaluate the kernel once
        grids = [np.linspace(-dx * L, dx * L, int(L * 2 + 1)) for (dx, L) in zip(dx, L)]
        kernel_grid = cartesian(grids)
        kernel_weights = self.kernel(kernel_grid, bw=self.bw, norm=self.norm)

        # Reshape in preparation to
        kernel_weights = kernel_weights.reshape(*[int(k * 2 + 1) for k in L])
        data = data.reshape(*tuple(num_grid_points))

        # Step 3 - Performing the convolution

        # The following code block surpressed the warning:
        #        anaconda3/lib/python3.6/site-packages/mkl_fft/_numpy_fft.py:
        #            FutureWarning: Using a non-tuple sequence for multidimensional ...
        #        output = mkl_fft.rfftn_numpy(a, s, axes)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ans = convolve(data, kernel_weights, mode="same").reshape(-1, 1)

        return self._evalate_return_logic(ans, self.grid_points)

    def evaluate_log(self, grid_points: Optional[Union[np.ndarray, int, tuple]] = None) -> Union[np.ndarray, tuple]:
        """
        Evaluate the logarithm of the KDE on grid points.

        Parameters
        ----------
        grid_points: array-like, int, tuple or None
            A grid (mesh) to evaluate on. High dimensional grids must have
            shape (obs, dims). If an integer is passed, it's the number of grid
            points on an equidistant grid. If a tuple is passed, it's the
            number of grid points in each dimension. If None, a grid will be
            automatically created.

        Returns
        -------
        y: array-like
            If a grid is supplied, `y` is returned. If no grid is supplied,
            a tuple (`x`, `y`) is returned.

        Examples
        --------
        >>> import numpy as np
        >>> from KDEpy import FFTKDE
        >>>
        >>> data = np.random.normal(size=(100, 3))
        >>> kde = FFTKDE(kernel='gaussian').fit(data)
        >>> x, y = kde.evaluate()
        >>> x_log, y_log = kde.evaluate_log()
        >>> # They give the same grid points
        >>> np.allclose(x, x_log)
        True
        >>> # They give a slighty different result when converting back from
        >>> # log probability due to a different kernel grid.
        >>> np.abs(y - np.exp(y_log)).max()
        2.5677168889386314e-06
        >>> # Using evaluate_log directly is more stable for log probability as
        >>> # the minimum value is lower than the float limit.
        >>> np.log(y).min(), y_log.min()
        (-36.044639298866954, -45.28706017927512)
        >>> # Double precision float limit for log probability
        >>> np.log(np.finfo(np.float64).eps)
        -36.04365338911715
        """
        # This method sets self.grid_points and verifies it
        super().evaluate(grid_points)

        # Extra verification for FFTKDE (checking the sorting property)
        if not grid_is_sorted(self.grid_points):
            raise ValueError("The grid must be sorted.")

        if isinstance(self.bw, numbers.Number) and self.bw > 0:
            bw = self.bw
        else:
            raise ValueError("The bw must be a number.")
        self.bw = bw

        # Step 0 - Make sure data points are inside of the grid
        min_grid = np.min(self.grid_points, axis=0)
        max_grid = np.max(self.grid_points, axis=0)

        min_data = np.min(self.data, axis=0)
        max_data = np.max(self.data, axis=0)
        if not ((min_grid < min_data).all() and (max_grid > max_data).all()):
            raise ValueError("Every data point must be inside of the grid.")

        # Step 1 - Obtaining the grid counts
        # TODO: Consider moving this to the fitting phase instead
        data = linear_binning(self.data, grid_points=self.grid_points, weights=self.weights)

        # Step 2 - Computing kernel weights
        g_shape = self.grid_points.shape[1]
        num_grid_points = np.array(list(len(np.unique(self.grid_points[:, i])) for i in range(g_shape)))

        num_intervals = num_grid_points - 1
        dx = (max_grid - min_grid) / num_intervals

        # Find the real bandwidth, the support times the desired bw factor
        if self.kernel.finite_support:
            real_bw = self.kernel.support * self.bw
        else:
            # The parent class should compute this already. If not, compute
            # it again. This optimization only dominates a little bit with
            # few data points
            try:
                real_bw = self._kernel_practical_support
            except AttributeError:
                real_bw = self.kernel.practical_support(self.bw)

        # For log evaluation, increase the bandwidth to be more sensitive to
        # the tails of the kernel
        real_bw *= self.bw_scale

        # Compute L, the number of dx'es to move out from 0 in kernel
        L = np.minimum(np.floor(real_bw / dx), num_intervals + 1)
        assert (dx * L <= real_bw).all()

        # Evaluate the kernel once
        grids = [np.linspace(-dx * L, dx * L, int(L * 2 + 1)) for (dx, L) in zip(dx, L)]
        kernel_grid = cartesian(grids)
        kernel_weights = self.kernel(kernel_grid, bw=self.bw, norm=self.norm)

        # Reshape in preparation to
        kernel_weights = kernel_weights.reshape(*[int(k * 2 + 1) for k in L])
        data = data.reshape(*tuple(num_grid_points))

        # Step 3 - Performing the convolution
        ans = log_convolve(data, kernel_weights).reshape(-1, 1)

        return self._evalate_return_logic(ans, self.grid_points)


if __name__ == "__main__":
    import pytest

    # --durations=10  <- May be used to show potentially slow tests
    pytest.main(args=[".", "--doctest-modules", "-v", "--capture=sys"])
