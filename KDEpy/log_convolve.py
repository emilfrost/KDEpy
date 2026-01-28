from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import logsumexp


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
    kernel: np.ndarray | None = None,
    *,
    log_kernel: np.ndarray | None = None,
    memory_mode: Literal["vectorized", "loop"] | int = 1000,
) -> np.ndarray:
    """Calculate the log of an N-dimensional convolution. This is done using
    the log-sum-exp trick to avoid numerical underflow. The input is
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
    if kernel is None:
        if log_kernel is None:
            raise ValueError("Either kernel or log_kernel must be provided.")
    else:
        if log_kernel is not None:
            raise ValueError(
                "Only one of kernel or log_kernel may be provided."
            )
        log_kernel = nice_log(kernel)

    if np.any(np.mod(log_kernel, 2) == 0):
        raise ValueError(
            f"Kernel dimensions must be odd. Got {log_kernel.shape}."
        )

    log_array = nice_log(array)

    # Pad the input array with -inf values (this is log(0)). The amount of
    # padding is half the kernel size in each dimension to match 'same' mode.
    array_padded = np.pad(
        log_array,
        pad_width=[(dim // 2, dim // 2) for dim in log_kernel.shape],
        mode="constant",
        constant_values=-np.inf,
    )

    # Flip the kernel for convolution.
    log_kernel_flip = np.ascontiguousarray(np.flip(log_kernel))

    # Create views of all sliding windows of the input array with the same
    # shape as the kernel.
    windows = sliding_window_view(array_padded, log_kernel.shape)
    kernel_axes = tuple(range(-log_kernel.ndim, 0))

    # Compute the output using the specified memory mode.
    if memory_mode == "vectorized":
        return logsumexp(windows + log_kernel_flip, axis=kernel_axes)  # type: ignore
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
        tmp = np.zeros((chunk,) + log_kernel.shape, dtype=np.float64)
        for start in range(0, array.size, chunk):
            end = min(start + chunk, array.size)
            idxs = np.unravel_index(np.arange(start, end), array.shape)
            np.add(windows[idxs], log_kernel_flip, out=tmp[: end - start])
            out[idxs] = logsumexp(tmp[: end - start], axis=kernel_axes)
        return out
    else:
        raise ValueError(f"Unknown memory_mode: {memory_mode}")
