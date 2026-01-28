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
    """
    Calculate the logarithm of an N-dimensional convolution using the
    log-sum-exp trick. This avoids numerical underflow but is otherwise
    equivalent to taking the logarithm of the convolution computed with 'same'
    mode in scipy.signal.convolve.

    Parameters
    ----------
    array : np.ndarray
        The input array to be convolved. Must be non-negative.
    kernel : np.ndarray, optional
        The kernel array. Must have odd dimensions and be non-negative.
        Either `kernel` or `log_kernel` must be provided.
    log_kernel : np.ndarray, optional
        The logarithm of the kernel. If provided, `kernel` must be None.
    memory_mode : {"vectorized", "loop"} or int, default=1000
        Controls the memory usage and computation strategy:
        - "vectorized": Fully vectorized computation (fastest, uses most memory).
        - "loop" or 1: Loop over each output element (slowest, uses least memory).
        - int: Chunked computation, processes `memory_mode` output elements at
               a time (balanced memory and speed).

    Returns
    -------
    out : np.ndarray
        The result of the log-convolution, with the same shape as `array`.

    Notes
    -----
    The function computes:
        out[i1, i2, ..., iN] = log(sum_j(array[...] * kernel[...]))
    using the log-sum-exp trick for numerical stability:
        out[i1, ..., iN] = logsumexp_j(log(array[...]) + log(kernel[...]))

    Examples
    --------
    >>> import numpy as np
    >>> from KDEpy.log_convolve import log_convolve
    >>> array = np.array([1.0, 2.0, 3.0, 4.0])
    >>> kernel = np.array([0.25, 0.5, 0.25])
    >>> log_conv = log_convolve(array, kernel)
    >>> np.allclose(np.exp(log_conv), np.convolve(array, kernel, mode='same'))
    True
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
