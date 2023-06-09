import numba
import numpy as np


@numba.jit(nopython=True, parallel=True, cache=True, error_model="numpy")
def compute_x_map_from_time_map(
    time_map: np.ndarray, x_map_width: int, t_px_scale: int, X_OFFSET: int, num_scanlines: int
):
    """Create an X-Map (y, t -> x) from a time map (x, y -> t).

    To create the X-Map, we perform a search for the optimal x-coordinate for each t-coordinate,
    akin to the epipolar search in stereo vision.

    All x values in the X-Map will be offset by X_OFFSET, so that x=0 starts at X_OFFSET.
    """

    x_map = np.zeros((time_map.shape[0], x_map_width), dtype=np.int16)

    # when matching, disregard candidates with more than time of two scanlines difference:
    # this is important at the top and bottom of the projector image, where the time map
    # may not be defined for the full width of the projector
    max_t_diff = 2 / num_scanlines

    t_diffs = np.zeros((time_map.shape[0], x_map_width), dtype=np.float32)

    for y in numba.prange(x_map.shape[0]):
        for t_coord in range(x_map.shape[1]):
            # compute optimal x for each t

            t = t_coord / t_px_scale

            # TODO 0-value is not defined - but also the timestamp at the first pixel
            # to fix, add something akin X_OFFSET to the proj time map
            if t == 0:
                continue

            min_t_diff = np.inf
            min_t_diff_x = -1

            for x in range(time_map.shape[1]):
                t_map = time_map[y, x]
                if t_map == 0:
                    continue

                t_diff = np.abs(t - t_map)
                if t_diff < min_t_diff:
                    min_t_diff = t_diff
                    min_t_diff_x = x

            if min_t_diff_x != -1:
                if min_t_diff <= max_t_diff:
                    x_map[y, t_coord] = min_t_diff_x + X_OFFSET
                    t_diffs[y, t_coord] = min_t_diff

    return x_map, t_diffs
