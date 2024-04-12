import math

import numpy as np
from numba import njit, prange


def init_boids(boids: np.ndarray, asp: float, vrange: tuple[float, float]):
    """
        Initialize the boids with random positions and velocities.

        Args:
        - boids (np.ndarray): Array representing the boids with shape (N, 6) where N is the number of boids.
                              Each row represents a boid with the following columns:
                              [x_position, y_position, x_velocity, y_velocity, x_acceleration, y_acceleration]
        - asp (float): Aspect ratio of the simulation area (width/height).
        - vrange (tuple[float, float]): Range of initial velocities for the boids.

        Returns:
        None
        """
    n = boids.shape[0]
    rng = np.random.default_rng()
    boids[:, 0] = rng.uniform(0., asp, size=n)
    boids[:, 1] = rng.uniform(0., 1., size=n)
    alpha = rng.uniform(0, 2 * np.pi, size=n)
    v = rng.uniform(*vrange, size=n)
    c, s = np.cos(alpha), np.sin(alpha)
    boids[:, 2] = v * c
    boids[:, 3] = v * s

@njit()
def nor(arr: np.ndarray):
    """
        Compute the norm of each vector in the given array.

        Args:
        - arr (np.ndarray): Input array of vectors with shape (N, 2) where N is the number of vectors.

        Returns:
        np.ndarray: Array containing the norms of each vector.
        """
    return np.sqrt(np.sum(arr ** 2, axis=1))


@njit()
def median(arr, axis):
    """
        Compute the median along the specified axis.

        Args:
        - arr (np.ndarray): Input array.
        - axis (int): Axis along which to compute the median (0 for columns, 1 for rows).

        Returns:
        np.ndarray: Array containing the median values along the specified axis.
        """
    assert arr.ndim == 2, "Input array must be 2D."
    assert axis in (0, 1), "Axis must be 0 or 1."


    if axis == 0:
        result_size = arr.shape[1]
    else:
        result_size = arr.shape[0]

    result = np.empty(result_size, dtype=arr.dtype)
    if axis == 0:
        for i in range(result_size):
            result[i] = np.median(arr[:, i])
    else:
        for i in range(result_size):
            result[i] = np.median(arr[i, :])
    return result


@njit()
def directions(boids: np.ndarray, dt: float) -> np.ndarray:
    """
        Compute the next positions of the boids based on their velocities.

        Args:
        - boids (np.ndarray): Array representing the boids with shape (N, 6).
        - dt (float): Time step.

        Returns:
        np.ndarray: Array containing the next positions of the boids.
        """
    return np.hstack((
        boids[:, :2] - dt * boids[:, 2:4],
        boids[:, :2]
    ))


@njit()
def clip_mag(arr: np.ndarray,
             lims: tuple[float, float] = (0., 1.)):
    """
        Clip the magnitudes of vectors in the given array to a specified range.

        Args:
        - arr (np.ndarray): Input array of vectors with shape (N, 2).
        - lims (tuple[float, float]): Lower and upper bounds for the magnitudes.

        Returns:
        None
        """
    v = nor(arr)
    mask = v > 0
    v_clip = np.clip(v, *lims)
    arr[mask] *= (v_clip[mask] / v[mask]).reshape(-1, 1)

@njit()
def propagate(boids: np.ndarray, dt: float, vrange: tuple[float, float]):
    """
        Propagate the boids based on their velocities and accelerations.

        Args:
        - boids (np.ndarray): Array representing the boids with shape (N, 6).
        - dt (float): Time step.
        - vrange (tuple[float, float]): Range of allowed velocities for the boids.

        Returns:
        None
        """
    boids[:, 2:4] += dt * boids[:, 4:6]
    clip_mag(boids[:, 2:4], vrange)
    boids[:, 0:2] += dt * boids[:, 2:4]


@njit(parallel=True)
def distances(boids: np.ndarray, perception: float, perception_angle: float) -> np.ndarray:
    """
        Compute pairwise distances between boids within perception range and angle.

        Args:
        - boids (np.ndarray): Array representing the boids with shape (N, 6).
        - perception (float): Maximum distance for boids to perceive each other.
        - perception_angle (float): Field of view angle for boids.

        Returns:
        np.ndarray: Array containing pairwise distances between boids.
        """
    n = boids.shape[0]
    D = np.empty((n, n), dtype=np.float64)

    for i in prange(n):
        for j in prange(n):
            delta = boids[i, :2] - boids[j, :2]
            distance = np.sqrt(delta @ delta)

            direction = np.arctan2(boids[i, 3], boids[i, 2])
            angle_to_neighbor = np.arctan2(delta[1], delta[0]) - direction
            angle_to_neighbor = (angle_to_neighbor + np.pi) % (2 * np.pi) - np.pi  # Нормализация угла в диапазон [-π, π]

            if distance < perception and -perception_angle / 2 <= angle_to_neighbor <= perception_angle / 2:
                D[i, j] = distance
            else:
                D[i, j] = 1e10

    return D

@njit()
def cohesion(boids: np.ndarray,
             idx: int,
             neigh_mask: np.ndarray,
             perception: float) -> np.ndarray:
    """
        Compute cohesion acceleration for a given boid.

        Args:
        - boids (np.ndarray): Array representing the boids with shape (N, 6).
        - idx (int): Index of the current boid.
        - neigh_mask (np.ndarray): Boolean mask indicating neighboring boids.
        - perception (float): Maximum distance for boids to perceive each other.

        Returns:
        np.ndarray: Cohesion acceleration vector.
        """
    center = median(boids[neigh_mask, :2], axis=0)
    a = (center - boids[idx, :2]) / perception
    return a

@njit()
def separation(boids: np.ndarray,
               idx: int,
               neigh_mask: np.ndarray,
               perception: float) -> np.ndarray:
    """
        Compute separation acceleration for a given boid.

        Args:
        - boids (np.ndarray): Array representing the boids with shape (N, 6).
        - idx (int): Index of the current boid.
        - neigh_mask (np.ndarray): Boolean mask indicating neighboring boids.
        - perception (float): Maximum distance for boids to perceive each other.

        Returns:
        np.ndarray: Separation acceleration vector.
        """

    d = median(boids[neigh_mask, :2] - boids[idx, :2], axis=0)
    return -d / ((d[0] ** 2 + d[1] ** 2) + 1)

@njit()
def alignment(boids: np.ndarray,
              idx: int,
              neigh_mask: np.ndarray,
              vrange: tuple) -> np.ndarray:
    """
        Compute alignment acceleration for a given boid.

        Args:
        - boids (np.ndarray): Array representing the boids with shape (N, 6).
        - idx (int): Index of the current boid.
        - neigh_mask (np.ndarray): Boolean mask indicating neighboring boids.
        - vrange (tuple): Range of allowed velocities for the boids.

        Returns:
        np.ndarray: Alignment acceleration vector.
        """
    v_mean = median(boids[neigh_mask, 2:4], axis=0)
    a = (v_mean - boids[idx, 2:4]) / (2 * vrange[1])
    return a


@njit(parallel=True)
def walls(boids: np.ndarray, asp: float):
    """
        Compute wall repulsion accelerations for all boids.

        Args:
        - boids (np.ndarray): Array representing the boids with shape (N, 6).
        - asp (float): Aspect ratio of the simulation area.

        Returns:
        np.ndarray: Array containing wall repulsion accelerations for all boids.
        """
    c = 1
    x = boids[:, 0]
    y = boids[:, 1]

    a_left = np.zeros_like(x)
    a_right = np.zeros_like(x)
    a_bottom = np.zeros_like(y)
    a_top = np.zeros_like(y)

    for i in prange(x.shape[0]):
        a_left[i] = 1 / (np.abs(x[i]) + c) ** 2
        a_right[i] = -1 / (np.abs(x[i] - asp) + c) ** 2
        a_bottom[i] = 1 / (np.abs(y[i]) + c) ** 2
        a_top[i] = -1 / (np.abs(y[i] - 1.) + c) ** 2

    return np.column_stack((a_left + a_right, a_bottom + a_top))

@njit()
def noise():
    """
        Generate random noise vector.

        Returns:
        np.ndarray: Random noise vector.
        """
    arr = np.random.rand(2)
    if np.random.rand(1) > .5:
        arr[0] *= -1
    if np.random.rand(1) > .5:
        arr[1] *= -1
    return arr
@njit(parallel=True)
def flocking(boids: np.ndarray,
             perception: float,
             perception_angle: float,
             coeffs: np.ndarray,
             asp: float,
             vrange: tuple):
    """
        Update boid positions and velocities based on flocking behaviors.

        Args:
        - boids (np.ndarray): Array representing the boids with shape (N, 6).
        - perception (float): Maximum distance for boids to perceive each other.
        - perception_angle (float): Field of view angle for boids.
        - coeffs (np.ndarray): Coefficients for cohesion, alignment, separation, wall avoidance, and noise.
        - asp (float): Aspect ratio of the simulation area.
        - vrange (tuple): Range of allowed velocities for the boids.

        Returns:
        None
        """
    D = distances(boids, perception, perception_angle)
    N = boids.shape[0]
    for i in prange(N):
        D[i, i] = perception + 1
    mask = D < perception
    wal = walls(boids, asp)
    for i in prange(N):
        if not np.any(mask[i]):
            coh = np.zeros(2)
            alg = np.zeros(2)
            sep = np.zeros(2)
            ns = np.zeros(2)
        else:
            coh = cohesion(boids, i, mask[i], perception)
            alg = alignment(boids, i, mask[i], vrange)
            sep = separation(boids, i, mask[i], perception)
            ns = noise()
        a = coeffs[0] * coh + coeffs[1] * alg + \
            coeffs[2] * sep + coeffs[3] * wal[i] + coeffs[4] * ns
        boids[i, 4:6] = a
