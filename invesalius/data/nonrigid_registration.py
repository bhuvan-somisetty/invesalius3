# --------------------------------------------------------------------------
# Software:     InVesalius - Software de Reconstrucao 3D de Imagens Medicas
# Copyright:    (C) 2001  Centro de Pesquisas Renato Archer
# Homepage:     http://www.softwarepublico.gov.br
# Contact:      invesalius@cti.gov.br
# License:      GNU - GPL 2 (LICENSE.txt/LICENCA.txt)
# --------------------------------------------------------------------------
#    Este programa e software livre; voce pode redistribui-lo e/ou
#    modifica-lo sob os termos da Licenca Publica Geral GNU, conforme
#    publicada pela Free Software Foundation; de acordo com a versao 2
#    da Licenca.
#
#    Este programa eh distribuido na expectativa de ser util, mas SEM
#    QUALQUER GARANTIA; sem mesmo a garantia implicita de
#    COMERCIALIZACAO ou de ADEQUACAO A QUALQUER PROPOSITO EM
#    PARTICULAR. Consulte a Licenca Publica Geral GNU para obter mais
#    detalhes.
# --------------------------------------------------------------------------
"""
Non-rigid (deformable) registration between two 3D image volumes.

Implements Thirion's Demons algorithm, an intensity-based, iterative
registration method that estimates a dense per-voxel displacement field
warping a moving image onto a fixed image. Unlike the point-based rigid
ICP registration in invesalius.navigation.iterativeclosestpoint (used to
align tracker fiducials to patient space), this operates directly on two
image volumes and can recover local, non-linear deformation such as brain
shift or soft tissue motion between two scans of the same subject.

Only numpy/scipy are used, both already project dependencies, so this
does not introduce a new dependency (see discussion on issue #1433 about
SimpleITK/Elastix). This module implements the registration backend only;
it is not yet wired into any UI panel or navigation workflow.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def warp_image(image: np.ndarray, displacement_field: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Resamples `image` through a dense displacement field.

    Parameters
    ----------
    image : ndarray
        3D array to be warped.
    displacement_field : ndarray
        Array of shape ``image.shape + (image.ndim,)`` giving, for each
        voxel, the offset to sample `image` at along each axis.
    order : int
        Spline interpolation order passed to
        ``scipy.ndimage.map_coordinates`` (1 = trilinear).

    Returns
    -------
    ndarray
        `image` resampled at ``coordinate + displacement_field[coordinate]``
        for every voxel, same shape and dtype as `image`.
    """
    if displacement_field.shape[:-1] != image.shape:
        raise ValueError(
            "displacement_field's leading shape must match image.shape, "
            f"got {displacement_field.shape[:-1]} for image shape {image.shape}"
        )

    grid = np.indices(image.shape, dtype=np.float64)
    sample_coords = grid + np.moveaxis(displacement_field, -1, 0)

    warped = map_coordinates(
        image.astype(np.float64, copy=False),
        sample_coords,
        order=order,
        mode="nearest",
    )
    return warped.astype(image.dtype, copy=False)


def demons_registration(
    fixed: np.ndarray,
    moving: np.ndarray,
    num_iterations: int = 100,
    smoothing_sigma: float = 1.0,
    step_length: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Registers `moving` onto `fixed` using Thirion's Demons algorithm.

    At each iteration, a per-voxel displacement update is computed from
    the local intensity difference between `fixed` and the
    currently-warped `moving`, using Thirion's diffeomorphic demons force
    (normalized by the local gradient magnitude and intensity difference
    to keep the update numerically stable in flat, low-gradient regions).
    The accumulated displacement field is Gaussian-smoothed after every
    iteration, which regularizes the deformation and is what keeps the
    algorithm from just overfitting to noise.

    Parameters
    ----------
    fixed : ndarray
        3D reference image that `moving` is registered onto.
    moving : ndarray
        3D image to be warped into alignment with `fixed`. Must have the
        same shape as `fixed`.
    num_iterations : int
        Number of demons update iterations to run.
    smoothing_sigma : float
        Gaussian smoothing sigma (in voxels) applied to the displacement
        field after every iteration, for regularization.
    step_length : float
        Scale factor applied to each iteration's displacement update.

    Returns
    -------
    (displacement_field, warped_moving) : (ndarray, ndarray)
        `displacement_field` has shape ``fixed.shape + (fixed.ndim,)`` and
        gives the total per-voxel offset applied to `moving`.
        `warped_moving` is `moving` resampled through that field, i.e.
        ``warp_image(moving, displacement_field)``.

    Raises
    ------
    ValueError
        If `fixed` and `moving` don't have the same shape, or aren't 3D.
    """
    if fixed.shape != moving.shape:
        raise ValueError(
            f"fixed and moving must have the same shape, got {fixed.shape} and {moving.shape}"
        )
    if fixed.ndim != 3:
        raise ValueError(f"demons_registration expects 3D volumes, got {fixed.ndim}D")

    fixed = fixed.astype(np.float64, copy=False)
    moving = moving.astype(np.float64, copy=False)

    displacement_field = np.zeros(fixed.shape + (3,), dtype=np.float64)
    fixed_gradient = np.stack(np.gradient(fixed), axis=-1)
    fixed_gradient_sq_norm = np.sum(fixed_gradient**2, axis=-1)

    warped_moving = moving.copy()

    for _ in range(num_iterations):
        intensity_diff = warped_moving - fixed

        # Thirion's demons force: intensity_diff * grad / (|grad|^2 + diff^2),
        # with a small epsilon to avoid dividing by zero in flat regions.
        denominator = fixed_gradient_sq_norm + intensity_diff**2
        denominator = np.where(denominator < 1e-8, 1e-8, denominator)
        force = -(intensity_diff[..., np.newaxis] * fixed_gradient) / denominator[..., np.newaxis]

        displacement_field += step_length * force
        for axis in range(3):
            displacement_field[..., axis] = gaussian_filter(
                displacement_field[..., axis], sigma=smoothing_sigma
            )

        warped_moving = warp_image(moving, displacement_field)

    return displacement_field, warped_moving
