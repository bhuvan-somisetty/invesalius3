import numpy as np
import pytest
from scipy.ndimage import shift

from invesalius.data.nonrigid_registration import demons_registration, warp_image


def _make_sphere_volume(shape=(30, 30, 30), radius=8, center=None):
    center = center if center is not None else np.array(shape) / 2
    zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    r = np.sqrt((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
    return (r < radius).astype(np.float64) * 255


def test_warp_image_zero_displacement_is_identity():
    image = _make_sphere_volume()
    displacement_field = np.zeros(image.shape + (3,))
    warped = warp_image(image, displacement_field)
    assert np.allclose(warped, image)


def test_warp_image_shape_mismatch_raises():
    image = _make_sphere_volume(shape=(10, 10, 10))
    bad_field = np.zeros((5, 5, 5, 3))
    with pytest.raises(ValueError):
        warp_image(image, bad_field)


def test_demons_registration_shape_mismatch_raises():
    fixed = _make_sphere_volume(shape=(10, 10, 10))
    moving = _make_sphere_volume(shape=(12, 12, 12))
    with pytest.raises(ValueError):
        demons_registration(fixed, moving)


def test_demons_registration_requires_3d_volumes():
    fixed = np.zeros((10, 10))
    moving = np.zeros((10, 10))
    with pytest.raises(ValueError):
        demons_registration(fixed, moving)


def test_demons_registration_identical_images_no_op():
    fixed = _make_sphere_volume()
    displacement_field, warped = demons_registration(fixed, fixed.copy())

    assert np.abs(displacement_field).max() == 0.0
    assert np.array_equal(warped, fixed)


def test_demons_registration_recovers_rigid_translation():
    fixed = _make_sphere_volume()
    moving = shift(fixed, shift=(1.5, -1.0, 2.0), order=1)

    mse_before = np.mean((moving - fixed) ** 2)
    _, warped = demons_registration(fixed, moving)
    mse_after = np.mean((warped - fixed) ** 2)

    assert mse_after < mse_before * 0.5


def test_demons_registration_recovers_smooth_nonrigid_deformation():
    fixed = _make_sphere_volume()

    zz, yy, xx = np.mgrid[0:30, 0:30, 0:30]
    synthetic_field = np.zeros(fixed.shape + (3,))
    synthetic_field[..., 0] = 1.5 * np.sin(yy / 6.0)
    synthetic_field[..., 1] = 1.0 * np.cos(xx / 6.0)
    synthetic_field[..., 2] = 0.8 * np.sin(zz / 5.0)
    moving = warp_image(fixed, synthetic_field)

    mse_before = np.mean((moving - fixed) ** 2)
    _, warped = demons_registration(fixed, moving)
    mse_after = np.mean((warped - fixed) ** 2)

    assert mse_after < mse_before * 0.5


def test_demons_registration_returns_field_shaped_for_moving():
    fixed = _make_sphere_volume(shape=(16, 16, 16))
    moving = shift(fixed, shift=(1.0, 0.5, -0.5), order=1)

    displacement_field, warped = demons_registration(fixed, moving, num_iterations=10)

    assert displacement_field.shape == fixed.shape + (3,)
    assert warped.shape == fixed.shape
