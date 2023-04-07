"""Zernikes.

A module to make Zernike maps.

:Author: Tobias Liaudat <tobiasliaudat@gmail.com> and Jennifer Pollack <jennifer.pollack@cea.fr>

"""
import numpy as np
import zernike as zk
import tensorflow as tf

# Prepare the inputs
# Generate Zernike maps
zernikes = zernike_generator(
    n_zernikes=args["n_zernikes"], wfe_dim=args["pupil_diameter"]
)
# Now as cubes
np_zernike_cube = np.zeros((len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1]))
for it in range(len(zernikes)):
    np_zernike_cube[it, :, :] = zernikes[it]
np_zernike_cube[np.isnan(np_zernike_cube)] = 0
tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)
print("Zernike cube:")
print(tf_zernike_cube.shape)


def zernike_generator(n_zernikes, wfe_dim):
    """
    Generate Zernike maps.

    Based on the zernike github repository.
    https://github.com/jacopoantonello/zernike

    Parameters
    ----------
    n_zernikes: int
        Number of Zernike modes desired.
    wfe_dim: int
        Dimension of the Zernike map [wfe_dim x wfe_dim].

    Returns
    -------
    zernikes: list of np.ndarray
        List containing the Zernike modes.
        The values outside the unit circle are filled with NaNs.
    """
    # Calculate which n (from the (n,m) Zernike convention) we need
    # so that we have the desired total number of Zernike coefficients
    min_n = (-3 + np.sqrt(1 + 8 * n_zernikes)) / 2
    n = int(np.ceil(min_n))

    # Initialize the zernike generator
    cart = zk.RZern(n)
    # Create a [-1,1] mesh
    ddx = np.linspace(-1.0, 1.0, wfe_dim)
    ddy = np.linspace(-1.0, 1.0, wfe_dim)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)

    c = np.zeros(cart.nk)
    zernikes = []

    # Extract each Zernike map one by one
    for i in range(n_zernikes):
        c *= 0.0
        c[i] = 1.0
        zernikes.append(cart.eval_grid(c, matrix=True))

    return zernikes
