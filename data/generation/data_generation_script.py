#!/usr/bin/env python
# coding: utf-8
"""Simulated PSF dataset generation script.

This script can generate different types of PSF dataset for validation.
It should fulfill all of WaveDiff's PSF simulation requirements.

:Author: Tobias Liaudat <tobias.liaudat@cea.fr>
"""


import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

from wf_psf.utils.utils import (
    scale_to_range,
    zernike_generator,
    calc_wfe_rms,
    add_noise,
    generate_n_mask,
)
from wf_psf.utils.read_config import read_conf, RecursiveNamespace
from wf_psf.sims.psf_simulator import PSFSimulator
from wf_psf.utils.preprocessing import shift_x_y_to_zk1_2_wavediff
from wf_psf.sims.spatial_varying_psf import SpatialVaryingPSF, ZernikeHelper
from wf_psf.utils.ccd_misalignments import CCDMisalignmentCalculator


# Pre-defined colormap
try:
    top = mpl.colormaps["Oranges_r"].resampled(128)
    bottom = mpl.colormaps["Blues"].resampled(128)
except AttributeError:
    # For older versions of matplotlib
    top = mpl.cm.get_cmap(
        "Oranges_r", 128
    )  # Get the "Oranges_r" colormap with 128 colors
    bottom = mpl.cm.get_cmap("Blues", 128)  # Get the "Blues" colormap with 128 colors

newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name="OrangeBlue")
font = {"size": 18}
mpl.rc("font", **font)


def recursively_convert_lists_to_floats(namespace):
    """Recursively convert all elements in lists in a RecursiveNamespace
    that are strings to floats.
    """
    for key, value in vars(namespace).items():
        if isinstance(value, list):
            # Convert list elements to floats
            setattr(
                namespace,
                key,
                [float(item) if isinstance(item, str) else item for item in value],
            )
        elif isinstance(value, RecursiveNamespace):
            # Recursively process nested namespaces
            recursively_convert_lists_to_floats(value)


def main(args):
    # Load config parameters
    config_params = read_conf(args.config)

    # Convert all strings in lists in the config to floats
    recursively_convert_lists_to_floats(config_params)

    # Version Id of the generated dataset
    dataset_version = config_params.dataset_version

    # Random seed for the generation of the dataset
    random_seed = config_params.random_seed
    # Set the random seed
    np.random.seed(random_seed)

    # Plot some figures regarding the generated dataset
    plot_option = config_params.plot_option

    # ------------ #
    # Paths

    # Input SEDs
    SED_dir_path = config_params.paths.SED_dir_path
    # Reference datasets
    ref_train_dataset_path = config_params.paths.ref_train_dataset_path
    ref_test_dataset_path = config_params.paths.ref_test_dataset_path
    # Output data directory path
    output_dir = config_params.paths.output_dir
    # Output figs directory path
    output_fig_dir = config_params.paths.output_fig_dir
    # Input data
    sim_optical_data_path = config_params.paths.sim_optical_data_path
    # CCD misalignments data
    ccd_misalignment_path = config_params.paths.ccd_misalignment_path

    # ------------ #
    # Base parameters
    # Define PSF generation parameters

    # Range of coordinate values
    x_lims = config_params.base_params.x_lims
    y_lims = config_params.base_params.y_lims
    # Max number of Zernike modes to be used
    max_order = config_params.base_params.max_order
    # Maximum WFE RMS in [um]
    max_wfe_rms = config_params.base_params.max_wfe_rms
    # Use an Euclid-like obscuration mask
    euclid_obsc = config_params.base_params.euclid_obsc
    # Postage stamp size in pixels
    output_dim = config_params.base_params.output_dim
    # Top-hat filter to avoid the aliasing effect in the obscuration mask
    LP_filter_length = config_params.base_params.LP_filter_length
    # WFE pupil size
    pupil_diameter = config_params.base_params.pupil_diameter
    # Maximum allowed oversampling rate with respect to the observation resolution
    oversampling_rate = config_params.base_params.oversampling_rate
    # The output oversampling Q value for the observation resolution of the PSF.
    # If `oversampling_rate/output_Q = 1`, then the PSF is not oversampled
    output_Q = config_params.base_params.output_Q
    # Number of spectral bins to be used in the spectral integration
    n_bins = config_params.base_params.n_bins
    # Parameters for the super resolution PSFs
    # Postage stamp size in pixels of the super resolved PSF
    SR_output_dim = config_params.base_params.SR_output_dim
    # Output_Q value to generate the super resolved PSF
    # The upsampling factor is equal to the ratio `output_Q/SR_output_Q`
    SR_output_Q = config_params.base_params.SR_output_Q
    # Defaulted to the Euclid pixel size of 12 um. Value in [m].
    pix_sampling = config_params.base_params.pix_sampling

    # ------------ #
    # PSF field parameters

    # Differentiate the two datasets
    n_train_stars = config_params.psf_field_params.n_train_stars
    n_test_stars = config_params.psf_field_params.n_test_stars

    # Simulation options can be: 'SFE', 'NoSFE', `new_polynomial`, `reference_polynomial`
    # If `SFE` or `NoSFE` are used, we load the corresponding optical simulated data and use itto generate the PSFs
    # If `new_polynomial` is used, we generate a new random polynomial variation
    # If `reference_polynomial` is used, we load a reference polynomial variation and use it togenerate the PSFs
    sims_option = config_params.psf_field_params.sims_option
    # Positions of the stars in the field of view
    # Simulation options can be: 'SFE', 'NoSFE', `random`, `reference`
    # If `random` is used, the positions are randomly generated in the field of view
    # If `reference` is used, the positions are taken from the reference dataset
    # If `SFE` or `NoSFE` are used, the positions are taken from the optical simulations
    # Note: if `sims_option` is `SFE` or `NoSFE` then the positions are forced to be the ones from the optical simulations
    positions_options = config_params.psf_field_params.positions_options
    # Polynomial variations of the PSF field
    # Used if the `sims_option` is set to "new_polynomial"
    psf_field_d_max = config_params.psf_field_params.psf_field_d_max
    psf_field_grid_points = config_params.psf_field_params.psf_field_grid_points
    # Gaussian noise for training stars
    SNR_range = config_params.psf_field_params.SNR_range

    # ------------ #
    # Dataset features

    # Centroid shift options
    add_intrapixel_shifts = config_params.dataset_features.add_intrapixel_shifts
    # In pixels (should be abs(limits)<0.5)
    intrapixel_shift_range = config_params.dataset_features.intrapixel_shift_range
    # CCD misalignment options
    add_ccd_misalignments = config_params.dataset_features.add_ccd_misalignments
    # Add random masks to the observed PSFs (simulates the effect of Cosmic Rays)
    add_masks = config_params.dataset_features.add_masks
    # Options: 'random', 'unitary' (the unitary mask is a dummy mask with all pixels unmasked)
    mask_type = config_params.dataset_features.mask_type

    # ------------ #
    # Prior error field parameters

    # Polynomial order of the error field
    error_field_d_max = config_params.prior_error_field_params.error_field_d_max
    # Number of anchor points to generate the randome realisation of the error field
    error_field_grid_points = (
        config_params.prior_error_field_params.error_field_grid_points
    )
    # The model's units are um
    # Input required wfe_rms to SpatialVaryingPSF for getting the required wfe_rms.
    # req_wfe_rms = np.array([53, 26, 13.5, 6.5, 2.65, 1.35]) * 1e-3
    # desired_wfe_rms = np.array([40, 20, 10, 5, 2, 1]) * 1e-3
    error_field_req_wfe_rms = (
        config_params.prior_error_field_params.error_field_req_wfe_rms
    )
    error_field_desired_wfe_rms = (
        config_params.prior_error_field_params.error_field_desired_wfe_rms
    )

    # ------------ #
    # PSF Simulator

    # Generate Zernike maps
    zernikes = zernike_generator(n_zernikes=max_order, wfe_dim=pupil_diameter)
    pupil_mask = ~np.isnan(zernikes[0])
    np_zernikes = np.array(zernikes)

    # Initialize PSF simulator
    sim_PSF_toolkit = PSFSimulator(
        max_order=max_order,
        max_wfe_rms=max_wfe_rms,
        output_dim=output_dim,
        oversampling_rate=oversampling_rate,
        output_Q=output_Q,
        pupil_diameter=pupil_diameter,
        euclid_obsc=euclid_obsc,
        LP_filter_length=LP_filter_length,
    )
    # Save obscurations
    obscurations = sim_PSF_toolkit.obscurations

    # ------------ #
    # Plots
    # Do some plots of the dataset
    # Plot the obscurations
    if plot_option:
        plt.figure(figsize=(6, 6))
        plt.imshow(obscurations)
        plt.colorbar()
        plt.savefig(output_fig_dir + dataset_version + "-obscurations.pdf")
        # plt.show()
        plt.close()

    # Compute positions

    if (
        sims_option == "SFE"
        or positions_options == "SFE"
        or sims_option == "NoSFE"
        or positions_options == "NoSFE"
    ):
        # We use the positions from the optical simulations
        if sims_option == "SFE" or positions_options == "SFE":
            # We use the positions from the optical simulations "SFE"
            sim_data_str_X = "rZ_SFE_X"
            sim_data_str_Y = "rZ_SFE_Y"

        if sims_option == "NoSFE" or positions_options == "NoSFE":
            # We use the positions from the optical simulations "NoSFE"
            sim_data_str_X = "rZ_NoSFE_X"
            sim_data_str_Y = "rZ_NoSFE_Y"

        if positions_options != "SFE" and positions_options != "NoSFE":
            print(
                f"Warning: The positions options is set to {sims_option}. The positions will be taken from the optical simulations."
            )

        # Load simulated optical data
        sim_optical_data = np.load(sim_optical_data_path, allow_pickle=True)[()]
        sim_optical_data.keys()

        # Replace the position that has NaN
        sim_optical_data[sim_data_str_X][21, 30] = (
            sim_optical_data[sim_data_str_X][22, 30]
            + sim_optical_data[sim_data_str_X][20, 30]
            + sim_optical_data[sim_data_str_X][21, 29]
            + sim_optical_data[sim_data_str_X][21, 31]
        ) / 4
        sim_optical_data[sim_data_str_Y][21, 30] = (
            sim_optical_data[sim_data_str_Y][22, 30]
            + sim_optical_data[sim_data_str_Y][20, 30]
            + sim_optical_data[sim_data_str_Y][21, 29]
            + sim_optical_data[sim_data_str_Y][21, 31]
        ) / 4

        # Flatten arrays SFE/NoSFE
        flat_X = np.reshape(sim_optical_data[sim_data_str_X], (-1))
        flat_Y = np.reshape(sim_optical_data[sim_data_str_Y], (-1))

        # X coordinate
        x_old_range = [np.min([flat_X]), np.max([flat_X])]
        flat_X = scale_to_range(flat_X, x_old_range, x_lims)
        # Y coordinate
        y_old_range = [np.min([flat_Y]), np.max([flat_Y])]
        flat_Y = scale_to_range(flat_Y, y_old_range, y_lims)

        # Generate random subset
        rand_idx_list = np.arange(flat_Y.shape[0])
        np.random.shuffle(rand_idx_list)

        # Shuffle lists
        flat_X = flat_X[rand_idx_list]
        flat_Y = flat_Y[rand_idx_list]

        # Check that the number of stars is valid for the simulations
        assert (n_train_stars + n_test_stars) <= flat_X.shape[0]

        # Generate training and testing positions
        # Train
        train_positions = np.zeros((n_train_stars, 2))
        train_positions[:, 0] = flat_X[0:n_train_stars]
        train_positions[:, 1] = flat_Y[0:n_train_stars]
        # Test
        test_positions = np.zeros((n_test_stars, 2))
        test_positions[:, 0] = flat_X[n_train_stars : n_train_stars + n_test_stars]
        test_positions[:, 1] = flat_Y[n_train_stars : n_train_stars + n_test_stars]

    elif positions_options == "random":
        # We generate random positions in the field of view
        # Generate unfiorm random positions in the field of view
        train_positions = np.random.rand(n_train_stars, 2)
        test_positions = np.random.rand(n_test_stars, 2)
        # Scale the positions to the field of view
        train_positions[:, 0] = scale_to_range(train_positions[:, 0], [0, 1], x_lims)
        train_positions[:, 1] = scale_to_range(train_positions[:, 1], [0, 1], y_lims)
        test_positions[:, 0] = scale_to_range(test_positions[:, 0], [0, 1], x_lims)
        test_positions[:, 1] = scale_to_range(test_positions[:, 1], [0, 1], y_lims)

    elif positions_options == "reference":
        # We use the positions from the reference dataset
        # Load the reference dataset
        ref_train_dataset = np.load(ref_train_dataset_path, allow_pickle=True)[()]
        ref_test_dataset = np.load(ref_test_dataset_path, allow_pickle=True)[()]

        # Get the positions of the stars in the field of view
        train_positions = ref_train_dataset["positions"]
        test_positions = ref_test_dataset["positions"]

        # Scale the positions to the field of view in case they are not in the xlims and ylims
        train_positions[:, 0] = scale_to_range(
            train_positions[:, 0], ref_train_dataset["parameters"]["x_lims"], x_lims
        )
        train_positions[:, 1] = scale_to_range(
            train_positions[:, 1], ref_train_dataset["parameters"]["y_lims"], y_lims
        )
        test_positions[:, 0] = scale_to_range(
            test_positions[:, 0], ref_test_dataset["parameters"]["x_lims"], x_lims
        )
        test_positions[:, 1] = scale_to_range(
            test_positions[:, 1], ref_test_dataset["parameters"]["y_lims"], y_lims
        )

        # Get the number of stars in the field of view
        num_ref_train_stars = train_positions.shape[0]
        num_ref_test_stars = test_positions.shape[0]

        # Check the number of positions in the field of view is valid
        assert num_ref_train_stars >= n_train_stars
        assert num_ref_test_stars >= n_test_stars

        # Fix to the number of required stars for each dataset
        train_positions = train_positions[0:n_train_stars, :]
        test_positions = test_positions[0:n_test_stars, :]

    else:
        raise ValueError(
            f"Invalid positions options: {positions_options}. Must be one of 'random', 'reference', 'SFE', or 'NoSFE'."
        )

    # ------------ #
    # Plot positions
    # Check out positions of the training and testing datasets
    if plot_option:
        marker_size = 10
        plt.figure(figsize=(10, 8))
        plt.scatter(
            train_positions[0:n_train_stars, 0],
            train_positions[0:n_train_stars, 1],
            s=marker_size,
            label="Train",
        )
        plt.scatter(
            test_positions[0:n_test_stars, 0],
            test_positions[0:n_test_stars, 1],
            s=marker_size,
            label="Test",
        )
        plt.legend(fontsize=16)
        plt.xlabel("X", fontsize=20)
        plt.ylabel("Y", fontsize=20)
        plt.savefig(output_fig_dir + dataset_version + "-positions_plot.pdf")
        # plt.show()
        plt.close()

    if sims_option == "SFE" or sims_option == "NoSFE":
        # We already loaded the optical data when loading the positions

        if sims_option == "SFE":
            # We use the positions from the optical simulations "SFE"
            sim_data_str = "rZ_SFE_Cube"

        if sims_option == "NoSFE":
            # We use the positions from the optical simulations "NoSFE"
            sim_data_str = "rZ_NoSFE_Cube"

        rZ_cube = sim_optical_data[sim_data_str]
        # Shift rZ dimension to the first one
        rZ_cube = rZ_cube.swapaxes(2, 1).swapaxes(1, 0)
        # Remove the piston (order 0), the x,y shifts (order 1 and 2), and Remove defocus (order 3)
        rZ_cube[0:4, :, :] = 0

        # Replace NaNs with the element next to it
        # Ask PA why this position has NaNs
        rZ_cube[:, 21, 30] = rZ_cube[:, 22, 30]
        # Some positions have bad values, overliers (>4000)
        # Replace with value from neighbouring position
        problematic_idx = np.argwhere(rZ_cube > 4000)
        for idx in problematic_idx:
            print("Problem in: ", idx)
            rZ_cube[idx[0], idx[1], idx[2]] = rZ_cube[idx[0], idx[1] + 1, idx[2]]

        # Flatten arrays SFE/NoSFE
        flat_rZ_cube = np.reshape(rZ_cube, (rZ_cube.shape[0], -1)).T
        # Change units in Zks from nm to um
        flat_rZ_cube /= 1000.0

        # Shuffle list using the same random indexes as the positions
        flat_rZ_cube = flat_rZ_cube[rand_idx_list, :]

        # Check `max_order`
        if max_order != flat_rZ_cube.shape[1]:
            print(
                f"Warning: `max_order` is not the same as the one in the optical simulations dataset. The `max_order` will be forced to {max_order}"
            )

        if max_order < flat_rZ_cube.shape[1]:
            # Remove the last orders
            flat_rZ_cube = flat_rZ_cube[:, :max_order]

        # Check that the number of stars is valid for the simulations
        assert (n_train_stars + n_test_stars) <= flat_rZ_cube.shape[0]

        # Generate training and testing datasets
        # Train
        train_zks = np.zeros((n_train_stars, max_order))
        train_zks[:, :] = flat_rZ_cube[0:n_train_stars, :]
        # Test
        test_zks = np.zeros((n_test_stars, max_order))
        test_zks[:, :] = flat_rZ_cube[n_train_stars : n_train_stars + n_test_stars, :]

    elif sims_option == "new_polynomial":

        # Compute PSF field
        polynomial_PSF_field = SpatialVaryingPSF(
            psf_simulator=sim_PSF_toolkit,
            d_max=psf_field_d_max,
            grid_points=psf_field_grid_points,
            max_order=max_order,
            x_lims=x_lims,
            y_lims=y_lims,
            n_bins=n_bins,
            lim_max_wfe_rms=max_wfe_rms,
        )
        # Extract the Cpoly from the polynomial PSF field
        C_poly_field_variations = polynomial_PSF_field.polynomial_coeffs

        # Calculate the specific field's zernike coeffs
        train_zks = ZernikeHelper.calculate_zernike(
            train_positions[:, 0],
            train_positions[:, 1],
            x_lims,
            y_lims,
            psf_field_d_max,
            C_poly_field_variations,
        ).T

        # Calculate the specific field's zernike coeffs
        test_zks = ZernikeHelper.calculate_zernike(
            test_positions[:, 0],
            test_positions[:, 1],
            x_lims,
            y_lims,
            psf_field_d_max,
            C_poly_field_variations,
        ).T

    elif sims_option == "reference_polynomial":

        # Load the reference dataset
        ref_train_dataset = np.load(ref_train_dataset_path, allow_pickle=True)[()]
        ref_test_dataset = np.load(ref_test_dataset_path, allow_pickle=True)[()]

        # Pick the C_poly from the reference train dataset
        C_poly_field_variations = ref_train_dataset["C_poly"]

        # The `C_poly` from the train and test datasets should be the same
        try:
            assert np.all(ref_train_dataset["C_poly"] == ref_test_dataset["C_poly"])
        except AssertionError:
            print(
                "Warning: `C_poly` from train and test datasets are not the same. This might be intentional or not. The `C_poly` from the train dataset will be used."
            )

        if psf_field_d_max != ref_train_dataset["parameters"]["d_max"]:
            print(
                f"Warning: `psf_field_d_max` is not the same as the one in the reference dataset. The `psf_field_d_max` will be set to {ref_train_dataset['parameters']['d_max']}"
            )
        psf_field_d_max = ref_train_dataset["parameters"]["d_max"]

        if max_order != ref_train_dataset["parameters"]["max_order"]:
            print(
                f"Warning: `max_order` is not the same as the one in the reference dataset. The `max_order` will be set to {ref_train_dataset['parameters']['max_order']}"
            )
        max_order = ref_train_dataset["parameters"]["max_order"]

        # Calculate the specific field's zernike coeffs
        train_zks = ZernikeHelper.calculate_zernike(
            train_positions[:, 0],
            train_positions[:, 1],
            x_lims,
            y_lims,
            psf_field_d_max,
            C_poly_field_variations,
        ).T

        # Calculate the specific field's zernike coeffs
        test_zks = ZernikeHelper.calculate_zernike(
            test_positions[:, 0],
            test_positions[:, 1],
            x_lims,
            y_lims,
            psf_field_d_max,
            C_poly_field_variations,
        ).T

    # Remove low order zernikes
    train_zks[:, 0:4] = 0
    test_zks[:, 0:4] = 0

    # ------------ #
    # SEDs
    # Load the SEDs
    stellar_SEDs = np.load(SED_dir_path + "SEDs.npy", allow_pickle=True)
    stellar_lambdas = np.load(SED_dir_path + "lambdas.npy", allow_pickle=True)

    # Compute total number of stars
    total_n_stars = n_train_stars + n_test_stars

    # Select random SEDs for train dataset
    train_SED_list = []
    for it in range(n_train_stars):
        selected_id_SED = np.random.randint(low=0, high=13)
        concat_SED_wv = np.concatenate(
            (
                stellar_lambdas.reshape(-1, 1),
                stellar_SEDs[selected_id_SED, :].reshape(-1, 1),
            ),
            axis=1,
        )
        train_SED_list.append(concat_SED_wv)

    # Select random SEDs for test dataset
    test_SED_list = []
    for it in range(n_test_stars):
        selected_id_SED = np.random.randint(low=0, high=13)
        concat_SED_wv = np.concatenate(
            (
                stellar_lambdas.reshape(-1, 1),
                stellar_SEDs[selected_id_SED, :].reshape(-1, 1),
            ),
            axis=1,
        )
        test_SED_list.append(concat_SED_wv)

    # ------------ #
    # Centroid shifts

    if add_intrapixel_shifts:

        # Centroid shifts
        delta_pix_x = np.random.rand(total_n_stars).reshape(-1, 1)
        delta_pix_x = scale_to_range(delta_pix_x, [0.0, 1.0], intrapixel_shift_range)
        delta_Z1_arr = shift_x_y_to_zk1_2_wavediff(delta_pix_x * pix_sampling)

        delta_pix_y = np.random.rand(total_n_stars).reshape(-1, 1)
        delta_pix_y = scale_to_range(delta_pix_y, [0.0, 1.0], intrapixel_shift_range)
        delta_Z2_arr = shift_x_y_to_zk1_2_wavediff(delta_pix_y * pix_sampling)

        delta_Z1 = np.roll(
            np.hstack([delta_Z1_arr, np.zeros((total_n_stars, max_order - 1))]),
            1,
            axis=1,
        )
        delta_Z2 = np.roll(
            np.hstack([delta_Z2_arr, np.zeros((total_n_stars, max_order - 1))]),
            2,
            axis=1,
        )

        delta_centroid_shifts = delta_Z1 + delta_Z2

        # Extract the centroid shifts for the training and testing datasets
        train_delta_centroid_shifts = delta_centroid_shifts[0:n_train_stars, :]
        test_delta_centroid_shifts = delta_centroid_shifts[
            n_train_stars : n_train_stars + n_test_stars, :
        ]
        # Extract the delta_pix for train and test datasets
        train_delta_pix_x = delta_pix_x[0:n_train_stars, :]
        train_delta_pix_y = delta_pix_y[0:n_train_stars, :]
        test_delta_pix_x = delta_pix_x[n_train_stars : n_train_stars + n_test_stars, :]
        test_delta_pix_y = delta_pix_y[n_train_stars : n_train_stars + n_test_stars, :]

        # Add the centroid shifts to the Zernike coefficients
        train_zks += train_delta_centroid_shifts
        # TO_DEFINE: For now we do add the centroid shifts to the test dataset. Should we?
        test_zks += test_delta_centroid_shifts

    # ------------ #
    # CCD missalignements

    if add_ccd_misalignments:

        # Generate the CCD misalignment calculator from file
        ccd_missalignment_calculator = CCDMisalignmentCalculator(
            ccd_misalignment_path, x_lims, y_lims
        )

        # Train
        train_delta_Z3_arr = np.array(
            [
                ccd_missalignment_calculator.get_zk4_from_position(pos)
                for pos in train_positions
            ]
        )
        train_delta_Z3 = np.roll(
            np.hstack(
                [
                    train_delta_Z3_arr.reshape(-1, 1),
                    np.zeros((train_positions.shape[0], max_order - 1)),
                ]
            ),
            3,
            axis=1,
        )

        # Test
        test_delta_Z3_arr = np.array(
            [
                ccd_missalignment_calculator.get_zk4_from_position(pos)
                for pos in test_positions
            ]
        )
        test_delta_Z3 = np.roll(
            np.hstack(
                [
                    test_delta_Z3_arr.reshape(-1, 1),
                    np.zeros((test_positions.shape[0], max_order - 1)),
                ]
            ),
            3,
            axis=1,
        )
        # Add the CCD misalignments to the Zernike coefficients
        train_zks += train_delta_Z3
        test_zks += test_delta_Z3

    # Plot an example PSF
    if plot_option:
        it = 0
        wfe_example = np.einsum(
            "ijk,ijk->jk", zernikes, train_zks[it, :].reshape(-1, 1, 1)
        )
        wfe_rms = np.sqrt(
            np.mean((wfe_example[pupil_mask] - np.mean(wfe_example[pupil_mask])) ** 2)
        )
        # Set the Z coefficients to the PSF toolkit generator
        sim_PSF_toolkit.set_z_coeffs(train_zks[it, :])
        poly_psf = sim_PSF_toolkit.generate_poly_PSF(train_SED_list[it], n_bins=n_bins)
        opd = sim_PSF_toolkit.opd
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.imshow(poly_psf, cmap="gist_stern")
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(opd, cmap=newcmp)
        plt.colorbar()
        plt.savefig(output_fig_dir + dataset_version + "-example_psf.pdf")
        # plt.show()
        plt.close()

    # Check out the WFE RMS
    # We avoid using the first 3 zernikes for the WFE RMS calculations
    # The RMS values in these do not impact the PSF's morphology
    # The 0th order doesn't affect the PSF
    # The 1st and 2nd order only produce shifts
    if plot_option:

        for zks, pos, dataset_type in zip(
            [train_zks, test_zks], [train_positions, test_positions], ["train", "test"]
        ):
            wfes_rms = np.array(
                [
                    calc_wfe_rms(np_zernikes[3:, :, :], zks[it, 3:], pupil_mask)
                    for it in range(zks.shape[0])
                ]
            )

            plt.figure(figsize=(10, 6))
            _ = plt.hist(
                wfes_rms * 1e3,
                bins=100,
                alpha=0.5,
                range=[0.05 * 1e3, 0.15 * 1e3],
                label="WFE",
            )
            plt.ylabel("Count number")
            plt.xlabel("WFE nm RMS")
            plt.legend()
            plt.savefig(
                output_fig_dir
                + dataset_version
                + "-"
                + dataset_type
                + "-hist_WFE_RMS_original_dataset.pdf"
            )
            # plt.show()
            plt.close()

            # ------------ #

            plt.figure(figsize=(12, 8))
            plt.scatter(pos[:, 0], pos[:, 1], c=wfes_rms * 1e3)
            plt.title("WFE nm RMS")
            cbar = plt.colorbar()
            cbar.ax.get_yaxis().labelpad = 20
            cbar.ax.set_ylabel("WFE nm RMS", rotation=270)
            plt.savefig(
                output_fig_dir
                + dataset_version
                + "-"
                + dataset_type
                + "-spatial_dist_WFE_RMS_original_dataset.pdf"
            )
            # plt.show()
            plt.close()

    # ------------ #
    # Pixel PSF data generation

    # Generate train polychromatic PSFs
    train_poly_psf_list = []

    print("Generate train PSFs at observation resolution")
    for it in tqdm(range(train_zks.shape[0])):
        sim_PSF_toolkit.set_z_coeffs(train_zks[it, :])
        train_poly_psf_list.append(
            sim_PSF_toolkit.generate_poly_PSF(train_SED_list[it], n_bins=n_bins)
        )

    # Generate test polychromatic PSFs
    test_poly_psf_list = []
    print("Generate test PSFs at observation resolution")
    for it in tqdm(range(test_zks.shape[0])):
        sim_PSF_toolkit.set_z_coeffs(test_zks[it, :])
        test_poly_psf_list.append(
            sim_PSF_toolkit.generate_poly_PSF(test_SED_list[it], n_bins=n_bins)
        )

    # Generate numpy arrays from the lists
    train_poly_psf_np = np.array(train_poly_psf_list)
    train_SED_np = np.array(train_SED_list)

    test_poly_psf_np = np.array(test_poly_psf_list)
    test_SED_np = np.array(test_SED_list)

    # Generate the noisy train stars
    # Copy the training stars
    noisy_train_poly_psf_np = np.copy(train_poly_psf_np)

    # Generate a dataset with a SNR varying randomly within the desired range
    rand_SNR = (
        np.random.rand(noisy_train_poly_psf_np.shape[0]) * (SNR_range[1] - SNR_range[0])
    ) + SNR_range[0]
    # Add Gaussian noise to the observations
    noisy_train_poly_psf_np = np.stack(
        [
            add_noise(_im, desired_SNR=_SNR)
            for _im, _SNR in zip(noisy_train_poly_psf_np, rand_SNR)
        ],
        axis=0,
    )

    # Also add noise to the test stars
    noisy_test_poly_psf_np = np.copy(test_poly_psf_np)
    # Generate a dataset with a SNR varying randomly within the desired range
    rand_SNR = (
        np.random.rand(noisy_test_poly_psf_np.shape[0]) * (SNR_range[1] - SNR_range[0])
    ) + SNR_range[0]
    # Add Gaussian noise to the observations
    noisy_test_poly_psf_np = np.stack(
        [
            add_noise(_im, desired_SNR=_SNR)
            for _im, _SNR in zip(noisy_test_poly_psf_np, rand_SNR)
        ],
        axis=0,
    )

    # ------------ #
    # Generate masks

    if add_masks:

        if mask_type == "random":
            # Generate random train masks
            train_masks = generate_n_mask(
                shape=noisy_train_poly_psf_np[0].shape,
                n_masks=noisy_train_poly_psf_np.shape[0],
            )
            test_masks = generate_n_mask(
                shape=test_poly_psf_np[0].shape,
                n_masks=test_poly_psf_np.shape[0],
            )

            # Apply the random masks to the observations
            noisy_train_poly_psf_np = noisy_train_poly_psf_np * train_masks.astype(
                noisy_train_poly_psf_np.dtype
            )

            masked_noisy_test_poly_psf_np = np.copy(noisy_test_poly_psf_np)
            # Apply the random masks to the test stars
            masked_noisy_test_poly_psf_np = (
                masked_noisy_test_poly_psf_np
                * test_masks.astype(noisy_test_poly_psf_np.dtype)
            )

            # Turn masks to SHE convention. 1 (True) means to mask and 0 (False) means to keep
            train_masks = ~train_masks
            test_masks = ~test_masks

        elif mask_type == "unitary":
            train_masks = np.zeros_like(noisy_train_poly_psf_np, dtype=bool)
            test_masks = np.zeros_like(test_poly_psf_np, dtype=bool)

        # Plot some examples of the generated masks
        if plot_option:
            it = 0
            plt.figure(figsize=(12, 10))
            plt.subplot(2, 2, 1)
            plt.imshow(train_masks[it].astype(float), vmin=0, cmap="gist_stern")
            plt.title("Mask")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(2, 2, 2)
            plt.imshow(
                noisy_train_poly_psf_np[it] * train_masks[it].astype(float),
                vmin=0,
                cmap="gist_stern",
            )
            plt.title("Masked noisy Star")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(2, 2, 3)
            plt.imshow(train_masks[it + 1].astype(float), vmin=0, cmap="gist_stern")
            plt.title("Mask")
            plt.xticks([])
            plt.yticks([])
            plt.subplot(2, 2, 4)
            plt.imshow(
                noisy_train_poly_psf_np[it + 1] * train_masks[it + 1].astype(float),
                vmin=0,
                cmap="gist_stern",
            )
            plt.title("Masked noisy Star")
            plt.xticks([])
            plt.yticks([])
            plt.savefig(output_fig_dir + dataset_version + "-masked_noisy_star.pdf")
            # plt.show()
            plt.close()

    # ------------ #
    # Error WFE fields generation

    # Saving lists
    C_poly_error_list = []
    train_error_zks_list = []
    test_error_zks_list = []

    # Simulation options can be: 'SFE', 'NoSFE', or `None`
    if sims_option == "SFE" or sims_option == "NoSFE":

        for it in range(len(error_field_desired_wfe_rms)):
            # Initialize PSF field
            error_field = SpatialVaryingPSF(
                psf_simulator=sim_PSF_toolkit,
                d_max=error_field_d_max,
                grid_points=error_field_grid_points,
                max_order=max_order,
                x_lims=x_lims,
                y_lims=y_lims,
                n_bins=n_bins,
                lim_max_wfe_rms=error_field_req_wfe_rms[it],
            )

            # Calculate the field's zernike coeffs for the train dataset
            train_error_zks = ZernikeHelper.calculate_zernike(
                train_positions[:, 0],
                train_positions[:, 1],
                x_lims,
                y_lims,
                error_field_d_max,
                error_field.polynomial_coeffs,
            ).T
            # Remove low order zernikes
            train_error_zks[:, 0:4] = 0

            # Calculate the field's zernike coeffs for the test dataset
            test_error_zks = ZernikeHelper.calculate_zernike(
                test_positions[:, 0],
                test_positions[:, 1],
                x_lims,
                y_lims,
                error_field_d_max,
                error_field.polynomial_coeffs,
            ).T
            # Remove low order zernikes
            test_error_zks[:, 0:4] = 0

            # Add error Zernikes to the lists
            train_error_zks_list.append(np.copy(train_error_zks))
            test_error_zks_list.append(np.copy(test_error_zks))
            # Add error C_poly to the lists
            C_poly_error_list.append(np.copy(error_field.polynomial_coeffs))

            if plot_option:

                for zks, pos, dataset_type in zip(
                    [train_error_zks, test_error_zks],
                    [train_positions, test_positions],
                    ["train", "test"],
                ):

                    # Compute WFE RMS
                    wfes_rms_error_zks = np.array(
                        [
                            calc_wfe_rms(np_zernikes[3:, :, :], zk[3:], pupil_mask)
                            for zk in zks
                        ]
                    )
                    # Plot the Zks RMS map
                    plt.figure(figsize=(8, 6))
                    plt.scatter(
                        pos[:, 0], pos[:, 1], c=wfes_rms_error_zks * 1e3, cmap=newcmp
                    )
                    cbar = plt.colorbar()
                    cbar.ax.get_yaxis().labelpad = 20
                    cbar.ax.set_ylabel("WFE nm RMS", rotation=270)
                    plt.title(
                        "Err %dnm RMS, mean RMS: %.2f nm"
                        % (
                            error_field_desired_wfe_rms[it] * 1e3,
                            np.mean(wfes_rms_error_zks) * 1e3,
                        )
                    )
                    plt.savefig(
                        output_fig_dir
                        + dataset_version
                        + "-"
                        + dataset_type
                        + "-spatial_dist_WFE_RMS_error_%dnm.pdf"
                        % (error_field_desired_wfe_rms[it] * 1e3)
                    )
                    # plt.show()
                    plt.close()
                    # Plot some Zk map examples
                    zk_ex = np.array([1, 3, 6, 15, 25, 45])
                    zk_ex = zk_ex[zk_ex < max_order]
                    plt.figure(figsize=(20, 10))
                    for j in range(len(zk_ex)):
                        plt.subplot(231 + j)
                        plt.scatter(pos[:, 0], pos[:, 1], c=zks[:, zk_ex[j]] * 1e3)
                        plt.colorbar()
                        plt.title("Zk %d, in [nm]" % zk_ex[j])
                    plt.savefig(
                        output_fig_dir
                        + dataset_version
                        + "-"
                        + dataset_type
                        + "-spatial_dist_Zk_examples_error_%dnm.pdf"
                        % (error_field_desired_wfe_rms[it] * 1e3)
                    )
                    # plt.show()
                    plt.close()

    # Generate SR stars for the training dataset

    # Initialize PSF simulator
    SR_sim_PSF_toolkit = PSFSimulator(
        # zernikes,
        max_order=max_order,
        max_wfe_rms=max_wfe_rms,
        oversampling_rate=oversampling_rate,
        output_Q=SR_output_Q,
        output_dim=SR_output_dim,
        pupil_diameter=pupil_diameter,
        euclid_obsc=euclid_obsc,
        LP_filter_length=LP_filter_length,
    )

    # Generate all the super resolved (SR) polychromatic PSFs
    SR_train_poly_psf_list = []

    print("Generate training SR PSFs")
    for it_j in tqdm(range(n_train_stars)):
        SR_sim_PSF_toolkit.set_z_coeffs(train_zks[it_j, :])
        SR_train_poly_psf_list.append(
            SR_sim_PSF_toolkit.generate_poly_PSF(train_SED_list[it_j], n_bins=n_bins)
        )

    # Generate numpy arrays from the lists
    SR_train_poly_psf_np = np.array(SR_train_poly_psf_list)

    # ------------ #
    # Save training datasets
    # Parameter dictionary
    train_dataset_params = {
        "dataset_version": dataset_version,
        "n_stars": n_train_stars,
        "output_dim": output_dim,
        "LP_filter_length": LP_filter_length,
        "euclid_obsc": euclid_obsc,
        "pupil_diameter": pupil_diameter,
        "oversampling_rate": oversampling_rate,
        "output_Q": output_Q,
        "SNR_range": SNR_range,
        "max_order": max_order,
        "x_lims": x_lims,
        "y_lims": y_lims,
        "n_bins": n_bins,
        "max_wfe_rms": max_wfe_rms,
        "sims_option": sims_option,
        "positions_options": positions_options,
        "SR_output_dim": SR_output_dim,
        "SR_output_Q": SR_output_Q,
        "pix_sampling": pix_sampling,
        "add_intrapixel_shifts": add_intrapixel_shifts,
        "intrapixel_shift_range": intrapixel_shift_range,
        "add_ccd_misalignments": add_ccd_misalignments,
        "random_seed": random_seed,
        "add_masks": add_masks,
        "mask_type": mask_type,
    }

    # dataset
    train_psf_dataset = {
        "stars": train_poly_psf_np,
        "SR_stars": SR_train_poly_psf_np,
        "noisy_stars": noisy_train_poly_psf_np,
        "positions": train_positions,
        "SEDs": train_SED_np,
        "zernike_GT": train_zks,
    }

    if add_masks:
        train_psf_dataset["masks"] = train_masks

    if add_ccd_misalignments:
        train_psf_dataset["zernike_ccd_misalignments"] = train_delta_Z3_arr

    if add_intrapixel_shifts:
        train_psf_dataset["zernike_centroid_shifts"] = train_delta_centroid_shifts
        train_psf_dataset["pix_centroid_shifts"] = np.stack(
            [train_delta_pix_x.reshape(-1), train_delta_pix_y.reshape(-1)], axis=1
        )

    if sims_option != "SFE" and sims_option != "NoSFE":
        # We include the polynomial coefficients
        train_psf_dataset["C_poly"] = C_poly_field_variations

        # Include the dataset parameters
        train_psf_dataset["dataset_params"] = train_dataset_params

        # Save dataset
        np.save(
            output_dir + dataset_version + "-train.npy",
            train_psf_dataset,
            allow_pickle=True,
        )

    elif sims_option == "SFE" or sims_option == "NoSFE":

        for it in range(len(error_field_desired_wfe_rms)):

            train_dataset_params["error_field_d_max"] = error_field_d_max
            train_dataset_params["error_field_grid_points"] = error_field_grid_points
            train_dataset_params["error_field_req_wfe_rms"] = error_field_req_wfe_rms[
                it
            ]
            train_dataset_params["error_field_desired_wfe_rms"] = (
                error_field_desired_wfe_rms[it]
            )

            # Include the prior Zernike coefficients and the error terms
            train_psf_dataset["zernike_prior"] = train_error_zks_list[it] + train_zks
            train_psf_dataset["zernike_error"] = train_error_zks_list[it]
            # Also include the error field polynomial coefficients
            train_psf_dataset["error_field_C_poly"] = C_poly_error_list[it]

            # Include the dataset parameters
            train_psf_dataset["dataset_params"] = train_dataset_params

            # Save dataset
            np.save(
                output_dir
                + dataset_version
                + "-train"
                + "_prior_error_rms_{:.0e}".format(error_field_desired_wfe_rms[it])
                + ".npy",
                train_psf_dataset,
                allow_pickle=True,
            )

    # Generate the test super resolved (SR) polychromatic PSFs
    SR_test_poly_psf_list = []

    print("Generate testing SR PSFs")
    for it_j in tqdm(range(n_test_stars)):
        SR_sim_PSF_toolkit.set_z_coeffs(test_zks[it_j, :])
        SR_test_poly_psf_list.append(
            SR_sim_PSF_toolkit.generate_poly_PSF(test_SED_list[it_j], n_bins=n_bins)
        )

    # Generate numpy arrays from the lists
    SR_test_poly_psf_np = np.array(SR_test_poly_psf_list)

    # ------------ #
    # Save test datasets
    # Parameter dictionary
    test_dataset_params = {
        "dataset_version": dataset_version,
        "n_stars": n_test_stars,
        "output_dim": output_dim,
        "LP_filter_length": LP_filter_length,
        "euclid_obsc": euclid_obsc,
        "pupil_diameter": pupil_diameter,
        "oversampling_rate": oversampling_rate,
        "output_Q": output_Q,
        "SNR_range": SNR_range,
        "max_order": max_order,
        "x_lims": x_lims,
        "y_lims": y_lims,
        "n_bins": n_bins,
        "max_wfe_rms": max_wfe_rms,
        "sims_option": sims_option,
        "positions_options": positions_options,
        "SR_output_dim": SR_output_dim,
        "SR_output_Q": SR_output_Q,
        "pix_sampling": pix_sampling,
        "add_intrapixel_shifts": add_intrapixel_shifts,
        "intrapixel_shift_range": intrapixel_shift_range,
        "add_ccd_misalignments": add_ccd_misalignments,
        "random_seed": random_seed,
        "add_masks": add_masks,
        "mask_type": mask_type,
    }

    # Test dataset
    test_psf_dataset = {
        "stars": test_poly_psf_np,
        "SR_stars": SR_test_poly_psf_np,
        "noisy_stars": noisy_test_poly_psf_np,
        "positions": test_positions,
        "SEDs": test_SED_np,
        "zernike_GT": test_zks,
    }

    if add_masks:
        test_psf_dataset["masks"] = test_masks
        test_psf_dataset["masked_noisy_stars"] = masked_noisy_test_poly_psf_np

    if add_ccd_misalignments:
        test_psf_dataset["zernike_ccd_misalignments"] = test_delta_Z3_arr

    if add_intrapixel_shifts:
        test_psf_dataset["zernike_centroid_shifts"] = test_delta_centroid_shifts
        test_psf_dataset["pix_centroid_shifts"] = np.stack(
            [test_delta_pix_x.reshape(-1), test_delta_pix_y.reshape(-1)], axis=1
        )

    if sims_option != "SFE" and sims_option != "NoSFE":
        # We include the polynomial coefficients
        test_psf_dataset["C_poly"] = C_poly_field_variations

        # Include the dataset parameters
        test_psf_dataset["dataset_params"] = test_dataset_params

        # Save dataset
        np.save(
            output_dir + dataset_version + "-test.npy",
            test_psf_dataset,
            allow_pickle=True,
        )

    elif sims_option == "SFE" or sims_option == "NoSFE":

        for it in range(len(error_field_desired_wfe_rms)):

            test_dataset_params["error_field_d_max"] = error_field_d_max
            test_dataset_params["error_field_grid_points"] = error_field_grid_points
            test_dataset_params["error_field_req_wfe_rms"] = error_field_req_wfe_rms[it]
            test_dataset_params["error_field_desired_wfe_rms"] = (
                error_field_desired_wfe_rms[it]
            )

            # Include the prior Zernike coefficients and the error terms
            test_psf_dataset["zernike_prior"] = test_error_zks_list[it] + test_zks
            test_psf_dataset["zernike_error"] = test_error_zks_list[it]
            # Also include the error field polynomial coefficients
            test_psf_dataset["error_field_C_poly"] = C_poly_error_list[it]

            # Include the dataset parameters
            test_psf_dataset["dataset_params"] = test_dataset_params

            # Save dataset
            np.save(
                output_dir
                + dataset_version
                + "-test"
                + "_prior_error_rms_{:.0e}".format(error_field_desired_wfe_rms[it])
                + ".npy",
                test_psf_dataset,
                allow_pickle=True,
            )


if __name__ == "__main__":
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description="PSF dataset generation script")
    parser.add_argument(
        "-c",
        "--config",
        default="./../configs/diffnest_DiffPIR_blur_64.yml",
        type=str,
        help="path to YAML config file",
    )
    args = parser.parse_args()

    # Train model
    main(args)
