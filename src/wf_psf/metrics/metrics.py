import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import galsim as gs
import wf_psf.utils.utils as utils
from wf_psf.psf_models.tf_psf_field import build_PSF_model
from wf_psf.psf_models import tf_psf_field as psf_field
from wf_psf.sims import SimPSFToolkit as SimPSFToolkit
import logging

logger = logging.getLogger(__name__)


def compute_poly_metric(
    tf_semiparam_field,
    GT_tf_semiparam_field,
    simPSF_np,
    tf_pos,
    tf_SEDs,
    n_bins_lda=20,
    n_bins_gt=20,
    batch_size=16,
    dataset_dict=None,
):
    """Calculate metrics for polychromatic reconstructions.

    The ``tf_semiparam_field`` should be the model to evaluate, and the
    ``GT_tf_semiparam_field`` should be loaded with the ground truth PSF field.

    Relative values returned in [%] (so multiplied by 100).

    Parameters
    ----------
    tf_semiparam_field: PSF field object
        Trained model to evaluate.
    GT_tf_semiparam_field: PSF field object
        Ground truth model to produce GT observations at any position
        and wavelength.
    simPSF_np: PSF simulator object
        Simulation object to be used by ``generate_packed_elems`` function.
    tf_pos: Tensor or numpy.ndarray [batch x 2] floats
        Positions to evaluate the model.
    tf_SEDs: numpy.ndarray [batch x SED_samples x 2]
        SED samples for the corresponding positions.
    n_bins_lda: int
        Number of wavelength bins to use for the polychromatic PSF.
    n_bins_gt: int
        Number of wavelength bins to use for the ground truth polychromatic PSF.
    batch_size: int
        Batch size for the PSF calcualtions.
    dataset_dict: dict
        Dictionary containing the dataset information. If provided, and if the `'stars'` key
        is present, the noiseless stars from the dataset are used to compute the metrics.
        Otherwise, the stars are generated from the GT model.
        Default is `None`.

    Returns
    -------
    rmse: float
        RMSE value.
    rel_rmse: float
        Relative RMSE value. Values in %.
    std_rmse: float
        Sstandard deviation of RMSEs.
    std_rel_rmse: float
        Standard deviation of relative RMSEs. Values in %.

    """
    # Generate SED data list for the model
    packed_SED_data = [
        utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
        for _sed in tf_SEDs
    ]
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
    pred_inputs = [tf_pos, tf_packed_SED_data]

    # Model prediction
    preds = tf_semiparam_field.predict(x=pred_inputs, batch_size=batch_size)

    # GT data preparation
    if dataset_dict is None or "stars" not in dataset_dict:
        logger.info("Regenerating GT stars from model.")
        # Change interpolation parameters for the GT simPSF
        interp_pts_per_bin = simPSF_np.SED_interp_pts_per_bin
        simPSF_np.SED_interp_pts_per_bin = 0
        SED_sigma = simPSF_np.SED_sigma
        simPSF_np.SED_sigma = 0
        # Generate SED data list for GT model
        packed_SED_data = [
            utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_gt)
            for _sed in tf_SEDs
        ]
        tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
        tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
        pred_inputs = [tf_pos, tf_packed_SED_data]

        # GT model prediction
        GT_preds = GT_tf_semiparam_field.predict(x=pred_inputs, batch_size=batch_size)

    else:
        logger.info("Using GT stars from dataset.")
        GT_preds = dataset_dict["stars"]

    # Calculate residuals
    residuals = np.sqrt(np.mean((GT_preds - preds) ** 2, axis=(1, 2)))
    GT_star_mean = np.sqrt(np.mean((GT_preds) ** 2, axis=(1, 2)))

    # RMSE calculations
    rmse = np.mean(residuals)
    rel_rmse = 100.0 * np.mean(residuals / GT_star_mean)

    # STD calculations
    std_rmse = np.std(residuals)
    std_rel_rmse = 100.0 * np.std(residuals / GT_star_mean)

    # Print RMSE values
    logger.info("Absolute RMSE:\t %.4e \t +/- %.4e" % (rmse, std_rmse))
    logger.info("Relative RMSE:\t %.4e %% \t +/- %.4e %%" % (rel_rmse, std_rel_rmse))

    return rmse, rel_rmse, std_rmse, std_rel_rmse


def compute_mono_metric(
    tf_semiparam_field,
    GT_tf_semiparam_field,
    simPSF_np,
    tf_pos,
    lambda_list,
    batch_size=32,
):
    """Calculate metrics for monochromatic reconstructions.

    The ``tf_semiparam_field`` should be the model to evaluate, and the
    ``GT_tf_semiparam_field`` should be loaded with the ground truth PSF field.

    Relative values returned in [%] (so multiplied by 100).

    Parameters
    ----------
    tf_semiparam_field: PSF field object
        Trained model to evaluate.
    GT_tf_semiparam_field: PSF field object
        Ground truth model to produce GT observations at any position
        and wavelength.
    simPSF_np: PSF simulator object
        Simulation object capable of calculating ``phase_N`` values from
        wavelength values.
    tf_pos: list of floats [batch x 2]
        Positions to evaluate the model.
    lambda_list: list of floats [wavelength_values]
        List of wavelength values in [um] to evaluate the model.
    batch_size: int
        Batch size to process the monochromatic PSF calculations.

    Returns
    -------
    rmse_lda: list of float
        List of RMSE as a function of wavelength.
    rel_rmse_lda: list of float
        List of relative RMSE as a function of wavelength. Values in %.
    std_rmse_lda: list of float
        List of standard deviation of RMSEs as a function of wavelength.
    std_rel_rmse_lda: list of float
        List of standard deviation of relative RMSEs as a function of
        wavelength. Values in %.

    """
    # Initialise lists
    rmse_lda = []
    rel_rmse_lda = []
    std_rmse_lda = []
    std_rel_rmse_lda = []

    total_samples = tf_pos.shape[0]

    # Main loop for each wavelength
    for it in range(len(lambda_list)):
        # Set the lambda (wavelength) and the required wavefront N
        lambda_obs = lambda_list[it]
        phase_N = simPSF_np.feasible_N(lambda_obs)

        residuals = np.zeros((total_samples))
        GT_star_mean = np.zeros((total_samples))

        # Total number of epochs
        n_epochs = int(np.ceil(total_samples / batch_size))
        ep_low_lim = 0
        for ep in range(n_epochs):
            # Define the upper limit
            if ep_low_lim + batch_size >= total_samples:
                ep_up_lim = total_samples
            else:
                ep_up_lim = ep_low_lim + batch_size
            # Extract the batch
            batch_pos = tf_pos[ep_low_lim:ep_up_lim, :]

            # Estimate the monochromatic PSFs
            GT_mono_psf = GT_tf_semiparam_field.predict_mono_psfs(
                input_positions=batch_pos, lambda_obs=lambda_obs, phase_N=phase_N
            )

            model_mono_psf = tf_semiparam_field.predict_mono_psfs(
                input_positions=batch_pos, lambda_obs=lambda_obs, phase_N=phase_N
            )

            num_pixels = GT_mono_psf.shape[1] * GT_mono_psf.shape[2]

            residuals[ep_low_lim:ep_up_lim] = (
                np.sum((GT_mono_psf - model_mono_psf) ** 2, axis=(1, 2)) / num_pixels
            )
            GT_star_mean[ep_low_lim:ep_up_lim] = (
                np.sum((GT_mono_psf) ** 2, axis=(1, 2)) / num_pixels
            )

            # Increase lower limit
            ep_low_lim += batch_size

        # Calculate residuals
        residuals = np.sqrt(residuals)
        GT_star_mean = np.sqrt(GT_star_mean)

        # RMSE calculations
        rmse_lda.append(np.mean(residuals))
        rel_rmse_lda.append(100.0 * np.mean(residuals / GT_star_mean))

        # STD calculations
        std_rmse_lda.append(np.std(residuals))
        std_rel_rmse_lda.append(100.0 * np.std(residuals / GT_star_mean))

    return rmse_lda, rel_rmse_lda, std_rmse_lda, std_rel_rmse_lda


def compute_opd_metrics(tf_semiparam_field, GT_tf_semiparam_field, pos, batch_size=16):
    """Compute the OPD metrics.

    Need to handle a batch size to avoid Out-Of-Memory errors with
    the GPUs. This is specially due to the fact that the OPD maps
    have a higher dimensionality than the observed PSFs.

    The OPD RMSE is computed after having removed the mean from the
    different reconstructions. It is computed only on the
    non-obscured elements from the OPD.

    Parameters
    ----------
    tf_semiparam_field: PSF field object
        Trained model to evaluate.
    GT_tf_semiparam_field: PSF field object
        Ground truth model to produce GT observations at any position
        and wavelength.
    pos: numpy.ndarray [batch x 2]
        Positions at where to predict the OPD maps.
    batch_size: int
        Batch size to process the OPD calculations.

    Returns
    -------
    rmse: float
        Absolute RMSE value.
    rel_rmse: float
        Relative RMSE value.
    rmse_std: float
        Absolute RMSE standard deviation.
    rel_rmse_std: float
        Relative RMSE standard deviation.

    """
    # Get OPD obscurations
    np_obscurations = np.real(GT_tf_semiparam_field.obscurations.numpy())
    # Define total number of samples
    n_samples = pos.shape[0]

    # Initialise batch variables
    opd_batch = None
    GT_opd_batch = None
    counter = 0
    # Initialise result lists
    rmse_vals = np.zeros(n_samples)
    rel_rmse_vals = np.zeros(n_samples)

    while counter < n_samples:
        # Calculate the batch end element
        if counter + batch_size <= n_samples:
            end_sample = counter + batch_size
        else:
            end_sample = n_samples

        # Define the batch positions
        batch_pos = pos[counter:end_sample, :]
        # We calculate a batch of OPDs
        opd_batch = tf_semiparam_field.predict_opd(batch_pos).numpy()
        GT_opd_batch = GT_tf_semiparam_field.predict_opd(batch_pos).numpy()
        # Remove the mean of the OPD
        opd_batch -= np.mean(opd_batch, axis=(1, 2)).reshape(-1, 1, 1)
        GT_opd_batch -= np.mean(GT_opd_batch, axis=(1, 2)).reshape(-1, 1, 1)
        # Obscure the OPDs
        opd_batch *= np_obscurations
        GT_opd_batch *= np_obscurations
        # Generate obscuration mask
        obsc_mask = np_obscurations > 0
        nb_mask_elems = np.sum(obsc_mask)
        # Compute the OPD RMSE with the masked obscurations
        res_opd = np.sqrt(
            np.array(
                [
                    np.sum((im1[obsc_mask] - im2[obsc_mask]) ** 2) / nb_mask_elems
                    for im1, im2 in zip(opd_batch, GT_opd_batch)
                ]
            )
        )
        GT_opd_mean = np.sqrt(
            np.array(
                [np.sum(im2[obsc_mask] ** 2) / nb_mask_elems for im2 in GT_opd_batch]
            )
        )
        # RMSE calculations
        rmse_vals[counter:end_sample] = res_opd
        rel_rmse_vals[counter:end_sample] = 100.0 * (res_opd / GT_opd_mean)

        # Add the results to the lists
        counter += batch_size

    # Calculate final values
    rmse = np.mean(rmse_vals)
    rel_rmse = np.mean(rel_rmse_vals)
    rmse_std = np.std(rmse_vals)
    rel_rmse_std = np.std(rel_rmse_vals)

    # Print RMSE values
    logger.info("Absolute RMSE:\t %.4e \t +/- %.4e" % (rmse, rmse_std))
    logger.info("Relative RMSE:\t %.4e %% \t +/- %.4e %%" % (rel_rmse, rel_rmse_std))

    return rmse, rel_rmse, rmse_std, rel_rmse_std


def compute_shape_metrics(
    tf_semiparam_field,
    GT_tf_semiparam_field,
    simPSF_np,
    SEDs,
    tf_pos,
    n_bins_lda,
    n_bins_gt,
    output_Q=1,
    output_dim=64,
    batch_size=16,
    opt_stars_rel_pix_rmse=False,
    dataset_dict=None,
):
    """Compute the pixel, shape and size RMSE of a PSF model.

    This is done at a specific sampling and output image dimension.
    It is done for polychromatic PSFs so SEDs are needed.

    Parameters
    ----------
    tf_semiparam_field: PSF field object
        Trained model to evaluate.
    GT_tf_semiparam_field: PSF field object
        Ground truth model to produce GT observations at any position
        and wavelength.
    simPSF_np:
    SEDs: numpy.ndarray [batch x SED_samples x 2]
        SED samples for the corresponding positions.
    tf_pos: Tensor [batch x 2]
        Positions at where to predict the PSFs.
    n_bins_lda: int
        Number of wavelength bins to use for the polychromatic PSF.
    n_bins_gt: int
        Number of wavelength bins to use for the ground truth polychromatic PSF.
    output_Q: int
        Downsampling rate to match the specified telescope's sampling. The value
        of `output_Q` should be equal to `oversampling_rate` in order to have
        the right pixel sampling corresponding to the telescope characteristics
        `pix_sampling`, `tel_diameter`, `tel_focal_length`. The final
        oversampling obtained is `oversampling_rate/output_Q`.
        Default is `1`, so the output psf will be super-resolved by a factor of
        `oversampling_rate`. TLDR: better use `1` and measure shapes on the
        super-resolved PSFs.
    output_dim: int
        Output dimension of the square PSF stamps.
    batch_size: int
        Batch size to process the PSF estimations.
    opt_stars_rel_pix_rmse: bool
        If `True`, the relative pixel RMSE of each star is added to ther saving dictionary.
        The summary statistics are always computed.
        Default is `False`.
    dataset_dict: dict
        Dictionary containing the dataset information. If provided, and if the `'super_res_stars'`
        key is present, the noiseless super resolved stars from the dataset are used to compute
        the metrics. Otherwise, the stars are generated from the GT model.
        Default is `None`.

    Returns
    -------
    result_dict: dict
        Dictionary with all the results.

    """
    # Save original output_Q and output_dim
    original_out_Q = tf_semiparam_field.output_Q
    original_out_dim = tf_semiparam_field.output_dim
    GT_original_out_Q = GT_tf_semiparam_field.output_Q
    GT_original_out_dim = GT_tf_semiparam_field.output_dim

    # Set the required output_Q and output_dim parameters in the models
    tf_semiparam_field.set_output_Q(output_Q=output_Q, output_dim=output_dim)
    GT_tf_semiparam_field.set_output_Q(output_Q=output_Q, output_dim=output_dim)

    # Need to compile the models again
    tf_semiparam_field = build_PSF_model(tf_semiparam_field)
    GT_tf_semiparam_field = build_PSF_model(GT_tf_semiparam_field)

    # Generate SED data list
    packed_SED_data = [
        utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda) for _sed in SEDs
    ]

    # Prepare inputs
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
    pred_inputs = [tf_pos, tf_packed_SED_data]

    # PSF model
    predictions = tf_semiparam_field.predict(x=pred_inputs, batch_size=batch_size)

    # GT data preparation
    if (
        dataset_dict is None
        or "super_res_stars" not in dataset_dict
        or "SR_stars" not in dataset_dict
    ):
        logger.info("Generating GT super resolved stars from the GT model.")
        # Change interpolation parameters for the GT simPSF
        interp_pts_per_bin = simPSF_np.SED_interp_pts_per_bin
        simPSF_np.SED_interp_pts_per_bin = 0
        SED_sigma = simPSF_np.SED_sigma
        simPSF_np.SED_sigma = 0
        # Generate SED data list for GT model
        packed_SED_data = [
            utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_gt)
            for _sed in SEDs
        ]

        # Prepare inputs
        tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
        tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
        pred_inputs = [tf_pos, tf_packed_SED_data]

        # Ground Truth model
        GT_predictions = GT_tf_semiparam_field.predict(
            x=pred_inputs, batch_size=batch_size
        )

    else:
        logger.info("Using super resolved stars from dataset.")
        if "super_res_stars" in dataset_dict:
            GT_predictions = dataset_dict["super_res_stars"]
        elif "SR_stars" in dataset_dict:
            GT_predictions = dataset_dict["SR_stars"]

    # Calculate residuals
    residuals = np.sqrt(np.mean((GT_predictions - predictions) ** 2, axis=(1, 2)))
    GT_star_mean = np.sqrt(np.mean((GT_predictions) ** 2, axis=(1, 2)))

    # Pixel RMSE for each star
    if opt_stars_rel_pix_rmse:
        stars_rel_pix_rmse = 100.0 * residuals / GT_star_mean

    # RMSE calculations
    pix_rmse = np.mean(residuals)
    rel_pix_rmse = 100.0 * np.mean(residuals / GT_star_mean)

    # STD calculations
    pix_rmse_std = np.std(residuals)
    rel_pix_rmse_std = 100.0 * np.std(residuals / GT_star_mean)

    # Print pixel RMSE values
    logger.info(
        "\nPixel star absolute RMSE:\t %.4e \t +/- %.4e " % (pix_rmse, pix_rmse_std)
    )
    logger.info(
        "Pixel star relative RMSE:\t %.4e %% \t +/- %.4e %%"
        % (rel_pix_rmse, rel_pix_rmse_std)
    )

    # Measure shapes of the reconstructions
    pred_moments = [
        gs.hsm.FindAdaptiveMom(gs.Image(_pred), strict=False) for _pred in predictions
    ]

    # Measure shapes of the reconstructions
    GT_pred_moments = [
        gs.hsm.FindAdaptiveMom(gs.Image(_pred), strict=False)
        for _pred in GT_predictions
    ]

    pred_e1_HSM, pred_e2_HSM, pred_R2_HSM = [], [], []
    GT_pred_e1_HSM, GT_pred_e2_HSM, GT_pred_R2_HSM = [], [], []

    for it in range(len(GT_pred_moments)):
        if (
            pred_moments[it].moments_status == 0
            and GT_pred_moments[it].moments_status == 0
        ):
            pred_e1_HSM.append(pred_moments[it].observed_shape.g1)
            pred_e2_HSM.append(pred_moments[it].observed_shape.g2)
            pred_R2_HSM.append(2 * (pred_moments[it].moments_sigma ** 2))

            GT_pred_e1_HSM.append(GT_pred_moments[it].observed_shape.g1)
            GT_pred_e2_HSM.append(GT_pred_moments[it].observed_shape.g2)
            GT_pred_R2_HSM.append(2 * (GT_pred_moments[it].moments_sigma ** 2))

    pred_e1_HSM = np.array(pred_e1_HSM)
    pred_e2_HSM = np.array(pred_e2_HSM)
    pred_R2_HSM = np.array(pred_R2_HSM)

    GT_pred_e1_HSM = np.array(GT_pred_e1_HSM)
    GT_pred_e2_HSM = np.array(GT_pred_e2_HSM)
    GT_pred_R2_HSM = np.array(GT_pred_R2_HSM)

    # Calculate metrics

    # e1
    e1_res = GT_pred_e1_HSM - pred_e1_HSM
    e1_res_rel = (GT_pred_e1_HSM - pred_e1_HSM) / GT_pred_e1_HSM

    rmse_e1 = np.sqrt(np.mean(e1_res**2))
    rel_rmse_e1 = 100.0 * np.sqrt(np.mean(e1_res_rel**2))
    std_rmse_e1 = np.std(e1_res)
    std_rel_rmse_e1 = 100.0 * np.std(e1_res_rel)

    # e2
    e2_res = GT_pred_e2_HSM - pred_e2_HSM
    e2_res_rel = (GT_pred_e2_HSM - pred_e2_HSM) / GT_pred_e2_HSM

    rmse_e2 = np.sqrt(np.mean(e2_res**2))
    rel_rmse_e2 = 100.0 * np.sqrt(np.mean(e2_res_rel**2))
    std_rmse_e2 = np.std(e2_res)
    std_rel_rmse_e2 = 100.0 * np.std(e2_res_rel)

    # R2
    R2_res = GT_pred_R2_HSM - pred_R2_HSM

    rmse_R2_meanR2 = np.sqrt(np.mean(R2_res**2)) / np.mean(GT_pred_R2_HSM)
    std_rmse_R2_meanR2 = np.std(R2_res / GT_pred_R2_HSM)

    # Print shape/size errors
    logger.info("\nsigma(e1) RMSE =\t\t %.4e \t +/- %.4e " % (rmse_e1, std_rmse_e1))
    logger.info("sigma(e2) RMSE =\t\t %.4e \t +/- %.4e " % (rmse_e2, std_rmse_e2))
    logger.info(
        "sigma(R2)/<R2> =\t\t %.4e \t +/- %.4e " % (rmse_R2_meanR2, std_rmse_R2_meanR2)
    )

    # Print relative shape/size errors
    logger.info(
        "\nRelative sigma(e1) RMSE =\t %.4e %% \t +/- %.4e %%"
        % (rel_rmse_e1, std_rel_rmse_e1)
    )
    logger.info(
        "Relative sigma(e2) RMSE =\t %.4e %% \t +/- %.4e %%"
        % (rel_rmse_e2, std_rel_rmse_e2)
    )

    # Print number of stars
    logger.info("\nTotal number of stars: \t\t%d" % (len(GT_pred_moments)))
    logger.info(
        "Problematic number of stars: \t%d"
        % (len(GT_pred_moments) - GT_pred_e1_HSM.shape[0])
    )

    # Re-et the original output_Q and output_dim parameters in the models
    tf_semiparam_field.set_output_Q(
        output_Q=original_out_Q, output_dim=original_out_dim
    )
    GT_tf_semiparam_field.set_output_Q(
        output_Q=GT_original_out_Q, output_dim=GT_original_out_dim
    )

    # Need to compile the models again
    tf_semiparam_field = build_PSF_model(tf_semiparam_field)
    GT_tf_semiparam_field = build_PSF_model(GT_tf_semiparam_field)

    # Moment results
    result_dict = {
        "pred_e1_HSM": pred_e1_HSM,
        "pred_e2_HSM": pred_e2_HSM,
        "pred_R2_HSM": pred_R2_HSM,
        "GT_pred_e1_HSM": GT_pred_e1_HSM,
        "GT_ped_e2_HSM": GT_pred_e2_HSM,
        "GT_pred_R2_HSM": GT_pred_R2_HSM,
        "rmse_e1": rmse_e1,
        "std_rmse_e1": std_rmse_e1,
        "rel_rmse_e1": rel_rmse_e1,
        "std_rel_rmse_e1": std_rel_rmse_e1,
        "rmse_e2": rmse_e2,
        "std_rmse_e2": std_rmse_e2,
        "rel_rmse_e2": rel_rmse_e2,
        "std_rel_rmse_e2": std_rel_rmse_e2,
        "rmse_R2_meanR2": rmse_R2_meanR2,
        "std_rmse_R2_meanR2": std_rmse_R2_meanR2,
        "pix_rmse": pix_rmse,
        "pix_rmse_std": pix_rmse_std,
        "rel_pix_rmse": rel_pix_rmse,
        "rel_pix_rmse_std": rel_pix_rmse_std,
        "output_Q": output_Q,
        "output_dim": output_dim,
        "n_bins_lda": n_bins_lda,
    }

    if opt_stars_rel_pix_rmse:
        result_dict["stars_rel_pix_rmse"] = stars_rel_pix_rmse

    return result_dict


def gen_GT_wf_model(test_wf_file_path, pred_output_Q=1, pred_output_dim=64):
    r"""Generate the ground truth model and output test PSF ar required resolution.

    If `pred_output_Q=1` the resolution will be 3 times the one of Euclid.
    """
    # Load dataset
    wf_test_dataset = np.load(test_wf_file_path, allow_pickle=True)[()]

    # Extract parameters from the wf test dataset
    wf_test_params = wf_test_dataset["parameters"]
    wf_test_C_poly = wf_test_dataset["C_poly"]
    wf_test_pos = wf_test_dataset["positions"]
    tf_test_pos = tf.convert_to_tensor(wf_test_pos, dtype=tf.float32)
    wf_test_SEDs = wf_test_dataset["SEDs"]

    # Generate GT model
    batch_size = 16

    # Generate Zernike maps
    zernikes = utils.zernike_generator(
        n_zernikes=wf_test_params["max_order"], wfe_dim=wf_test_params["pupil_diameter"]
    )

    ## Generate initializations
    # Prepare np input
    simPSF_np = SimPSFToolkit(
        zernikes,
        max_order=wf_test_params["max_order"],
        pupil_diameter=wf_test_params["pupil_diameter"],
        output_dim=wf_test_params["output_dim"],
        oversampling_rate=wf_test_params["oversampling_rate"],
        output_Q=wf_test_params["output_Q"],
    )
    simPSF_np.gen_random_Z_coeffs(max_order=wf_test_params["max_order"])
    z_coeffs = simPSF_np.normalize_zernikes(
        simPSF_np.get_z_coeffs(), simPSF_np.max_wfe_rms
    )
    simPSF_np.set_z_coeffs(z_coeffs)
    simPSF_np.generate_mono_PSF(lambda_obs=0.7, regen_sample=False)
    # Obscurations
    obscurations = simPSF_np.generate_pupil_obscurations(
        N_pix=wf_test_params["pupil_diameter"],
        N_filter=wf_test_params["LP_filter_length"],
    )
    tf_obscurations = tf.convert_to_tensor(obscurations, dtype=tf.complex64)

    ## Prepare ground truth model
    # Now Zernike's as cubes
    np_zernike_cube = np.zeros(
        (len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1])
    )
    for it in range(len(zernikes)):
        np_zernike_cube[it, :, :] = zernikes[it]

    np_zernike_cube[np.isnan(np_zernike_cube)] = 0
    tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)

    # Initialize the model
    GT_tf_semiparam_field = psf_field.TF_SemiParam_field(
        zernike_maps=tf_zernike_cube,
        obscurations=tf_obscurations,
        batch_size=batch_size,
        output_Q=wf_test_params["output_Q"],
        d_max_nonparam=2,
        output_dim=wf_test_params["output_dim"],
        n_zernikes=wf_test_params["max_order"],
        d_max=wf_test_params["d_max"],
        x_lims=wf_test_params["x_lims"],
        y_lims=wf_test_params["y_lims"],
    )

    # For the Ground truth model
    GT_tf_semiparam_field.tf_poly_Z_field.assign_coeff_matrix(wf_test_C_poly)
    _ = GT_tf_semiparam_field.tf_np_poly_opd.alpha_mat.assign(
        tf.zeros_like(GT_tf_semiparam_field.tf_np_poly_opd.alpha_mat)
    )

    # Set required output_Q

    GT_tf_semiparam_field.set_output_Q(
        output_Q=pred_output_Q, output_dim=pred_output_dim
    )

    GT_tf_semiparam_field = psf_field.build_PSF_model(GT_tf_semiparam_field)

    packed_SED_data = [
        utils.generate_packed_elems(_sed, simPSF_np, n_bins=wf_test_params["n_bins"])
        for _sed in wf_test_SEDs
    ]

    # Prepare inputs
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
    pred_inputs = [tf_test_pos, tf_packed_SED_data]

    # Ground Truth model
    GT_predictions = GT_tf_semiparam_field.predict(x=pred_inputs, batch_size=batch_size)

    return GT_predictions, wf_test_pos, GT_tf_semiparam_field


## Below this line there are DEPRECATED functions


def compute_metrics(
    tf_semiparam_field,
    simPSF_np,
    test_SEDs,
    train_SEDs,
    tf_test_pos,
    tf_train_pos,
    tf_test_stars,
    tf_train_stars,
    n_bins_lda,
    batch_size=16,
):
    # Generate SED data list
    test_packed_SED_data = [
        utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
        for _sed in test_SEDs
    ]

    tf_test_packed_SED_data = tf.convert_to_tensor(
        test_packed_SED_data, dtype=tf.float32
    )
    tf_test_packed_SED_data = tf.transpose(tf_test_packed_SED_data, perm=[0, 2, 1])
    test_pred_inputs = [tf_test_pos, tf_test_packed_SED_data]
    test_predictions = tf_semiparam_field.predict(
        x=test_pred_inputs, batch_size=batch_size
    )

    # Initialize the SED data list
    packed_SED_data = [
        utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
        for _sed in train_SEDs
    ]
    # First estimate the stars for the observations
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
    inputs = [tf_train_pos, tf_packed_SED_data]
    train_predictions = tf_semiparam_field.predict(x=inputs, batch_size=batch_size)

    # Calculate RMSE values
    test_res = np.sqrt(np.mean((tf_test_stars - test_predictions) ** 2))
    train_res = np.sqrt(np.mean((tf_train_stars - train_predictions) ** 2))

    # Calculate relative RMSE values
    relative_test_res = test_res / np.sqrt(np.mean((tf_test_stars) ** 2))
    relative_train_res = train_res / np.sqrt(np.mean((tf_train_stars) ** 2))

    # Print RMSE values
    logger.info("Test stars absolute RMSE:\t %.4e" % test_res)
    logger.info("Training stars absolute RMSE:\t %.4e" % train_res)

    # Print RMSE values
    logger.info("Test stars relative RMSE:\t %.4e %%" % (relative_test_res * 100.0))
    logger.info(
        "Training stars relative RMSE:\t %.4e %%" % (relative_train_res * 100.0)
    )

    return test_res, train_res


def compute_opd_metrics_mccd(
    tf_semiparam_field, GT_tf_semiparam_field, test_pos, train_pos
):
    """Compute the OPD metrics."""

    np_obscurations = np.real(tf_semiparam_field.obscurations.numpy())

    ## For test positions
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(test_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    NP_opd_pred = tf_semiparam_field.tf_NP_mccd_OPD.predict(test_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(test_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy()) * np_obscurations

    # Calculate absolute RMSE values
    test_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Calculate relative RMSE values
    relative_test_opd_rmse = test_opd_rmse / np.sqrt(
        np.mean((GT_opd_maps.numpy() * np_obscurations) ** 2)
    )

    # Print RMSE values
    logger.info("Test stars absolute OPD RMSE:\t %.4e" % test_opd_rmse)
    logger.info(
        "Test stars relative OPD RMSE:\t %.4e %%\n" % (relative_test_opd_rmse * 100.0)
    )

    ## For train part
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(train_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    NP_opd_pred = tf_semiparam_field.tf_NP_mccd_OPD.predict(train_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(train_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy()) * np_obscurations

    # Calculate RMSE values
    train_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Calculate relative RMSE values
    relative_train_opd_rmse = train_opd_rmse / np.sqrt(
        np.mean((GT_opd_maps.numpy() * np_obscurations) ** 2)
    )

    # Print RMSE values
    logger.info("Train stars absolute OPD RMSE:\t %.4e" % train_opd_rmse)
    logger.info(
        "Train stars relative OPD RMSE:\t %.4e %%\n" % (relative_train_opd_rmse * 100.0)
    )

    return test_opd_rmse, train_opd_rmse


def compute_opd_metrics_polymodel(
    tf_semiparam_field, GT_tf_semiparam_field, test_pos, train_pos
):
    """Compute the OPD metrics."""

    np_obscurations = np.real(tf_semiparam_field.obscurations.numpy())

    ## For test positions
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(test_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    NP_opd_pred = tf_semiparam_field.tf_np_poly_opd(test_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(test_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy()) * np_obscurations

    # Calculate RMSE values
    test_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Calculate relative RMSE values
    relative_test_opd_rmse = test_opd_rmse / np.sqrt(
        np.mean((GT_opd_maps.numpy() * np_obscurations) ** 2)
    )

    # Print RMSE values
    logger.info("Test stars OPD RMSE:\t %.4e" % test_opd_rmse)
    logger.info(
        "Test stars relative OPD RMSE:\t %.4e %%\n" % (relative_test_opd_rmse * 100.0)
    )

    ## For train part
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(train_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    NP_opd_pred = tf_semiparam_field.tf_np_poly_opd(train_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(train_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy()) * np_obscurations

    # Calculate RMSE values
    train_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Calculate relative RMSE values
    relative_train_opd_rmse = train_opd_rmse / np.sqrt(
        np.mean((GT_opd_maps.numpy() * np_obscurations) ** 2)
    )

    # Pritn RMSE values
    logger.info("Train stars OPD RMSE:\t %.4e" % train_opd_rmse)
    logger.info(
        "Train stars relative OPD RMSE:\t %.4e %%\n" % (relative_train_opd_rmse * 100.0)
    )

    return test_opd_rmse, train_opd_rmse


def compute_opd_metrics_param_model(
    tf_semiparam_field, GT_tf_semiparam_field, test_pos, train_pos
):
    """Compute the OPD metrics."""

    np_obscurations = np.real(tf_semiparam_field.obscurations.numpy())

    ## For test positions
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(test_pos)
    opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(test_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy()) * np_obscurations

    # Calculate absolute RMSE values
    test_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Calculate relative RMSE values
    relative_test_opd_rmse = test_opd_rmse / np.sqrt(
        np.mean((GT_opd_maps.numpy() * np_obscurations) ** 2)
    )

    # Print RMSE values
    logger.info("Test stars absolute OPD RMSE:\t %.4e" % test_opd_rmse)
    logger.info(
        "Test stars relative OPD RMSE:\t %.4e %%\n" % (relative_test_opd_rmse * 100.0)
    )

    ## For train part
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(train_pos)
    opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(train_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy()) * np_obscurations

    # Calculate RMSE values
    train_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Calculate relative RMSE values
    relative_train_opd_rmse = train_opd_rmse / np.sqrt(
        np.mean((GT_opd_maps.numpy() * np_obscurations) ** 2)
    )

    # Print RMSE values
    logger.info("Train stars absolute OPD RMSE:\t %.4e" % train_opd_rmse)
    logger.info(
        "Train stars relative OPD RMSE:\t %.4e %%\n" % (relative_train_opd_rmse * 100.0)
    )

    return test_opd_rmse, train_opd_rmse


def compute_one_opd_rmse(GT_tf_semiparam_field, tf_semiparam_field, pos, is_poly=False):
    """Compute the OPD map for one position."""

    np_obscurations = np.real(tf_semiparam_field.obscurations.numpy())

    tf_pos = tf.convert_to_tensor(pos, dtype=tf.float32)

    ## For test positions
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(tf_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    if is_poly == False:
        NP_opd_pred = tf_semiparam_field.tf_NP_mccd_OPD.predict(tf_pos)
    else:
        NP_opd_pred = tf_semiparam_field.tf_np_poly_opd(tf_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(tf_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy()) * np_obscurations

    # Calculate RMSE values
    opd_rmse = np.sqrt(np.mean(res_opd**2))

    return opd_rmse


def plot_function(mesh_pos, residual, tf_train_pos, tf_test_pos, title="Error"):
    vmax = np.max(residual)
    vmin = np.min(residual)

    plt.figure(figsize=(12, 8))
    plt.scatter(
        mesh_pos[:, 0],
        mesh_pos[:, 1],
        s=100,
        c=residual.reshape(-1, 1),
        cmap="viridis",
        marker="s",
        vmax=vmax,
        vmin=vmin,
    )
    plt.colorbar()
    plt.scatter(
        tf_train_pos[:, 0],
        tf_train_pos[:, 1],
        c="k",
        marker="*",
        s=10,
        label="Train stars",
    )
    plt.scatter(
        tf_test_pos[:, 0],
        tf_test_pos[:, 1],
        c="r",
        marker="*",
        s=10,
        label="Test stars",
    )
    plt.title(title)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()


def plot_residual_maps(
    GT_tf_semiparam_field,
    tf_semiparam_field,
    simPSF_np,
    train_SEDs,
    tf_train_pos,
    tf_test_pos,
    n_bins_lda=20,
    n_points_per_dim=30,
    is_poly=False,
):
    # Recover teh grid limits
    x_lims = tf_semiparam_field.x_lims
    y_lims = tf_semiparam_field.y_lims

    # Generate mesh of testing positions
    x = np.linspace(x_lims[0], x_lims[1], n_points_per_dim)
    y = np.linspace(y_lims[0], y_lims[1], n_points_per_dim)
    x_pos, y_pos = np.meshgrid(x, y)

    mesh_pos = np.concatenate(
        (x_pos.flatten().reshape(-1, 1), y_pos.flatten().reshape(-1, 1)), axis=1
    )
    tf_mesh_pos = tf.convert_to_tensor(mesh_pos, dtype=tf.float32)

    # Testing the positions
    rec_x_pos = mesh_pos[:, 0].reshape(x_pos.shape)
    rec_y_pos = mesh_pos[:, 1].reshape(y_pos.shape)

    # Get random SED from the training catalog
    SED_random_integers = np.random.choice(
        np.arange(train_SEDs.shape[0]), size=mesh_pos.shape[0], replace=True
    )
    # Build the SED catalog for the testing mesh
    mesh_SEDs = np.array([train_SEDs[_id, :, :] for _id in SED_random_integers])

    # Generate SED data list
    mesh_packed_SED_data = [
        utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
        for _sed in mesh_SEDs
    ]

    # Generate inputs
    tf_mesh_packed_SED_data = tf.convert_to_tensor(
        mesh_packed_SED_data, dtype=tf.float32
    )
    tf_mesh_packed_SED_data = tf.transpose(tf_mesh_packed_SED_data, perm=[0, 2, 1])
    mesh_pred_inputs = [tf_mesh_pos, tf_mesh_packed_SED_data]

    # Predict mesh stars
    model_mesh_preds = tf_semiparam_field.predict(x=mesh_pred_inputs, batch_size=16)
    GT_mesh_preds = GT_tf_semiparam_field.predict(x=mesh_pred_inputs, batch_size=16)

    # Calculate pixel RMSE for each star
    pix_rmse = np.array(
        [
            np.sqrt(np.mean((_GT_pred - _model_pred) ** 2))
            for _GT_pred, _model_pred in zip(GT_mesh_preds, model_mesh_preds)
        ]
    )

    relative_pix_rmse = np.array(
        [
            np.sqrt(np.mean((_GT_pred - _model_pred) ** 2))
            / np.sqrt(np.mean((_GT_pred) ** 2))
            for _GT_pred, _model_pred in zip(GT_mesh_preds, model_mesh_preds)
        ]
    )

    # Plot absolute pixel error
    plot_function(
        mesh_pos, pix_rmse, tf_train_pos, tf_test_pos, title="Absolute pixel error"
    )
    # Plot relative pixel error
    plot_function(
        mesh_pos,
        relative_pix_rmse,
        tf_train_pos,
        tf_test_pos,
        title="Relative pixel error",
    )

    # Compute OPD errors
    opd_rmse = np.array(
        [
            compute_one_opd_rmse(
                GT_tf_semiparam_field, tf_semiparam_field, _pos.reshape(1, -1), is_poly
            )
            for _pos in mesh_pos
        ]
    )

    # Plot absolute pixel error
    plot_function(
        mesh_pos, opd_rmse, tf_train_pos, tf_test_pos, title="Absolute OPD error"
    )


def plot_imgs(mat, cmap="gist_stern", figsize=(20, 20)):
    """Function to plot 2D images of a tensor.
    The Tensor is (batch,xdim,ydim) and the matrix of subplots is
    chosen "intelligently".
    """

    def dp(n, left):  # returns tuple (cost, [factors])
        memo = {}
        if (n, left) in memo:
            return memo[(n, left)]

        if left == 1:
            return (n, [n])

        i = 2
        best = n
        bestTuple = [n]
        while i * i <= n:
            if n % i == 0:
                rem = dp(n / i, left - 1)
                if rem[0] + i < best:
                    best = rem[0] + i
                    bestTuple = [i] + rem[1]
            i += 1

        memo[(n, left)] = (best, bestTuple)
        return memo[(n, left)]

    n_images = mat.shape[0]
    row_col = dp(n_images, 2)[1]
    row_n = int(row_col[0])
    col_n = int(row_col[1])

    plt.figure(figsize=figsize)
    idx = 0

    for _i in range(row_n):
        for _j in range(col_n):
            plt.subplot(row_n, col_n, idx + 1)
            plt.imshow(mat[idx, :, :], cmap=cmap)
            plt.colorbar()
            plt.title("matrix id %d" % idx)

            idx += 1

    plt.show()
