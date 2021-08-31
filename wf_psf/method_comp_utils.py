
import os
import numpy as np
from astropy.io import fits
import mccd
import galsim as gs
import tensorflow as tf
from wf_psf import utils as utils
from wf_psf import SimPSFToolkit as SimPSFToolkit
from wf_psf import tf_psf_field as psf_field

def interpsfex(dotpsfpath, pos):
    """Use PSFEx generated model to perform spatial PSF interpolation.
        Parameters
        ----------
        dotpsfpath : string
            Path to .psf file (PSFEx output).
        pos : np.ndaray
            Positions where the PSF model should be evaluated.
        Returns
        -------
        PSFs : np.ndarray
            Each row is the PSF imagette at the corresponding asked position.
    """

    if not os.path.exists(dotpsfpath):
        return None

    # read PSF model and extract basis and polynomial degree and scale position
    PSF_model = fits.open(dotpsfpath)[1]

    PSF_basis = np.array(PSF_model.data)[0][0]
    try:
        deg = PSF_model.header['POLDEG1']
    except KeyError:
        # constant PSF model
        return PSF_basis[0, :, :]

    # scale coordinates
    x_interp, x_scale = (PSF_model.header['POLZERO1'],
                         PSF_model.header['POLSCAL1'])
    y_interp, y_scale = (PSF_model.header['POLZERO2'],
                         PSF_model.header['POLSCAL2'])
    xs, ys = (pos[:, 0] - x_interp) / x_scale, (pos[:, 1] - y_interp) / y_scale

    # compute polynomial coefficients
    coeffs = np.array([[x**i for i in range(deg+1)] for x in xs])
    cross_coeffs = np.array([np.concatenate([[(x ** j) * (y ** i)
                                              for j in range(deg - i + 1)]
                                             for i in range(1, deg + 1)])
                             for x, y in zip(xs, ys)])
    coeffs = np.hstack((coeffs, cross_coeffs))

    # compute interpolated PSF
    PSFs = np.array([np.sum([coeff * atom for coeff, atom in
                     zip(coeffs_posi, PSF_basis)], axis=0)
                     for coeffs_posi in coeffs])

    return PSFs


def validation_stars(dotpsfpath, test_stars, test_pos, psf_size=1.25):
    """ Match PSF model to stars - in flux, shift and pixel sampling - for validation tests.
    Returns both the matched PSFs' stamps.

    Parameters
    ----------
    dotpsfpath: str
        Path to the .psf model output of PSFEx.
    test_stars: np.ndarray
        Star stamps to be used for comparison with the PSF model.
    test_pos: np.ndarray
        Their corresponding positions.
    psf_size: float
        PSF size in sigma format.
    """
    
    sigmas = np.ones((test_pos.shape[0],)) * psf_size
        
    cents = [
        mccd.utils.CentroidEstimator(test_stars[it, :, :], sig=sigmas[it])
        for it in range(test_stars.shape[0])
    ]
    # Calculate shifts
    test_shifts = np.array([ce.return_shifts() for ce in cents])

    # Interpolate PSFs
    interp_psfs = interpsfex(dotpsfpath, test_pos)
    
    # Estimate shift kernels
    lanc_rad = np.ceil(3. * np.max(sigmas)).astype(int)
    shift_kernels, _ = mccd.utils.shift_ker_stack(
        test_shifts,
        upfact=1,
        lanc_rad=lanc_rad
    )
    
    # Degrade PSFs
    interp_psfs = np.array([
        mccd.utils.degradation_op(
            interp_psfs[j, :, :],
            shift_kernels[:, :, j],
            D=1
        )
        for j in range(test_pos.shape[0])
    ])

    # Optimised fulx matching
    norm_factor = np.array([
        np.sum(_star * _psf) / np.sum(_psf * _psf)
        for _star, _psf in zip(test_stars, interp_psfs)
    ]).reshape(-1, 1, 1)
    interp_psfs *= norm_factor
            
    return interp_psfs


def calc_shapes(matched_psfs, test_stars):
    # Measure shapes of the reconstructions
    pred_moments = [gs.hsm.FindAdaptiveMom(gs.Image(_pred), strict=False) for _pred in matched_psfs]

    # Measure shapes of the reconstructions
    GT_pred_moments = [gs.hsm.FindAdaptiveMom(gs.Image(_pred), strict=False) for _pred in test_stars]

    pred_e1_HSM, pred_e2_HSM, pred_R2_HSM = [], [], []
    GT_pred_e1_HSM, GT_pred_e2_HSM, GT_pred_R2_HSM = [], [], []

    for it in range(len(test_stars)):
        # Only save shape if both measurements do not raise errors
        if pred_moments[it].moments_status == 0 and GT_pred_moments[it].moments_status == 0:

            pred_e1_HSM.append(pred_moments[it].observed_shape.g1)
            pred_e2_HSM.append(pred_moments[it].observed_shape.g2)
            pred_R2_HSM.append(2*(pred_moments[it].moments_sigma**2))

            GT_pred_e1_HSM.append(GT_pred_moments[it].observed_shape.g1)
            GT_pred_e2_HSM.append(GT_pred_moments[it].observed_shape.g2)
            GT_pred_R2_HSM.append(2*(GT_pred_moments[it].moments_sigma**2))

    print(
        'Total number of stars: %d\t Number of shape measurements problems %d.'%(
            len(test_stars), len(test_stars) - len(pred_e1_HSM)
        )
    )
    pred_e1_HSM = np.array(pred_e1_HSM)
    pred_e2_HSM = np.array(pred_e2_HSM)
    pred_R2_HSM = np.array(pred_R2_HSM)

    GT_pred_e1_HSM = np.array(GT_pred_e1_HSM)
    GT_pred_e2_HSM = np.array(GT_pred_e2_HSM)
    GT_pred_R2_HSM = np.array(GT_pred_R2_HSM)
    
    return (pred_e1_HSM, pred_e2_HSM, pred_R2_HSM), (GT_pred_e1_HSM, GT_pred_e2_HSM, GT_pred_R2_HSM)


def gen_GT_wf_model(test_wf_file_path, pred_output_Q=1, pred_output_dim=64):
    r""" Generate the ground truth model and output test PSF ar required resolution. 

    If `pred_output_Q=1` the resolution will be 3 times the one of Euclid.
    """
    # Load dataset
    wf_test_dataset = np.load(test_wf_file_path, allow_pickle=True)[()]
    
    # Extract parameters from the wf test dataset
    wf_test_params = wf_test_dataset['parameters']
    wf_test_C_poly = wf_test_dataset['C_poly']
    wf_test_pos = wf_test_dataset['positions']
    tf_test_pos = tf.convert_to_tensor(wf_test_pos, dtype=tf.float32)
    wf_test_SEDs = wf_test_dataset['SEDs']

    # Generate GT model
    batch_size = 16

    # Generate Zernike maps
    zernikes = utils.zernike_generator(
        n_zernikes=wf_test_params['max_order'],
        wfe_dim=wf_test_params['pupil_diameter']
    )

    ## Generate initializations
    # Prepare np input
    simPSF_np = SimPSFToolkit(
        zernikes,
        max_order=wf_test_params['max_order'],
        pupil_diameter=wf_test_params['pupil_diameter'],
        output_dim=wf_test_params['output_dim'],
        oversampling_rate=wf_test_params['oversampling_rate'],
        output_Q=wf_test_params['output_Q']
    )
    simPSF_np.gen_random_Z_coeffs(max_order=wf_test_params['max_order'])
    z_coeffs = simPSF_np.normalize_zernikes(simPSF_np.get_z_coeffs(), simPSF_np.max_wfe_rms)
    simPSF_np.set_z_coeffs(z_coeffs)
    simPSF_np.generate_mono_PSF(lambda_obs=0.7, regen_sample=False)
    # Obscurations
    obscurations = simPSF_np.generate_pupil_obscurations(
        N_pix=wf_test_params['pupil_diameter'],
        N_filter=wf_test_params['LP_filter_length']
    )
    tf_obscurations = tf.convert_to_tensor(obscurations, dtype=tf.complex64)


    ## Prepare ground truth model
    # Now Zernike's as cubes
    np_zernike_cube = np.zeros((len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1]))
    for it in range(len(zernikes)):
        np_zernike_cube[it,:,:] = zernikes[it]

    np_zernike_cube[np.isnan(np_zernike_cube)] = 0
    tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)

    # Initialize the model
    GT_tf_semiparam_field = psf_field.TF_SemiParam_field(
        zernike_maps=tf_zernike_cube,
        obscurations=tf_obscurations,
        batch_size=batch_size,
        output_Q=wf_test_params['output_Q'],
        d_max_nonparam=2,
        output_dim=wf_test_params['output_dim'],
        n_zernikes=wf_test_params['max_order'],
        d_max=wf_test_params['d_max'],
        x_lims=wf_test_params['x_lims'],
        y_lims=wf_test_params['y_lims']
    )

    # For the Ground truth model
    GT_tf_semiparam_field.tf_poly_Z_field.assign_coeff_matrix(wf_test_C_poly)
    _ = GT_tf_semiparam_field.tf_np_poly_opd.alpha_mat.assign(
        tf.zeros_like(GT_tf_semiparam_field.tf_np_poly_opd.alpha_mat)
    )

    # Set required output_Q

    GT_tf_semiparam_field.set_output_Q(output_Q=pred_output_Q, output_dim=pred_output_dim)

    GT_tf_semiparam_field = psf_field.build_PSF_model(GT_tf_semiparam_field)

    packed_SED_data = [
        utils.generate_packed_elems(
            _sed,
            simPSF_np,
            n_bins=wf_test_params['n_bins']
        )
        for _sed in wf_test_SEDs
    ]

    # Prepare inputs
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
    pred_inputs = [tf_test_pos , tf_packed_SED_data]

    # Ground Truth model
    GT_predictions = GT_tf_semiparam_field.predict(x=pred_inputs, batch_size=batch_size)

    return GT_predictions, wf_test_pos

def shape_pix_metrics(exp_psfs, exp_stars):

    # Calculate residuals
    residuals = np.sqrt(np.mean((exp_psfs - exp_stars)**2, axis=(1,2)))
    GT_star_mean = np.sqrt(np.mean((exp_stars)**2, axis=(1,2)))

    # RMSE calculations
    rmse = np.mean(residuals)
    rel_rmse = 100. * np.mean(residuals/GT_star_mean)

    # STD calculations
    pix_rmse_std = np.std(residuals)
    rel_pix_rmse_std = 100. * np.std(residuals/GT_star_mean)

    ## Shape error
    # Calculate shapes
    psf_shapes, star_shapes = calc_shapes(exp_psfs, exp_stars)
    # Extract results
    pred_e1_HSM, pred_e2_HSM, pred_R2_HSM = psf_shapes[0], psf_shapes[1], psf_shapes[2]
    GT_pred_e1_HSM, GT_pred_e2_HSM, GT_pred_R2_HSM = star_shapes[0], star_shapes[1], star_shapes[2]

    # e1
    e1_res = GT_pred_e1_HSM - pred_e1_HSM
    e1_res_rel = (GT_pred_e1_HSM - pred_e1_HSM) / GT_pred_e1_HSM

    rmse_e1 = np.sqrt(np.mean(e1_res**2))
    rel_rmse_e1 = 100.* np.sqrt(np.mean(e1_res_rel**2))
    std_rmse_e1 = np.std(e1_res)
    std_rel_rmse_e1 = 100. * np.std(e1_res_rel)

    # e2
    e2_res = GT_pred_e2_HSM - pred_e2_HSM
    e2_res_rel = (GT_pred_e2_HSM - pred_e2_HSM) / GT_pred_e2_HSM

    rmse_e2 = np.sqrt(np.mean(e2_res**2))
    rel_rmse_e2 = 100.* np.sqrt(np.mean(e2_res_rel**2))
    std_rmse_e2 = np.std(e2_res)
    std_rel_rmse_e2 = 100. * np.std(e2_res_rel)

    # R2
    R2_res = GT_pred_R2_HSM - pred_R2_HSM

    rmse_R2_meanR2 = np.sqrt(np.mean(R2_res**2))/np.mean(GT_pred_R2_HSM)
    std_rmse_R2_meanR2 = np.std(R2_res/GT_pred_R2_HSM)

    # Save the metrics
    metrics_dict = {
        'pix_rmse': rmse, 'pix_rel_rmse': rel_rmse,
        'pix_rmse_std': pix_rmse_std, 'rel_pix_rmse_std': rel_pix_rmse_std,
        'pred_e1_HSM': pred_e1_HSM, 'pred_e2_HSM': pred_e2_HSM, 'pred_R2_HSM': pred_R2_HSM,
        'GT_pred_e1_HSM': GT_pred_e1_HSM, 'GT_ped_e2_HSM': GT_pred_e2_HSM, 'GT_pred_R2_HSM': GT_pred_R2_HSM,
        'rmse_e1': rmse_e1, 'std_rmse_e1': std_rmse_e1,
        'rel_rmse_e1': rel_rmse_e1, 'std_rel_rmse_e1': std_rel_rmse_e1,
        'rmse_e2': rmse_e2, 'std_rmse_e2': std_rmse_e2,
        'rel_rmse_e2': rel_rmse_e2, 'std_rel_rmse_e2': std_rel_rmse_e2,
        'rmse_R2_meanR2': rmse_R2_meanR2, 'std_rmse_R2_meanR2': std_rmse_R2_meanR2,
    }

    return metrics_dict