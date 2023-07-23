import os
import numpy as np
from astropy.io import fits
import galsim as gs
import warnings

try:
    import mccd
except:
    warnings.warn(
        "Warning! Problem importing MCCD package."
        "Check out that MCCD dependencies are correctly installed."
    )


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
        deg = PSF_model.header["POLDEG1"]
    except KeyError:
        # constant PSF model
        return PSF_basis[0, :, :]

    # scale coordinates
    x_interp, x_scale = (PSF_model.header["POLZERO1"], PSF_model.header["POLSCAL1"])
    y_interp, y_scale = (PSF_model.header["POLZERO2"], PSF_model.header["POLSCAL2"])
    xs, ys = (pos[:, 0] - x_interp) / x_scale, (pos[:, 1] - y_interp) / y_scale

    # compute polynomial coefficients
    coeffs = np.array([[x**i for i in range(deg + 1)] for x in xs])
    cross_coeffs = np.array(
        [
            np.concatenate(
                [
                    [(x**j) * (y**i) for j in range(deg - i + 1)]
                    for i in range(1, deg + 1)
                ]
            )
            for x, y in zip(xs, ys)
        ]
    )
    coeffs = np.hstack((coeffs, cross_coeffs))

    # compute interpolated PSF
    PSFs = np.array(
        [
            np.sum(
                [coeff * atom for coeff, atom in zip(coeffs_posi, PSF_basis)], axis=0
            )
            for coeffs_posi in coeffs
        ]
    )

    return PSFs


def match_psfs(interp_psfs, test_stars, psf_size=1.25):
    """Match PSF model to stars - in flux, shift and pixel sampling - for validation tests.
    Returns both the matched PSFs' stamps.

    Parameters
    ----------
    interp_psfs: np.ndarray
        PSFs stamps to be matched.
    test_stars: np.ndarray
        Star stamps to be used for comparison with the PSF model.
    psf_size: float
        PSF size in sigma format.
    """

    sigmas = np.ones((test_stars.shape[0],)) * psf_size

    cents = [
        mccd.utils.CentroidEstimator(test_stars[it, :, :], sig=sigmas[it])
        for it in range(test_stars.shape[0])
    ]
    # Calculate shifts
    test_shifts = np.array([ce.return_shifts() for ce in cents])

    # Estimate shift kernels
    lanc_rad = np.ceil(3.0 * np.max(sigmas)).astype(int)
    shift_kernels, _ = mccd.utils.shift_ker_stack(
        test_shifts, upfact=1, lanc_rad=lanc_rad
    )

    # Degrade PSFs
    interp_psfs = np.array(
        [
            mccd.utils.degradation_op(interp_psfs[j, :, :], shift_kernels[:, :, j], D=1)
            for j in range(test_stars.shape[0])
        ]
    )

    # Optimised fulx matching
    norm_factor = np.array(
        [
            np.sum(_star * _psf) / np.sum(_psf * _psf)
            for _star, _psf in zip(test_stars, interp_psfs)
        ]
    ).reshape(-1, 1, 1)
    interp_psfs *= norm_factor

    return interp_psfs


def crop_at_center(img, output_size):
    """ " Crop image to a desired output size."""
    x = img.shape[0]
    y = img.shape[1]
    startx = x // 2 - (output_size[0] // 2)
    starty = y // 2 - (output_size[1] // 2)
    return img[startx : startx + output_size[0], starty : starty + output_size[1]]


def calc_shapes(matched_psfs, test_stars):
    # Measure shapes of the reconstructions
    pred_moments = [
        gs.hsm.FindAdaptiveMom(gs.Image(_pred), strict=False) for _pred in matched_psfs
    ]

    # Measure shapes of the reconstructions
    GT_pred_moments = [
        gs.hsm.FindAdaptiveMom(gs.Image(_pred), strict=False) for _pred in test_stars
    ]

    pred_e1_HSM, pred_e2_HSM, pred_R2_HSM = [], [], []
    GT_pred_e1_HSM, GT_pred_e2_HSM, GT_pred_R2_HSM = [], [], []

    for it in range(len(test_stars)):
        # Only save shape if both measurements do not raise errors
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

    logger.info(
        "Total number of stars: %d\t Number of shape measurements problems %d."
        % (len(test_stars), len(test_stars) - len(pred_e1_HSM))
    )
    pred_e1_HSM = np.array(pred_e1_HSM)
    pred_e2_HSM = np.array(pred_e2_HSM)
    pred_R2_HSM = np.array(pred_R2_HSM)

    GT_pred_e1_HSM = np.array(GT_pred_e1_HSM)
    GT_pred_e2_HSM = np.array(GT_pred_e2_HSM)
    GT_pred_R2_HSM = np.array(GT_pred_R2_HSM)

    return (pred_e1_HSM, pred_e2_HSM, pred_R2_HSM), (
        GT_pred_e1_HSM,
        GT_pred_e2_HSM,
        GT_pred_R2_HSM,
    )


def shape_pix_metrics(exp_psfs, exp_stars):
    # Calculate residuals
    residuals = np.sqrt(np.mean((exp_psfs - exp_stars) ** 2, axis=(1, 2)))
    GT_star_mean = np.sqrt(np.mean((exp_stars) ** 2, axis=(1, 2)))

    # RMSE calculations
    rmse = np.mean(residuals)
    rel_rmse = 100.0 * np.mean(residuals / GT_star_mean)

    # STD calculations
    pix_rmse_std = np.std(residuals)
    rel_pix_rmse_std = 100.0 * np.std(residuals / GT_star_mean)

    ## Shape error
    # Calculate shapes
    psf_shapes, star_shapes = calc_shapes(exp_psfs, exp_stars)
    # Extract results
    pred_e1_HSM, pred_e2_HSM, pred_R2_HSM = psf_shapes[0], psf_shapes[1], psf_shapes[2]
    GT_pred_e1_HSM, GT_pred_e2_HSM, GT_pred_R2_HSM = (
        star_shapes[0],
        star_shapes[1],
        star_shapes[2],
    )

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

    # Save the metrics
    metrics_dict = {
        "pix_rmse": rmse,
        "pix_rel_rmse": rel_rmse,
        "pix_rmse_std": pix_rmse_std,
        "rel_pix_rmse_std": rel_pix_rmse_std,
        "pred_e1_HSM": pred_e1_HSM,
        "pred_e2_HSM": pred_e2_HSM,
        "pred_R2_HSM": pred_R2_HSM,
        "GT_pred_e1_HSM": GT_pred_e1_HSM,
        "GT_pred_e2_HSM": GT_pred_e2_HSM,
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
    }

    return metrics_dict
