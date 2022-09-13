# Inputs of the WaveDiff model

For the model to be trained a dataset is needed. The train and test datasets for the WaveDiff model are detailed below. 

## Input datasets
This model uses two datasets, one for optimising the model and another for testing the model. Both datasets are organised as python dictionaries.

### Dataset keys

```
train_dataset.keys() --> ['stars', 'noisy_stars', 'super_res_stars', 'positions', 'SEDs', 'zernike_coef', 'C_poly', 'parameters']

test_dataset.keys() --> ['stars', 'super_res_stars', 'positions', 'SEDs', 'zernike_coef', 'C_poly', 'parameters']
```

The contents of each entry are detailed below.

## Composition of the datasets

- Stars: it is a `numpy` ndarray containing the low resolution observations of each star. 
    - Shape: `(n_stars, output_dim, output_dim)`.
- Noisy stars: it is a `numpy` ndarray containing noisy version of the low resolution observations of each star.
    - Shape: `(n_stars, output_dim, output_dim)`.
- Super resolved stars: it is a `numpy` ndarray containing the high resolution observations of each star.
    - Shape: `(n_stars, super_out_res, super_out_res)`.
- Positions: it is a `numpy` ndarray containing the $(x,y)$ FOV position of each observation.
    - Shape: `(n_stars, 2)`.
    - Order: `pos[0]` $\rightarrow \, x$, &ensp; `pos[1]` $\rightarrow \, y$
- SEDs: it is a `numpy` ndarray containing the SED asociated to each observed star. For each star, the SED is composed by a set of `n_bins` frequency values, a set of `n_bins` SED values and a set of `n_bins` bin weights proportional to the relative size of each bin with respect to the whole SED bandwidth.
    - Shape: `(n_stars, n_bins, 3)`.
    - Order: `SED[0]` $\rightarrow \, \lambda$,&ensp; `SED[1]` $\rightarrow \, \text{SED}(\lambda)$,&ensp; `SED[2]` $\rightarrow \, \frac{\Delta_{\text{bin}}}{B}$.
- Zernike_coef: it is a `numpy` ndarray containing the ground truth `n_zernike_gt` coefficients of each observed PSF wavefront error.
    - Shape: `(n_stars, n_zernike_gt, 1)`.
- C_poly: it is a `numpy` ndarray containing the ground truth coefficients of the spatial variation polynomials asociated to each Zernike order. Each polynomial has a degree `d_max` and thus the number of coefficients is $n_{\text{max}} = (d_{\text{max}}+1)(d_{\text{max}}+2)/2$.
    - Shape: `(n_zernikes_gt, n_max)`.
- Parameters: it is a python dictionary containing information about the data generation model parameters.
    - Keys: `['d_max', 'max_order', 'x_lims', 'y_lims', 'grid_points', 'n_bins', 'max_wfe_rms', 'oversampling_rate', 'output_Q', 'output_dim', 'LP_filter_length', 'pupil_diameter', 'euclid_obsc', 'n_stars']`.