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
    - Order: `SED[i,:,0]` $\rightarrow \, \lambda$,&ensp; `SED[i,:,1]` $\rightarrow \, \text{SED}(\lambda)$,&ensp; `SED[i,:,2]` $\rightarrow \, \frac{\Delta_{\text{bin}}}{B}$.
- Zernike_coef: it is a `numpy` ndarray containing the ground truth `n_zernike_gt` coefficients of each observed PSF wavefront error.
    - Shape: `(n_stars, n_zernike_gt, 1)`.
- C_poly: it is a `numpy` ndarray containing the ground truth coefficients of the spatial variation polynomials asociated to each Zernike order. Each polynomial has a degree `d_max` and thus the number of coefficients is $n_{\text{max}} = (d_{\text{max}}+1)(d_{\text{max}}+2)/2$.
    - Shape: `(n_zernikes_gt, n_max)`.
- Parameters: it is a python dictionary containing information about the data generation model parameters.
    - Keys: `['d_max', 'max_order', 'x_lims', 'y_lims', 'grid_points', 'n_bins', 'max_wfe_rms', 'oversampling_rate', 'output_Q', 'output_dim', 'LP_filter_length', 'pupil_diameter', 'euclid_obsc', 'n_stars']`.


## Tensorflow inputs and outputs

From each dataset, the inputs and outputs for the model will be generated. The inputs of the model, $X$, are the star positions and the SEDs. The outputs of the model, $Y$, are the noisy stars which will be constrasted with the model's predictions $\hat{Y}$ to compute the loss $\mathcal{L}(Y, \hat{Y})$.

### Inputs
As mentioned before the input will contain two tensors, the first one containing the $(x,y)$ FOV positions and the second one containing the SED for each observation.

```
inputs = [tf_pos, tf_packed_SED_data]
```

- `tf_pos` is a tensor of shape `([n_stars,2])`. The order of the $(x,y)$ positions are the same as before. 

- `tf_packed_SED_data` is a tensor of shape `([n_stars,n_bins,3])`. For each observation $i$ this tensor  contains three `n_bins`-point SED features. The first one contains each integer `feasible_N` needed for the upscaling of the wavefront error in order to compute the PSF at each `feasible_wv` (listed in the second position). Finaly in the third position the SED values are listed for each `feasible_wv`. 
    - Notes
        - The second dimention of the packed SED data can change its lenght if SED interpolation is performed. The interpolated SED will then have the following number of samples:

            $n_{\text{bins\_interp}} = n_{\text{bins}} \times (\texttt{interp\_pts\_per\_bin} + 1) \; \pm \; 1$,

            if extrapolation is performed, a point is added to the SED, thus it corresponts to the $+1$ case. If exptrapolation is not performed the $-1$ case is considered.  

        - The `feasible_N` asociated to each `feasible_wv` is calculated using the [`SimPSFToolkit.feasible_N()`](https://github.com/tobias-liaudat/wf-psf/blob/main/wf_psf/SimPSFToolkit.py#L636) method. 
        - The packed SED date is generated with the [generate_packed_elems()](https://github.com/tobias-liaudat/wf-psf/blob/main/wf_psf/utils.py#L44) method of the `wf_utils` class detailed in the `utils.py` file.
        - The packed SED data must be converted to tensor to be an input of the model (a column permutation also takes place):
        ```
        tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
        tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
        ```
        
### Outputs
The output of the WaveDiff model contains the noisy low-resolution star observations. 
```
outputs = [tf_noisy_train_stars]
```

-  `tf_noisy_train_stars` is a tensor of shape `([n_stars, output_dim, output_dim])` containing the noisy observation of each star. 
    - Notes
        - The train stars must be converted to tensor in order to be used by the model. 
        ```
        tf_noisy_train_stars = tf.convert_to_tensor(train_dataset['noisy_stars'], dtype=tf.float32)
        ```