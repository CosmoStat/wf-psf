<!--
A new scriv changelog fragment.

Uncomment the section that is right (remove the HTML comment wrapper).
For top level release notes, leave all the headers commented out.
-->

### New features

- Added `NoiseEstimator.estimate_noise_batch` for batch noise estimation, which is reusable outside training (e.g. SHE validation) since it no longer depends on the training loss/output representation.

### Internal changes

- Moved `NoiseEstimator` from `utils.utils` to a dedicated `utils.noise` module, and initialised it with a default exclusion-window radius (`NoiseEstimator.default_win_rad`).
- Simplified `train_utils.calculate_sample_weights` to `calculate_sample_weights(images, masks=None, ...)`. It no longer needs to know about `loss` or `use_sample_weights`. `general_train_cycle` now resolves the training output representation based on `loss.name`: any name starting with the `masked_` prefix (e.g. `masked_mean_squared_error`) is treated as a masked loss, so new masked loss functions must follow this naming convention to be handled correctly.
