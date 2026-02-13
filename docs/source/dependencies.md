# Dependencies

Third-party software packages required by WaveDiff are installed automatically (see [Installation](installation.md)).

## Python Dependencies

| Package Name | References                                        |
|--------------|---------------------------------------------------|
| [numpy](https://numpy.org/)                  | {cite:t}`harris:20` |
| [scipy](https://scipy.org)                   | {cite:t}`SciPy-NMeth:20` |
| [keras](https://keras.io)                    | {cite:t}`chollet:2015keras`|
| [tensorflow](https://www.tensorflow.org)     | {cite:t}`tensorflow:15` |
| [tensorflow-estimator](https://www.tensorflow.org/api_docs/python/tf/estimator) |{cite:t}`tensorflow:15` |
| [zernike](https://github.com/jacopoantonello/zernike) |  {cite:t}`Antonello:15` |
| [opencv-python](https://docs.opencv.org/4.x/index.html) | {cite:t}`opencv_library:08`  |
| [pillow](https://pillow.readthedocs.io/en/stable/) | {cite:t}`clark:15`         |
| [galsim](http://galsim-developers.github.io/GalSim/_build/html/index.html#) |    {cite:t}`rowe:15`        |
| [astropy](https://www.astropy.org) |   {cite:t}`astropy:13,astropy:18`, <br>{cite:t}`astropy:22`       |
| [matplotlib](https://matplotlib.org) |   {cite:t}`Hunter:07`    |
| [pandas](https://pandas.pydata.org)  | {cite:t}`mckinney:2010pandas`   |
| [seaborn](https://seaborn.pydata.org)   |    {cite:t}`Waskom:21`        |

## Optional Dependencies

Some features in WaveDiff rely on optional third-party packages that are **not required for standard training and evaluation workflows**.

### TensorFlow Addons (Optional)

| Package Name | Purpose |
|--------------|---------|
| [tensorflow-addons](https://www.tensorflow.org/addons) | Optional optimizers (e.g. RectifiedAdam) |

Starting with WaveDiff **v3.1.0**, `tensorflow-addons` is no longer a required dependency, as TensorFlow Addons reached end-of-life in May 2024.

- By default, WaveDiff uses standard Keras/TensorFlow optimizers (e.g. `Adam`)
- TensorFlow Addons is only imported **at runtime** if explicitly requested in the configuration
- If a TensorFlow Addons optimizer is selected and the package is not installed, WaveDiff will raise a clear runtime error

To use TensorFlow Addons optimizers, install manually:

```bash
pip install tensorflow-addons
```