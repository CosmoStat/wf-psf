import numpy as np
import scipy.signal as spsig
import scipy.interpolate as sinterp
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import PIL
import time
from tqdm.notebook import tqdm

# Import wavefront code
import wf_psf.SimPSFToolkit as wf_sim
import wf_psf.psf_models.tf_layers as wf_layers
import wf_psf.psf_models.tf_modules as wf_modules
import wf_psf.psf_models.tf_psf_field as wf_psf_field
import wf_psf.utils.utils as wf_utils
import wf_psf.psf_models.GenPolyFieldPSF as wf_gen


plt.rcParams['figure.figsize'] = (16, 8)
import tensorflow as tf

# Pre-defined colormap
top = mpl.cm.get_cmap('Oranges_r', 128)
bottom = mpl.cm.get_cmap('Blues', 128)
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')


## Prepare inputs

# Zcube_path = '/content/drive/MyDrive/Colab Notebooks/psf_data/Zernike45.mat'
Zcube_path = '/Users/tliaudat/Documents/PhD/codes/WF_PSF/data/PA-zernike-cubes/Zernike45.mat'
Zcube = sio.loadmat(Zcube_path)
zernikes = []

# Decimation factor for Zernike polynomials
decim_f = 4  # Original shape (1024x1024)
n_bins_lda = 15
n_zernikes = 15
batch_size = 16
output_dim=64
d_max=2
x_lims=[0, 1e3]
y_lims=[0, 1e3]
l_rate = 1e-2

for it in range(n_zernikes):
    zernike_map = wf_utils.decimate_im(Zcube['Zpols'][0, it][5], decim_f)
    zernikes.append(zernike_map)

# Now as cubes
np_zernike_cube = np.zeros((len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1]))

for it in range(len(zernikes)):
    np_zernike_cube[it,:,:] = zernikes[it]

np_zernike_cube[np.isnan(np_zernike_cube)] = 0

tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)


# dataset_path = '/content/wf-psf/tf_notebooks/psf_field_dataset/'
# dataset_path = '/content/wf-psf/data/psf_field/'
# dataset_path = '/content/drive/MyDrive/Colab Notebooks/psf_field_dataset/'
dataset_path = '/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-psf/tf_notebooks/psf_field_dataset/'
# dataset_path = '/Users/tliaudat/Documents/PhD/codes/WF_PSF/data/psf_field_datasets/'
# Load the dictionaries
train_dataset = np.load(dataset_path + 'train_dataset_256_bin15_z15_bis.npy', allow_pickle=True)[()]
train_stars = train_dataset['stars']
train_pos = train_dataset['positions']
train_SEDs = train_dataset['SEDs']
train_zernike_coef = train_dataset['zernike_coef']
train_C_poly = train_dataset['C_poly']
train_parameters = train_dataset['parameters']

tf_train_stars = tf.convert_to_tensor(train_stars, dtype=tf.float32)
tf_train_pos = tf.convert_to_tensor(train_pos, dtype=tf.float32)


# Generate initializations

pupil_diameter = 1024 // decim_f

# Prepare np input
simPSF_np = wf_sim.SimPSFToolkit(zernikes, max_order=n_zernikes, pupil_diameter=pupil_diameter)
simPSF_np.gen_random_Z_coeffs(max_order=n_zernikes)
z_coeffs = simPSF_np.normalize_zernikes(simPSF_np.get_z_coeffs(), simPSF_np.max_wfe_rms)
simPSF_np.set_z_coeffs(z_coeffs)
simPSF_np.generate_mono_PSF(lambda_obs=0.7, regen_sample=False)

# Obscurations
obscurations = simPSF_np.generate_pupil_obscurations(N_pix=pupil_diameter, N_filter=2)
tf_obscurations = tf.convert_to_tensor(obscurations, dtype=tf.complex64)

# Initialize the SED data list
packed_SED_data = [wf_utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
                   for _sed in train_SEDs]

# Some parameters

tf_PSF_field_model = wf_psf_field.TF_PSF_field_model(zernike_maps=tf_zernike_cube,
                                        obscurations=tf_obscurations,
                                        batch_size=batch_size,
                                        output_dim=output_dim,
                                        n_zernikes=n_zernikes,
                                        d_max=d_max,
                                        x_lims=x_lims,
                                        y_lims=y_lims)

# Define the model optimisation
loss = tf.keras.losses.MeanSquaredError()

optimizer = tf.keras.optimizers.Adam(
    learning_rate=l_rate, beta_1=0.9, beta_2=0.999,
    epsilon=1e-07, amsgrad=False)
# optimizer = tf.keras.optimizers.SGD(
#     learning_rate=l_rate, momentum=0.9, nesterov=True)

metrics = [tf.keras.metrics.MeanSquaredError()]

tf_PSF_field_model = wf_psf_field.build_PSF_model(tf_PSF_field_model, optimizer=optimizer,
                                                  loss=loss, metrics=metrics)

# Assign the new init variable matrix
# tf_PSF_field_model.tf_poly_Z_field.coeff_mat.assign(init_C)
tf_PSF_field_model.tf_poly_Z_field.coeff_mat.assign(train_C_poly)





# Compare the Pi generating matrix
gen_poly_field = wf_gen.GenPolyFieldPSF(
    sim_psf_toolkit=simPSF_np, d_max=d_max, grid_points=[4, 4], max_order=15,
    x_lims=x_lims, y_lims=y_lims, n_bins=n_bins_lda,
    lim_max_wfe_rms=simPSF_np.max_wfe_rms, verbose=False)

gen_poly_field.C_poly = train_C_poly



# Regenerate the sample
_it = 6
tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
pred_inputs = [train_pos[_it:_it+1,:] , tf_packed_SED_data[_it:_it+1,:,:]]

expected_outputs = tf_train_stars[_it:_it+1,:,:]

predictions = tf_PSF_field_model.predict(x=pred_inputs, batch_size=batch_size)

xv_flat = train_pos[_it,0]
yv_flat = train_pos[_it,1]
SED = train_SEDs[_it,:,:]

np_poly_psf, np_zernikes, np_opd = gen_poly_field.get_poly_PSF(xv_flat, yv_flat, SED)


print('Bye')