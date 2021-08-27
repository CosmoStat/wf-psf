#!/usr/bin/env python
# coding: utf-8

import os
import subprocess
import platform

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import mccd

# Sextractor parameters
exec_path = 'sex'
config_files_dir = '/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-psf/method-comparison/config_files/'
dot_sex = config_files_dir + 'default.sex'
dot_param = config_files_dir + 'default.param' 
dot_conv = config_files_dir + 'default.conv'


# Saving parameters
rca_base_path = '/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-psf/method-comparison/compatible-datasets/rca/'
# Temporary folder for Sextractor intermediate products
base_sex_path = '/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-psf/method-comparison/compatible-datasets/psfex/tmp/'

# Input dataset dir
dataset_path = '/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-psf/data/coherent_euclid_dataset/'

# Initialize parameters
loc2glob = mccd.mccd_utils.Loc2Glob_EUCLID_sim()
euclid_CCD_n = loc2glob.ccd_tot  # 36
image_size = 32


separator = '-'
save_extension = '.fits'

catalog_ids = [200, 500, 1000, 2000, 0]
train_cat = [True, True, True, True, False]

for catalog_id, train_bool in zip(catalog_ids, train_cat):

    if train_bool:
        folder_path = 'train/'
        save_name = 'train_stars'
        datafile_path = 'train_Euclid_res_%d_TrainStars_id_001.npy'%(catalog_id)
    else:
        folder_path = 'test/'
        save_name = 'test_stars'
        datafile_path = 'test_Euclid_res_id_001.npy'

    output_dir = (
        '/Users/tliaudat/Documents/PhD/codes/WF_PSF/github/wf-psf/method-comparison/compatible-datasets/psfex/' +
        folder_path
    )

    # Load dataset
    dataset = np.load(dataset_path + datafile_path, allow_pickle=True)[()]
    if train_bool:
        data_stars = dataset['noisy_stars']
    else:
        data_stars = dataset['stars']  # Noiseless stars for the test
    train_pos = dataset['positions']

    # Convert positions and calculate CCD_n
    # Need to switch these coordinates to the global coordinates
    # Each CCD is 4096x4096 and the origin is bottom left corner of the CCD 15.

    # Shift origin to the global positions
    global_pos = train_pos * (4096*6 /1000) - (4096 * 3)
    # Calculate local positions and CCD_n
    local_pos = np.array([
        loc2glob.glob2loc_img_coord(_x_glob, _y_glob) for _x_glob, _y_glob in global_pos
    ])

    ccd_n_list = local_pos[:,0].astype(int)
    local_pos = local_pos[:,1:]

    # CCD list
    CCD_ids = np.arange(euclid_CCD_n)

    CCD_w = loc2glob.x_npix
    CCD_h = loc2glob.y_npix

    half_size = np.floor(image_size/2).astype(int)


    for ccd_it in CCD_ids:
        # Select the CCD

        # Build CCD mask
        ccd_mask = (ccd_n_list == ccd_it)

        # Extract elements from selected CCD
        x_loc = local_pos[ccd_mask, 0]
        y_loc = local_pos[ccd_mask, 1]
        obs_stars = data_stars[ccd_mask, :, :]
        # GT_stars = train_stars[ccd_mask, :, :]

        # Check the borders
        for k in range(x_loc.shape[0]):
            if x_loc[k]-half_size < 0:
                print('Modif border x low. id=%d, x=%f'%(k, x_loc[k]))
                x_loc[k] -= x_loc[k]-half_size
            if x_loc[k]+half_size+1 > CCD_w:
                print('Modif border x high. id=%d, x=%f'%(k, x_loc[k]))
                x_loc[k] -= x_loc[k]+half_size+1-CCD_w
            if y_loc[k]-half_size < 0:
                print('Modif border y low. id=%d, y=%f'%(k, y_loc[k]))
                y_loc[k] -= y_loc[k]-half_size
            if y_loc[k]+half_size+1 > CCD_h:
                print('Modif border y high. id=%d, y=%f'%(k, y_loc[k]))
                y_loc[k] -= y_loc[k]+half_size+1-CCD_h   

        # Number of stars in the CCD
        star_num = x_loc.shape[0]
        
        # Create big pictures
        Im_vig = np.zeros((CCD_w, CCD_h))
        Im_pos = np.zeros((CCD_w, CCD_h))
        
        # No need to add noise as the obs_stars already contain noise
        # And as we are using the Im_pos to force the detection
        # sigma_noise = 1e-10
        # noise = (sigma_noise * np.random.randn(CCD_w * CCD_h)).reshape((CCD_w,CCD_h))
        # Im_vig += noise
        # Im_pos += noise
        
        
        # Add the stars to the big picture
        for k in range(obs_stars.shape[0]):
            Im_vig[
                int(x_loc[k]-half_size):int(x_loc[k]+half_size),
                int(y_loc[k]-half_size):int(y_loc[k]+half_size)
            ] += obs_stars[k,:,:].T 

            Im_pos[
                int(x_loc[k]-1):int(x_loc[k]+2),
                int(y_loc[k]-1):int(y_loc[k]+2)
            ] += 10

            
        # Plot just for testing
        # k=2
        # ll = 15
        # plt.figure()
        # plt.imshow(
        #     Im_pos[int(x_loc[k]-ll):int(x_loc[k]+ll), int(y_loc[k]-ll):int(y_loc[k]+ll)],
        #     cmap='gist_stern'
        # )
        # plt.colorbar()
        # plt.show()

        # plt.figure()
        # plt.imshow(
        #     Im_vig[int(x_loc[k]-ll):int(x_loc[k]+ll), int(y_loc[k]-ll):int(y_loc[k]+ll)],
        #     cmap='gist_stern'
        # )
        # plt.colorbar()
        # plt.show()


        # Define the header and save the big images
        hdr = fits.Header()
        hdr['OWNER'] = 'WF_PSF package'

        overwrite = False

        data_im = Im_vig.T
        data_pos = Im_pos.T

        sex_im_savepath = base_sex_path + '/tmp_ccd_image.fits'
        sex_pos_savepath = base_sex_path + '/tmp_ccd_pos.fits'


        if os.path.isfile(sex_im_savepath):
            os.remove(sex_im_savepath)
        if os.path.isfile(sex_pos_savepath):
            os.remove(sex_pos_savepath)    

        fits.PrimaryHDU(data_im, hdr).writeto(sex_im_savepath, overwrite=overwrite)
        fits.PrimaryHDU(data_pos, hdr).writeto(sex_pos_savepath, overwrite=overwrite)

        
        # Prepare the command an launch sextractor
        detection_image_path = sex_pos_savepath
        measurement_image = sex_im_savepath
        output_file_path = output_dir + '%s_psfex-%07d-%02d.fits'%(save_name, catalog_id, ccd_it)
        
        # Base arguments for SExtractor
        command_line_base = (
            '{0} {1},{2} -c {3} -PARAMETERS_NAME {4} -FILTER_NAME {5} '
            '-CATALOG_NAME {6} -WEIGHT_TYPE None -CHECKIMAGE_TYPE NONE -CHECKIMAGE_NAME none'.format(
                exec_path,
                detection_image_path,
                measurement_image,
                dot_sex,
                dot_param,
                dot_conv,
                output_file_path
            )
        )
        # For >=python3.7
        if platform.python_version_tuple()[0] ==3 and platform.python_version_tuple()[1] >= 7:
            process_output = subprocess.run(command_line_base, shell=True, check=True, capture_output=True)
            # Uncomment to print sextractor output
            # print('STDERR:\n', process_output.stderr.decode("utf-8"))
            # print('STDOUT:\n', process_output.stdout.decode("utf-8"))
        elif platform.python_version_tuple()[0] ==3 and platform.python_version_tuple()[1] < 7:
            # For python3.6
            process_output = subprocess.run(command_line_base, shell=True, check=True)

        sexcat = fits.open(output_file_path, memmap=False)
        print(
            'Catalog %s, CCD %d, \t Detected stars: %d,\t Diff (inserted - detected) = %d'%(
                catalog_id,
                ccd_it,
                sexcat[2].data['VIGNET'].shape[0],
                star_num - sexcat[2].data['VIGNET'].shape[0]
            )
        )
        sexcat.close()
        
    if os.path.isfile(sex_im_savepath):
        os.remove(sex_im_savepath)
    if os.path.isfile(sex_pos_savepath):
        os.remove(sex_pos_savepath)    
