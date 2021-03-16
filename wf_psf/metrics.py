import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



def compute_one_opd_rmse(GT_tf_semiparam_field, tf_semiparam_field, pos):
    """ Compute the OPD map for one position!. """

    np_obscurations = np.real(tf_semiparam_field.obscurations.numpy())

    tf_pos = tf.convert_to_tensor(pos, dtype=tf.float32)

    ## For test positions
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(tf_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    NP_opd_pred =  tf_semiparam_field.tf_NP_mccd_OPD.predict(tf_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(tf_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy())*np_obscurations

    # Calculate RMSE values
    opd_rmse = np.sqrt(np.mean(res_opd**2))

    return opd_rmse


def plot_function(mesh_pos, residual, tf_train_pos, tf_test_pos, title='Error'):
    vmax = np.max(residual)
    vmin = np.min(residual)

    plt.figure(figsize=(12,8))
    plt.scatter(mesh_pos[:,0], mesh_pos[:,1], s=100, c=residual.reshape(-1,1), cmap='viridis', marker='s', vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.scatter(tf_train_pos[:,0], tf_train_pos[:,1], c='k', marker='*', s=10, label='Train stars')
    plt.scatter(tf_test_pos[:,0], tf_test_pos[:,1], c='r', marker='*', s=10, label='Test stars')
    plt.title(title)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.show()


def plot_residual_maps(GT_tf_semiparam_field, tf_semiparam_field, simPSF_np, train_SEDs,
                       tf_train_pos, tf_test_pos, n_bins_lda=20, n_points_per_dim=30):

    # Recover teh grid limits
    x_lims = tf_semiparam_field.x_lims
    y_lims = tf_semiparam_field.y_lims

    # Generate mesh of testing positions
    x = np.linspace(x_lims[0], x_lims[1], n_points_per_dim)
    y = np.linspace(y_lims[0], y_lims[1], n_points_per_dim)
    x_pos, y_pos = np.meshgrid(x, y)

    mesh_pos = np.concatenate((x_pos.flatten().reshape(-1,1), y_pos.flatten().reshape(-1,1)), axis=1)
    tf_mesh_pos = tf.convert_to_tensor(mesh_pos, dtype=tf.float32)

    # Testing the positions
    rec_x_pos = mesh_pos[:,0].reshape(x_pos.shape)
    rec_y_pos = mesh_pos[:,1].reshape(y_pos.shape)

    # Get random SED from the training catalog
    SED_random_integers = np.random.choice(np.arange(train_SEDs.shape[0]), size=mesh_pos.shape[0], replace=True)
    # Build the SED catalog for the testing mesh
    mesh_SEDs = np.array([train_SEDs[_id,:,:] for _id in SED_random_integers])


    # Generate SED data list
    mesh_packed_SED_data = [wf_utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
                            for _sed in mesh_SEDs]

    # Generate inputs
    tf_mesh_packed_SED_data = tf.convert_to_tensor(mesh_packed_SED_data, dtype=tf.float32)
    tf_mesh_packed_SED_data = tf.transpose(tf_mesh_packed_SED_data, perm=[0, 2, 1])
    mesh_pred_inputs = [tf_mesh_pos , tf_mesh_packed_SED_data]


    # Predict mesh stars
    model_mesh_preds = tf_semiparam_field.predict(x=mesh_pred_inputs, batch_size=16)
    GT_mesh_preds = GT_tf_semiparam_field.predict(x=mesh_pred_inputs, batch_size=16)

    # Calculate pixel RMSE for each star
    pix_rmse = np.array([np.sqrt(np.mean((_GT_pred-_model_pred)**2))
                            for _GT_pred, _model_pred  in zip(GT_mesh_preds, model_mesh_preds)])

    relative_pix_rmse = np.array([np.sqrt(np.mean((_GT_pred-_model_pred)**2))/np.sqrt(np.mean((_GT_pred)**2))
                                    for _GT_pred, _model_pred  in zip(GT_mesh_preds, model_mesh_preds)])

    # Plot absolute pixel error
    plot_function(mesh_pos, pix_rmse, tf_train_pos, tf_test_pos, title='Absolute pixel error')
    # Plot relative pixel error
    plot_function(mesh_pos, relative_pix_rmse, tf_train_pos, tf_test_pos, title='Relative pixel error')

    # Compute OPD errors
    opd_rmse = np.array([compute_one_opd_rmse(GT_tf_semiparam_field, tf_semiparam_field, _pos.reshape(1,-1)) for _pos in mesh_pos])

    # Plot absolute pixel error
    plot_function(mesh_pos, opd_rmse, tf_train_pos, tf_test_pos, title='Absolute OPD error')


def compute_metrics(tf_semiparam_field, simPSF_np, test_SEDs, train_SEDs,
                    tf_test_pos, tf_train_pos, tf_test_stars, tf_train_stars,
                    n_bins_lda):
    # Generate SED data list
    test_packed_SED_data = [wf_utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
                            for _sed in test_SEDs]

    tf_test_packed_SED_data = tf.convert_to_tensor(test_packed_SED_data, dtype=tf.float32)
    tf_test_packed_SED_data = tf.transpose(tf_test_packed_SED_data, perm=[0, 2, 1])
    test_pred_inputs = [tf_test_pos , tf_test_packed_SED_data]
    test_predictions = tf_semiparam_field.predict(x=test_pred_inputs, batch_size=batch_size)


    # Initialize the SED data list
    packed_SED_data = [wf_utils.generate_packed_elems(_sed, simPSF_np, n_bins=n_bins_lda)
                    for _sed in train_SEDs]
    # First estimate the stars for the observations
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])
    inputs = [tf_train_pos, tf_packed_SED_data]
    train_predictions = tf_semiparam_field.predict(x=inputs, batch_size=batch_size)

    # Calculate RMSE values
    test_res = np.sqrt(np.mean((tf_test_stars - test_predictions)**2))
    train_res = np.sqrt(np.mean((tf_train_stars - train_predictions)**2))

    # Pritn RMSE values
    print('Test stars RMSE:\t %.4e'%test_res)
    print('Training stars RMSE:\t %.4e'%train_res)


    return test_res, train_res


def compute_opd_metrics(tf_semiparam_field, GT_tf_semiparam_field, test_pos, train_pos):
    """ Compute the OPD metrics. """

    np_obscurations = np.real(tf_semiparam_field.obscurations.numpy())

    ## For test positions
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(test_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    NP_opd_pred =  tf_semiparam_field.tf_NP_mccd_OPD.predict(test_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(test_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy())*np_obscurations

    # Calculate RMSE values
    test_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Pritn RMSE values
    print('Test stars OPD RMSE:\t %.4e'%test_opd_rmse)


    ## For train part
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(train_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)
    # Non-Param part
    NP_opd_pred =  tf_semiparam_field.tf_NP_mccd_OPD.predict(train_pos)
    # OPD prediction
    opd_pred = tf.math.add(P_opd_pred, NP_opd_pred)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(train_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - opd_pred.numpy())*np_obscurations

    # Calculate RMSE values
    train_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Pritn RMSE values
    print('Train stars OPD RMSE:\t %.4e'%train_opd_rmse)

    return test_opd_rmse, train_opd_rmse


def compute_opd_metrics_parametric(tf_semiparam_field, GT_tf_semiparam_field, test_pos, train_pos):
    """ Compute the OPD metrics. """

    np_obscurations = np.real(tf_semiparam_field.obscurations.numpy())

    ## For test positions
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(test_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(test_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - P_opd_pred.numpy())*np_obscurations

    # Calculate RMSE values
    test_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Pritn RMSE values
    print('Test stars OPD RMSE:\t %.4e'%test_opd_rmse)


    ## For train part
    # Param part
    zernike_coeffs = tf_semiparam_field.tf_poly_Z_field(train_pos)
    P_opd_pred = tf_semiparam_field.tf_zernike_OPD(zernike_coeffs)

    # GT model
    GT_zernike_coeffs = GT_tf_semiparam_field.tf_poly_Z_field(train_pos)
    GT_opd_maps = GT_tf_semiparam_field.tf_zernike_OPD(GT_zernike_coeffs)

    # Compute residual and obscure the OPD
    res_opd = (GT_opd_maps.numpy() - P_opd_pred.numpy())*np_obscurations

    # Calculate RMSE values
    train_opd_rmse = np.sqrt(np.mean(res_opd**2))

    # Pritn RMSE values
    print('Train stars OPD RMSE:\t %.4e'%train_opd_rmse)

    return test_opd_rmse, train_opd_rmse
