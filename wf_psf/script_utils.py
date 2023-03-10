# Import packages
import sys
import numpy as np
import time
import tensorflow as tf
import tensorflow_addons as tfa

import wf_psf.SimPSFToolkit as SimPSFToolkit
import wf_psf.utils.utils as wf_utils
import wf_psf.psf_models.tf_mccd_psf_field as tf_mccd_psf_field
import wf_psf.psf_models.tf_psf_field as tf_psf_field
import wf_psf.metrics as wf_metrics
import training.train_utils as wf_train_utils

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.ticker as mtick
    import seaborn as sns
except:
    print("\nProblem importing plotting packages: matplotlib and seaborn.\n")


def train_model(**args):
    r"""Train the PSF model.

    For parameters check the training script click help.

    """
    # Start measuring elapsed time
    starting_time = time.time()

    # Define model run id
    run_id_name = args["model"] + args["id_name"]

    # Define paths -> move to io
    log_save_file = args["base_path"] + args["log_folder"]
    model_save_file = args["base_path"] + args["model_folder"]
    optim_hist_file = args["base_path"] + args["optim_hist_folder"]
    saving_optim_hist = dict()

    # Save output prints to logfile
    old_stdout = sys.stdout
    log_file = open(log_save_file + run_id_name + "_output.log", "w")
    sys.stdout = log_file
    print("Starting the log file.")

    # Print GPU and tensorflow info
    device_name = tf.test.gpu_device_name()
    print("Found GPU at: {}".format(device_name))
    print("tf_version: " + str(tf.__version__))

    # Prepare the inputs
    # Generate Zernike maps
    zernikes = wf_utils.zernike_generator(
        n_zernikes=args["n_zernikes"], wfe_dim=args["pupil_diameter"]
    )
    # Now as cubes --- DONE
    np_zernike_cube = np.zeros(
        (len(zernikes), zernikes[0].shape[0], zernikes[0].shape[1])
    )
    for it in range(len(zernikes)):
        np_zernike_cube[it, :, :] = zernikes[it]
    np_zernike_cube[np.isnan(np_zernike_cube)] = 0
    tf_zernike_cube = tf.convert_to_tensor(np_zernike_cube, dtype=tf.float32)
    print("Zernike cube:")
    print(tf_zernike_cube.shape)

    # Load the dictionaries
    train_dataset = np.load(
        args["dataset_folder"] + args["train_dataset_file"], allow_pickle=True
    )[()]

    train_SEDs = train_dataset["SEDs"]
    train_parameters = train_dataset["parameters"]
    test_dataset = np.load(
        args["dataset_folder"] + args["test_dataset_file"], allow_pickle=True
    )[()]

    test_SEDs = test_dataset["SEDs"]

    # Convert to tensor
    tf_noisy_train_stars = tf.convert_to_tensor(
        train_dataset["noisy_stars"], dtype=tf.float32
    )
    tf_train_pos = tf.convert_to_tensor(train_dataset["positions"], dtype=tf.float32)
    tf_test_stars = tf.convert_to_tensor(test_dataset["stars"], dtype=tf.float32)
    tf_test_pos = tf.convert_to_tensor(test_dataset["positions"], dtype=tf.float32)

    print("Dataset parameters:")
    print(train_parameters)

    # New interp features backwards compatibility
    if "interp_pts_per_bin" not in args:
        args["interp_pts_per_bin"] = 0
        args["extrapolate"] = True
        args["sed_interp_kind"] = "linear"

    ## Generate initializations -- This looks like it could be moved to PSF model package
    # Prepare np input
    simPSF_np = SimPSFToolkit(
        zernikes,
        max_order=args["n_zernikes"],
        pupil_diameter=args["pupil_diameter"],
        output_dim=args["output_dim"],
        oversampling_rate=args["oversampling_rate"],
        output_Q=args["output_Q"],
        interp_pts_per_bin=args["interp_pts_per_bin"],
        extrapolate=args["extrapolate"],
        SED_interp_kind=args["sed_interp_kind"],
        SED_sigma=args["sed_sigma"],
    )

    # this method updates simPSF_np.zernikes
    simPSF_np.gen_random_Z_coeffs(max_order=args["n_zernikes"])

    z_coeffs = simPSF_np.normalize_zernikes(
        simPSF_np.get_z_coeffs(), simPSF_np.max_wfe_rms
    )
    simPSF_np.set_z_coeffs(z_coeffs)
    simPSF_np.generate_mono_PSF(lambda_obs=0.7, regen_sample=False)

    # Obscurations --> Moved to psf_models/psf_models.py and called static method
    obscurations = simPSF_np.generate_pupil_obscurations(
        N_pix=args["pupil_diameter"], N_filter=2
    )
    tf_obscurations = tf.convert_to_tensor(obscurations, dtype=tf.complex64)

    # Initialize the SED data list
    packed_SED_data = [
        wf_utils.generate_packed_elems(_sed, simPSF_np, n_bins=args["n_bins_lda"])
        for _sed in train_SEDs
    ]

    # Prepare the inputs for the training
    tf_packed_SED_data = tf.convert_to_tensor(packed_SED_data, dtype=tf.float32)
    tf_packed_SED_data = tf.transpose(tf_packed_SED_data, perm=[0, 2, 1])

    #JP : inputs = [tf_train_pos, tf_packed_SED_data]

    # Select the observed stars (noisy or noiseless)
    # JP: outputs = tf_noisy_train_stars
    # outputs = tf_train_stars

    # Prepare validation data inputs
    # JP: not necessary -- just making a duplicate copy in memory
   # validation_SEDs = test_SEDs
   # tf_validation_pos = tf_test_pos
   # tf_validation_stars = tf_test_stars

    # Initialize the SED data list
    validation_packed_SED_data = [
        wf_utils.generate_packed_elems(_sed, simPSF_np, n_bins=args["n_bins_lda"])
        for _sed in test_SEDs
    ]

    # Prepare the inputs for the validation
    tf_validation_packed_SED_data = tf.convert_to_tensor(validation_packed_SED_data, dtype=tf.float32)
    tf_validation_packed_SED_data = tf.transpose(tf_validation_packed_SED_data, perm=[0, 2, 1])

    # Prepare input validation tuple
  #  validation_x_inputs = [tf_test_pos, tf_validation_packed_SED_data]
  #  validation_y_inputs = tf_validation_stars
   # validation_data = ([tf_test_pos, tf_validation_packed_SED_data], tf_test_stars)

    # Select the model-- DONE in train.py
    if args["model"] == "poly":
        # Initialize the WaveDiff-original model
        tf_semiparam_field = tf_psf_field.TF_SemiParam_field(
            zernike_maps=tf_zernike_cube,
            obscurations=tf_obscurations,
            batch_size=args["batch_size"],
            output_Q=args["output_Q"],
            d_max_nonparam=args["d_max_nonparam"],
            l2_param=args["l2_param"],
            output_dim=args["output_dim"],
            n_zernikes=args["n_zernikes"],
            d_max=args["d_max"],
            x_lims=args["x_lims"],
            y_lims=args["y_lims"],
        )

    # Load pretrained model
    if args["model"] == "poly" and args["pretrained_model"] is not None:
        tf_semiparam_field.load_weights(args["pretrained_model"])
        print("Model loaded.")
        tf_semiparam_field.project_DD_features(tf_zernike_cube)
        print("DD features projected over parametric model")

    # If reset_dd_features is true we project the DD features onto the param model and reset them.
    if (
        args["model"] == "poly"
        and args["reset_dd_features"]
        and args["cycle_def"] != "only-parametric"
    ):
        tf_semiparam_field.tf_np_poly_opd.init_vars()
        print("DD features reseted to random initialisation.")

    # # Model Training
    # Prepare the saving callback
    # Prepare to save the model as a callback
    filepath_chkp_callback = (
        args["chkp_save_path"] + "chkp_callback_" + run_id_name + "_cycle1"
    )
    model_chkp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath_chkp_callback,
        monitor="mean_squared_error",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
        options=None,
    )

    # Prepare the optimisers
    param_optim = tfa.optimizers.RectifiedAdam(learning_rate=args["learning_rate_param"][0])
    non_param_optim = tfa.optimizers.RectifiedAdam(
        learning_rate=args["learning_rate_non_param"][0]
    )

    print("Starting cycle 1..")
    start_cycle1 = time.time()

    tf_semiparam_field, hist_param, hist_non_param = wf_train_utils.general_train_cycle(
        # poly model
        tf_semiparam_field,
        # training data
        inputs=[tf_train_pos, tf_packed_SED_data],
        #
        outputs=tf_noisy_train_stars,
        validation_data= ([tf_test_pos, tf_validation_packed_SED_data], tf_test_stars),
        batch_size=args["batch_size"],
        learning_rate_param=args["learning_rate_param"][0],
        learning_rate_non_param=args["learning_rate_non_param"][0],
        n_epochs_param=args["n_epochs_param"][0],
        n_epochs_non_param=args["n_epochs_non_param"][0],
        param_optim=param_optim,
        non_param_optim=non_param_optim,
        param_loss=None,
        non_param_loss=None,
        param_metrics=None,
        non_param_metrics=None,
        param_callback=None,
        non_param_callback=None,
        general_callback=[model_chkp_callback],
        first_run=True,
        cycle_def=args["cycle_def"],
        use_sample_weights=args["use_sample_weights"],
        verbose=2,
    )

    # Backwards compatibility with click scripts older than the projected learning feature
    if "save_all_cycles" not in args:
        args["save_all_cycles"] = False

    # Save weights
    if args["save_all_cycles"]:
        tf_semiparam_field.save_weights(
            model_save_file + "chkp_" + run_id_name + "_cycle1"
        )

    end_cycle1 = time.time()
    print("Cycle1 elapsed time: %f" % (end_cycle1 - start_cycle1))

    # Save optimisation history in the saving dict
    if hist_param is not None:
        saving_optim_hist["param_cycle1"] = hist_param.history
    if args["model"] != "param" and hist_non_param is not None:
        saving_optim_hist["nonparam_cycle1"] = hist_non_param.history

    # Perform all the necessary cycles
    current_cycle = 1

    while args["total_cycles"] > current_cycle:
        current_cycle += 1

        # If projected learning is enabled project DD_features.
        if args["project_dd_features"] and args["model"] == "poly":
            tf_semiparam_field.project_DD_features(tf_zernike_cube)
            print("Project non-param DD features onto param model: done!")
            if args["reset_dd_features"]:
                tf_semiparam_field.tf_np_poly_opd.init_vars()
                print("DD features reseted to random initialisation.")

        # Prepare to save the model as a callback
        filepath_chkp_callback = (
            args["chkp_save_path"]
            + "chkp_callback_"
            + run_id_name
            + "_cycle"
            + str(current_cycle)
        )
        model_chkp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath_chkp_callback,
            monitor="mean_squared_error",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            save_freq="epoch",
            options=None,
        )

        # Prepare the optimisers
        param_optim = tfa.optimizers.RectifiedAdam(
            learning_rate=args["learning_rate_param"][current_cycle - 1]
        )
        non_param_optim = tfa.optimizers.RectifiedAdam(
            learning_rate=args["learning_rate_non_param"][current_cycle - 1]
        )

        print("Starting cycle {}..".format(current_cycle))
        start_cycle = time.time()

        # Compute the next cycle
        (
            tf_semiparam_field,
            hist_param_2,
            hist_non_param_2,
        ) = wf_train_utils.general_train_cycle(
            tf_semiparam_field,
            inputs=inputs,
            outputs=outputs,
            validation_data=validation_data,
            batch_size=args["batch_size"],
            learning_rate_param=args["learning_rate_param"][current_cycle - 1],
            learning_rate_non_param=args["learning_rate_non_param"][current_cycle - 1],
            n_epochs_param=args["n_epochs_param"][current_cycle - 1],
            n_epochs_non_param=args["n_epochs_non_param"][current_cycle - 1],
            param_optim=param_optim,
            non_param_optim=non_param_optim,
            param_loss=None,
            non_param_loss=None,
            param_metrics=None,
            non_param_metrics=None,
            param_callback=None,
            non_param_callback=None,
            general_callback=[model_chkp_callback],
            first_run=False,
            cycle_def=args["cycle_def"],
            use_sample_weights=args["use_sample_weights"],
            verbose=2,
        )

        # Save the weights at the end of the second cycle
        if args["save_all_cycles"]:
            tf_semiparam_field.save_weights(
                model_save_file + "chkp_" + run_id_name + "_cycle" + str(current_cycle)
            )

        end_cycle = time.time()
        print("Cycle{} elapsed time: {}".format(current_cycle, end_cycle - start_cycle))

        # Save optimisation history in the saving dict
        if hist_param_2 is not None:
            saving_optim_hist[
                "param_cycle{}".format(current_cycle)
            ] = hist_param_2.history
        if args["model"] != "param" and hist_non_param_2 is not None:
            saving_optim_hist[
                "nonparam_cycle{}".format(current_cycle)
            ] = hist_non_param_2.history

    # Save last cycle if no cycles were saved
    if not args["save_all_cycles"]:
        tf_semiparam_field.save_weights(
            model_save_file + "chkp_" + run_id_name + "_cycle" + str(current_cycle)
        )

    # Save optimisation history dictionary
    np.save(optim_hist_file + "optim_hist_" + run_id_name + ".npy", saving_optim_hist)

    # Print final time
    final_time = time.time()
    print("\nTotal elapsed time: %f" % (final_time - starting_time))

    # Close log file
    print("\n Good bye..")
    sys.stdout = old_stdout
    log_file.close()
