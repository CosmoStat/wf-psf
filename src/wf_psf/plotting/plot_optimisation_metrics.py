"""Plot Optimisation Metrics.

A module to produce plots of the various
PSF optimation metrics.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""
from wf_psf.plotting.plots_interface import define_plot_style


def plot_optimisation_metrics(**args):
    r"""Plot optimisation results."""
    define_plot_style()

    # Define saving path
    plot_saving_path = args["base_path"] + args["plots_folder"]

    # Define the number of datasets to test
    if not isinstance(args["suffix_id_name"], str):
        n_datasets = len(args["suffix_id_name"])
    else:
        n_datasets = 1

    optim_hist_file = args["base_path"] + args["optim_hist_folder"]
    run_id_no_suff = args["model"] + args["base_id_name"]

    # Define the metric data paths
    if not isinstance(key[0], str):
        model_paths = [optim_hist_file + "optim_hist_" + key + ".npy" for _suff in key]

    else:
        model_paths = [
            optim_hist_file
            + "optim_hist_"
            + "run_id_no_suff"
            + str(args["suffix_id_name"])
            + ".npy"
        ]

    print("Model paths for optimisation plots: ", model_paths)

    try:
        # Load metrics
        metrics = [np.load(_path, allow_pickle=True)[()] for _path in model_paths]
    except FileNotFoundError:
        print("The required file for the plots was not found.")
        print("Probably I am not the last job for plotting the optimisation metrics.")
        raise 0

    ## Plot the first parametric cycle
    cycle_str = "param_cycle1"
    metric_str = "mean_squared_error"
    val_mertric_str = "val_mean_squared_error"

    try:
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(111)
        for it in range(n_datasets):
            try:
                ax1.plot(
                    metrics[it][cycle_str][metric_str],
                    label=args["model"] + args["suffix_id_name"][it],
                    alpha=0.75,
                )
            except KeyError as KE:
                print("Error with Key: ", KE)
        plt.yscale("log")
        plt.minorticks_on()
        ax1.legend()
        ax1.set_title("Parametric cycle 1.\n" + "run_id_no_suff" + "_" + cycle_str)
        ax1.set_xlabel("Number of epoch")
        ax1.set_ylabel("Training MSE")

        ax2 = ax1.twinx()
        kwargs = dict(
            linewidth=2, linestyle="dashed", markersize=2, marker="^", alpha=0.5
        )
        for it in range(n_datasets):
            try:
                ax2.plot(metrics[it][cycle_str][val_mertric_str], **kwargs)
            except KeyError as KE:
                print("Error with Key: ", KE)
        ax2.set_ylabel("Validation MSE")
        ax2.grid(False)
        plt.savefig(
            plot_saving_path + "optim_" + "run_id_no_suff" + "_" + cycle_str + ".png"
        )
        plt.show()
    except Exception:
        print(
            "Problem with the plot of the optimisation metrics of the first parametric cycle."
        )

    # Plot the first non-parametric cycle
    if args["model"] != "param":
        try:
            cycle_str = "nonparam_cycle1"
            metric_str = "mean_squared_error"
            val_mertric_str = "val_mean_squared_error"

            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_subplot(111)
            for it in range(n_datasets):
                try:
                    ax1.plot(
                        metrics[it][cycle_str][metric_str],
                        label=args["model"] + args["suffix_id_name"][it],
                        alpha=0.75,
                    )
                except KeyError as KE:
                    print("Error with Key: ", KE)
            plt.yscale("log")
            plt.minorticks_on()
            ax1.legend()
            ax1.set_title(
                "Non-parametric cycle 1.\n" + "run_id_no_suff" + "_" + cycle_str
            )
            ax1.set_xlabel("Number of epoch")
            ax1.set_ylabel("Training MSE")

            ax2 = ax1.twinx()
            kwargs = dict(
                linewidth=2, linestyle="dashed", markersize=2, marker="^", alpha=0.5
            )
            for it in range(n_datasets):
                try:
                    ax2.plot(metrics[it][cycle_str][val_mertric_str], **kwargs)
                except KeyError as KE:
                    print("Error with Key: ", KE)
            ax2.set_ylabel("Validation MSE")
            ax2.grid(False)
            plt.savefig(
                plot_saving_path
                + "optim_"
                + "run_id_no_suff"
                + "_"
                + cycle_str
                + ".png"
            )
            plt.show()
        except Exception:
            print(
                "Problem with the plot of the optimisation metrics of the first non-parametric cycle."
            )

    ## Plot the second parametric cycle
    if cycle_str in metrics[0]:
        cycle_str = "param_cycle2"
        metric_str = "mean_squared_error"
        val_mertric_str = "val_mean_squared_error"

        try:
            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_subplot(111)
            for it in range(n_datasets):
                try:
                    ax1.plot(
                        metrics[it][cycle_str][metric_str],
                        label=args["model"] + args["suffix_id_name"][it],
                        alpha=0.75,
                    )
                except KeyError as KE:
                    print("Error with Key: ", KE)
            plt.yscale("log")
            plt.minorticks_on()
            ax1.legend()
            ax1.set_title("Parametric cycle 2.\n" + "run_id_no_suff" + "_" + cycle_str)
            ax1.set_xlabel("Number of epoch")
            ax1.set_ylabel("Training MSE")

            ax2 = ax1.twinx()
            kwargs = dict(
                linewidth=2, linestyle="dashed", markersize=2, marker="^", alpha=0.5
            )
            for it in range(n_datasets):
                try:
                    ax2.plot(metrics[it][cycle_str][val_mertric_str], **kwargs)
                except KeyError as KE:
                    print("Error with Key: ", KE)
            ax2.set_ylabel("Validation MSE")
            ax2.grid(False)
            plt.savefig(
                plot_saving_path
                + "optim_"
                + "run_id_no_suff"
                + "_"
                + cycle_str
                + ".png"
            )
            plt.show()
        except Exception:
            print(
                "Problem with the plot of the optimisation metrics of the second parametric cycle."
            )

    ## Plot the second non-parametric cycle
    if cycle_str in metrics[0]:
        cycle_str = "nonparam_cycle2"
        metric_str = "mean_squared_error"
        val_mertric_str = "val_mean_squared_error"

        try:
            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_subplot(111)
            for it in range(n_datasets):
                try:
                    ax1.plot(
                        metrics[it][cycle_str][metric_str],
                        label=args["model"] + args["suffix_id_name"][it],
                        alpha=0.75,
                    )
                except KeyError as KE:
                    print("Error with Key: ", KE)
            plt.yscale("log")
            plt.minorticks_on()
            ax1.legend()
            ax1.set_title(
                "Non-parametric cycle 2.\n" + "run_id_no_suff" + "_" + cycle_str
            )
            ax1.set_xlabel("Number of epoch")
            ax1.set_ylabel("Training MSE")

            ax2 = ax1.twinx()
            kwargs = dict(
                linewidth=2, linestyle="dashed", markersize=2, marker="^", alpha=0.5
            )
            for it in range(n_datasets):
                try:
                    ax2.plot(metrics[it][cycle_str][val_mertric_str], **kwargs)
                except KeyError as KE:
                    print("Error with Key: ", KE)
            ax2.set_ylabel("Validation MSE")
            ax2.grid(False)
            plt.savefig(
                plot_saving_path
                + "optim_"
                + "run_id_no_suff"
                + "_"
                + cycle_str
                + ".png"
            )
            plt.show()
        except Exception:
            print(
                "Problem with the plot of the optimisation metrics of the second non-parametric cycle."
            )
