"""Plot Interface.

An interface module with classes to handle different
plot configurations.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
import seaborn as sns
import os
import re
import logging

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def define_plot_style():  # type: ignore
    """Define Plot Style.

    A function to set plot_style
    parameters.

    """
    plot_style = {
        "figure.figsize": (12, 8),
        "figure.dpi": 200,
        "figure.autolayout": True,
        "lines.linewidth": 2,
        "lines.linestyle": "-",
        "lines.marker": "o",
        "lines.markersize": 10,
        "legend.fontsize": 20,
        "legend.loc": "best",
        "axes.titlesize": 24,
        "font.size": 16,
    }
    mpl.rcParams.update(plot_style)
    # Use seaborn style
    sns.set()


def make_plot(
    x,
    y,
    yerr,
    label,
    plot_title,
    x_label,
    y_label_right_axis,
    y_label_left_axis,
    filename,
    plot_show=False,
):
    """Make Plot.

    A function to generate metrics plots.

    Parameters
    ----------
    x: list
        x-axis values
    y: list
        y-axis values
    yerr: list
        Error values for y-axis points
    label: str
        Label for the points
    plot_title: str
        Name of plot
    x_label: str
        Label for x-axis
    y_label_left_axis: str
        Label for left vertical axis of plot
    y_label_right_axis: str
        Label for right vertical axis of plot
    filename: str
        Name of file to save plot
    plot_show: bool
        Boolean flag to set plot display


    """
    define_plot_style()

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)

    ax1.set_title(plot_title)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label_left_axis)
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
    ax2 = ax1.twinx()
    plt.minorticks_on()

    ax2.set_ylabel(y_label_right_axis)
    ax2.grid(False)

    for it in range(len(y)):  # type: ignore
        for k, _ in y[it].items():
            ax1.errorbar(
                x=x[it],
                y=y[it][k],
                yerr=yerr[it][k],
                label=label[it],
                alpha=0.75,
            )
            ax1.legend()
            kwargs = dict(
                linewidth=2, linestyle="dashed", markersize=4, marker="^", alpha=0.5
            )
            ax2.plot(x[it], y[it][k], **kwargs)

    plt.savefig(filename)

    if plot_show is True:
        plt.show()


class MetricsPlotHandler:
    """MetricsPlotHandler class.

    A class to handle plot parameters for various
    metrics results.

    Parameters
    ----------
    id: str
        Class ID name
    plotting_params: Recursive Namespace object
    metrics: list
        Dictionary containing list of metrics
    list_of_stars: list
        List containing the number of training stars per run
    metric_name: str
        Name of metric
    rmse: str
        Root-mean square error label
    std_rmse: str
        Standard error on root-mean square error standard label
    plots_dir: str
        Output directory for metrics plots

    """

    ids = ("poly_metrics", "opd_metrics", "poly_pixel")

    def __init__(
        self,
        plotting_params,
        metrics,
        list_of_stars,
        metric_name,
        rmse,
        std_rmse,
        plot_title,
        plots_dir,
    ):
        self.plotting_params = plotting_params
        self.metrics = metrics
        self.metric_name = metric_name
        self.rmse = rmse
        self.std_rmse = std_rmse
        self.plot_title = plot_title
        self.plots_dir = plots_dir
        self.list_of_stars = list_of_stars

    def get_metrics(self, dataset):
        """Get Metrics.

        A function to get metrics: rmse, rmse_std
        for each run input, e.g. wf-outputs-xxxxxxxxxxxx.

        Parameters
        ----------
        dataset: str
            A str representing dataset type, i.e. test_metrics or train_metrics.

        Returns
        -------
        metrics_id: str
            A str representing the id of a metrics for a given run.
        rmse: list
            A list containing root-mean-square (rms) errors for a metric for each run input.
        std_rmse: list
            A list containing standard errors on the root-mean-square (rms) quantites for a metric for each run input.

        """
        rmse = []
        std_rmse = []
        metrics_id = []
        for k, v in self.metrics.items():
            run_id = list(v.keys())[0]
            metrics_id.append(run_id + "-" + k)

            rmse.append(
                {k: self.metrics[k][run_id][0][dataset][self.metric_name][self.rmse]}
            )
            std_rmse.append(
                {
                    k: self.metrics[k][run_id][0][dataset][self.metric_name][
                        self.std_rmse
                    ]
                }
            )
        return metrics_id, rmse, std_rmse

    def plot(self):
        """Plot.

        A function to generate metric plots as function of number of stars
        for the train and test metrics.

        """
        for plot_dataset in ["test_metrics", "train_metrics"]:
            metrics_id, rmse, std_rmse = self.get_metrics(plot_dataset)
            make_plot(
                x=self.list_of_stars,
                y=rmse,
                yerr=std_rmse,
                label=metrics_id,
                plot_title="Stars " + plot_dataset + self.plot_title,
                x_label="Number of stars",
                y_label_left_axis="Absolute error",
                y_label_right_axis="Relative error [%]",
                filename=os.path.join(
                    self.plots_dir,
                    plot_dataset + "_" + self.metric_name + "_RMSE.png",
                ),
                plot_show=self.plotting_params.plot_show,
            )


class MonochromaticMetricsPlotHandler:
    """MonochromaticMetricsPlotHandler class.

    A class to handle plot parameters for monochromatic
    metrics results.

    Parameters
    ----------
    id: str
        Class ID name
    plotting_params: Recursive Namespace object
        RecursiveNamespace object containing plotting parameters
    metrics_confs: dict
        Dictionary containing the metric configurations as RecursiveNamespace objects for each run
    metrics: list
        Dictionary containing list of metrics
    plots_dir: str
        Output directory for metrics plots

    """

    ids = ("mono_metrics",)

    def __init__(self, plotting_params, metrics_confs, metrics, plots_dir):
        self.plotting_params = plotting_params
        self.metrics_confs = metrics_confs
        self.metrics = metrics
        self.plots_dir = plots_dir

    def plot(self):
        """Plot.

        A function to generate plots for the train and test
        metrics.

        """
        # Define common data
        # Common data
        lambda_list = np.arange(0.55, 0.9, 0.01)
        for plot_dataset in ["test_metrics", "train_metrics"]:
            y = []
            yerr = []
            metrics_id = []

            for k, v in self.metrics.items():
                if self.metrics_confs[k].metrics.eval_mono_metric_rmse:
                    run_id = list(v.keys())[0]
                    metrics_id.append(run_id + "-" + k)
                    y.append(
                        {
                            k: self.metrics[k][run_id][0][plot_dataset]["mono_metric"][
                                "rmse_lda"
                            ]
                        }
                    )
                    yerr.append(
                        {
                            k: self.metrics[k][run_id][0][plot_dataset]["mono_metric"][
                                "std_rmse_lda"
                            ]
                        }
                    )

            make_plot(
                x=[lambda_list for _ in range(len(y))],
                y=y,
                yerr=yerr,
                label=metrics_id,
                plot_title="Stars "
                + plot_dataset  # type: ignore
                + "\nMonochromatic pixel RMSE @ Euclid resolution",
                x_label="Wavelength [um]",
                y_label_left_axis="Absolute error",
                y_label_right_axis="Relative error [%]",
                filename=os.path.join(
                    self.plots_dir,
                    (plot_dataset + "_monochrom_pixel_RMSE.png"),
                ),
                plot_show=self.plotting_params.plot_show,
            )


class ShapeMetricsPlotHandler:
    """ShapeMetricsPlotHandler class.

    A class to handle plot parameters shape
    metrics results.

    Parameters
    ----------
    id: str
        Class ID name
    plotting_params: Recursive Namespace object
        Recursive Namespace Object containing plotting parameters
    metrics: list
        Dictionary containing list of metrics
    plots_dir: str
        Output directory for metrics plots

    """

    id = "shape_metrics"

    def __init__(self, plotting_params, metrics, list_of_stars, plots_dir):
        self.plotting_params = plotting_params
        self.metrics = metrics
        self.list_of_stars = list_of_stars
        self.plots_dir = plots_dir

    def plot(self):
        """Plot.

        A function to generate plots for the train and test
        metrics.

        """
        # Define common data
        # Common data
        e1_req_euclid = 2e-04
        e2_req_euclid = 2e-04
        R2_req_euclid = 1e-03
        for plot_dataset in ["test_metrics", "train_metrics"]:
            e1_rmse = []
            e1_std_rmse = []
            e2_rmse = []
            e2_std_rmse = []
            rmse_R2_meanR2 = []
            std_rmse_R2_meanR2 = []
            metrics_id = []

            for k, v in self.metrics.items():
                run_id = list(v.keys())[0]
                metrics_id.append(run_id + "-" + k)

                e1_rmse.append(
                    {
                        k: self.metrics[k][run_id][0][plot_dataset][
                            "shape_results_dict"
                        ]["rmse_e1"]
                        / e1_req_euclid
                    }
                )
                e1_std_rmse.append(
                    {
                        k: self.metrics[k][run_id][0][plot_dataset][
                            "shape_results_dict"
                        ]["std_rmse_e1"]
                    }
                )

                e2_rmse.append(
                    {
                        k: self.metrics[k][run_id][0][plot_dataset][
                            "shape_results_dict"
                        ]["rmse_e2"]
                        / e2_req_euclid
                    }
                )
                e2_std_rmse.append(
                    {
                        k: self.metrics[k][run_id][0][plot_dataset][
                            "shape_results_dict"
                        ]["std_rmse_e2"]
                    }
                )

                rmse_R2_meanR2.append(
                    {
                        k: self.metrics[k][run_id][0][plot_dataset][
                            "shape_results_dict"
                        ]["rmse_R2_meanR2"]
                        / R2_req_euclid
                    }
                )

                std_rmse_R2_meanR2.append(
                    {
                        k: self.metrics[k][run_id][0][plot_dataset][
                            "shape_results_dict"
                        ]["std_rmse_R2_meanR2"]
                    }
                )

            make_plot(
                x=self.list_of_stars,
                y=e1_rmse,
                yerr=e1_std_rmse,
                label=metrics_id,
                plot_title="Stars " + plot_dataset + ".\nShape RMSE",
                x_label="Number of stars",
                y_label_left_axis="Absolute error",
                y_label_right_axis="Relative error [%]",
                filename=os.path.join(
                    self.plots_dir,
                    plot_dataset + "_Shape_RMSE.png",
                ),
                plot_show=self.plotting_params.plot_show,
            )


def get_number_of_stars(metrics):
    """Get Number of Stars.

    A function to get the number of stars used
    in training the model.

    Returns
    -------
    list_of_stars: list
        A list containing the number of training stars per run.
    """
    list_of_stars = []

    for k, v in metrics.items():
        run_id = list(v.keys())[0]
        list_of_stars.append(int(re.search(r"\d+", run_id).group()))

    return list_of_stars


def plot_metrics(plotting_params, list_of_metrics, metrics_confs, plot_saving_path):
    r"""Plot model results.

    Parameters
    ----------
    plotting_params: RecursiveNamespace Object
        Recursive Namespace object containing plot configuration parameters
    list_of_metrics: list
        List containing all model metrics
    metrics_confs: list
        List containing all metrics configuration parameters
    plot_saving_path: str
        Directory path for saving output plots

    """
    metrics = {
        "poly_metric": {
            "rmse": "rmse",
            "std_rmse": "std_rmse",
            "plot_title": ".\nPolychromatic pixel RMSE @ Euclid resolution",
        },
        "opd_metric": {
            "rmse": "rmse_opd",
            "std_rmse": "rmse_std_opd",
            "plot_title": ".\nOPD RMSE",
        },
        "shape_results_dict": {
            "rmse": "pix_rmse",
            "std_rmse": "pix_rmse_std",
            "plot_title": "\nPixel RMSE @ 3x Euclid resolution",
        },
    }

    list_of_stars = get_number_of_stars(list_of_metrics)

    for k, v in metrics.items():
        metrics_plot = MetricsPlotHandler(
            plotting_params,
            list_of_metrics,
            list_of_stars,
            k,
            v["rmse"],
            v["std_rmse"],
            v["plot_title"],
            plot_saving_path,
        )
        metrics_plot.plot()

    monochrom_metrics_plot = MonochromaticMetricsPlotHandler(
        plotting_params, metrics_confs, list_of_metrics, plot_saving_path
    )
    monochrom_metrics_plot.plot()

    shape_metrics_plot = ShapeMetricsPlotHandler(
        plotting_params, list_of_metrics, list_of_stars, plot_saving_path
    )
    shape_metrics_plot.plot()
