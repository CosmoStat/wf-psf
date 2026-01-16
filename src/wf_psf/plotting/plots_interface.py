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
    x_axis,
    y_axis,
    y_axis_err,
    y2_axis,
    label,
    plot_title,
    x_axis_label,
    y_right_axis_label,
    y_left_axis_label,
    filename,
    plot_show=False,
):
    """Make Plot.

    A function to generate metrics plots.

    Parameters
    ----------
    x_axis: list
        x-axis values
    y_axis: list
        y-axis values
    y_axis_err: list
        Error values for y-axis points
    y2_axis: list
        y2-axis values for right axis
    label: str
        Label for the points
    plot_title: str
        Name of plot
    x_axis_label: str
        Label for x-axis
    y_left_axis_label: str
        Label for left vertical axis of plot
    y_right_axis_label: str
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
    ax1.set_xlabel(x_axis_label)
    ax1.set_ylabel(y_left_axis_label)
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1e"))
    ax2 = ax1.twinx()
    plt.minorticks_on()

    ax2.set_ylabel(y_right_axis_label)
    ax2.grid(False)

    for it in range(len(y_axis)):  # type: ignore
        for k, _ in y_axis[it].items():
            ax1.errorbar(
                x=x_axis[it],
                y=y_axis[it][k],
                yerr=y_axis_err[it][k],
                label=label[it],
                alpha=0.75,
            )
            ax1.legend()
            kwargs = dict(
                linewidth=2, linestyle="dashed", markersize=4, marker="^", alpha=0.5
            )
            ax2.plot(x_axis[it], y2_axis[it][k], **kwargs)

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
    metrics: dict
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
        rel_rmse,
        plot_title,
        plots_dir,
    ):
        self.plotting_params = plotting_params
        self.metrics = metrics
        self.metric_name = metric_name
        self.rmse = rmse
        self.std_rmse = std_rmse
        self.rel_rmse = rel_rmse
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
        tuple:
            A tuple consisting of the id, root-mean-square (rms) and standard rms errors for a metric for each input run.

        """
        rmse = []
        std_rmse = []
        rel_rmse = []
        metrics_id = []
        for k, v in self.metrics.items():
            for metrics_data in v:
                run_id = list(metrics_data.keys())[0]
                metrics_id.append(run_id + "-" + k)

                rmse.append(
                    {
                        (k + "-" + run_id): metrics_data[run_id][0][dataset][
                            self.metric_name
                        ][self.rmse]
                    }
                )
                std_rmse.append(
                    {
                        (k + "-" + run_id): metrics_data[run_id][0][dataset][
                            self.metric_name
                        ][self.std_rmse]
                    }
                )

                rel_rmse.append(
                    {
                        (k + "-" + run_id): metrics_data[run_id][0][dataset][
                            self.metric_name
                        ][self.rel_rmse]
                    }
                )

        return metrics_id, rmse, std_rmse, rel_rmse

    def plot(self):
        """Plot.

        A function to generate metric plots as function of number of stars
        for the train and test metrics.

        """
        for plot_dataset in ["test_metrics", "train_metrics"]:
            metrics_id, rmse, std_rmse, rel_rmse = self.get_metrics(plot_dataset)
            make_plot(
                x_axis=self.list_of_stars,
                y_axis=rmse,
                y_axis_err=std_rmse,
                y2_axis=rel_rmse,
                label=metrics_id,
                plot_title="Stars " + plot_dataset + self.plot_title,
                x_axis_label="Number of stars",
                y_left_axis_label="Absolute error",
                y_right_axis_label="Relative error [%]",
                filename=os.path.join(
                    self.plots_dir,
                    plot_dataset
                    + "_"
                    + self.metric_name
                    + "_nstars_"
                    + "_".join(str(nstar) for nstar in self.list_of_stars)
                    + "_RMSE.png",
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
    list_of_stars: list
        List containing the number of stars used for each training data set
    plots_dir: str
        Output directory for metrics plots

    """

    ids = ("mono_metrics",)

    def __init__(
        self,
        plotting_params,
        metrics_confs,
        metrics,
        list_of_stars,
        plots_dir,
    ):
        self.plotting_params = plotting_params
        self.metrics_confs = metrics_confs
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
        lambda_list = np.arange(0.55, 0.9, 0.01)
        for plot_dataset in ["test_metrics", "train_metrics"]:
            y_axis = []
            y_axis_err = []
            y2_axis = []
            metrics_id = []

            for k, v in self.metrics.items():
                if self.metrics_confs[k].metrics.eval_mono_metric:
                    for metrics_data in v:
                        run_id = list(metrics_data.keys())[0]
                        metrics_id.append(run_id + "-" + k)
                        y_axis.append(
                            {
                                (k + "-" + run_id): metrics_data[run_id][0][
                                    plot_dataset
                                ]["mono_metric"]["rmse_lda"]
                            }
                        )
                        y_axis_err.append(
                            {
                                (k + "-" + run_id): metrics_data[run_id][0][
                                    plot_dataset
                                ]["mono_metric"]["std_rmse_lda"]
                            }
                        )
                        y2_axis.append(
                            {
                                (k + "-" + run_id): metrics_data[run_id][0][
                                    plot_dataset
                                ]["mono_metric"]["rel_rmse_lda"]
                            }
                        )

            make_plot(
                x_axis=[lambda_list for _ in range(len(y_axis))],
                y_axis=y_axis,
                y_axis_err=y_axis_err,
                y2_axis=y2_axis,
                label=metrics_id,
                plot_title="Stars "
                + plot_dataset  # type: ignore
                + "\nMonochromatic pixel RMSE @ Euclid resolution",
                x_axis_label="Wavelength [um]",
                y_left_axis_label="Absolute error",
                y_right_axis_label="Relative error [%]",
                filename=os.path.join(
                    self.plots_dir,
                    (
                        plot_dataset
                        + "_nstars_"
                        + "_".join(str(nstar) for nstar in self.list_of_stars)
                        + "_monochrom_pixel_RMSE.png"
                    ),
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
    list_of_stars: list
        List containing the number of stars used for each training data set
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

        A generic function to generate plots for the train and test
        metrics.

        """
        e1_req_euclid = 2e-04
        e2_req_euclid = 2e-04
        R2_req_euclid = 1e-03

        for plot_dataset in ["test_metrics", "train_metrics"]:
            metrics_data = self.prepare_metrics_data(
                plot_dataset, e1_req_euclid, e2_req_euclid, R2_req_euclid
            )

            # Plot for e1
            for k, v in metrics_data.items():
                self.make_shape_metrics_plot(
                    metrics_data[k]["rmse"],
                    metrics_data[k]["std_rmse"],
                    metrics_data[k]["rel_rmse"],
                    plot_dataset,
                    k,
                )

    def prepare_metrics_data(
        self, plot_dataset, e1_req_euclid, e2_req_euclid, R2_req_euclid
    ):
        """Prepare Metrics Data.

        A function to prepare the metrics data for plotting.

        Parameters
        ----------
        plot_dataset: str
            A string representing the dataset, i.e. training or test metrics.
        e1_req_euclid: float
            A float denoting the Euclid requirement for the `e1` shape metric.
        e2_req_euclid: float
            A float denoting the Euclid requirement for the `e2` shape metric.
        R2_req_euclid: float
            A float denoting the Euclid requirement for the `R2` shape metric.

        Returns
        -------
        shape_metrics_data: dict
            A dictionary containing the shape metrics data from a set of runs.

        """
        shape_metrics_data = {
            "e1": {"rmse": [], "std_rmse": [], "rel_rmse": []},
            "e2": {"rmse": [], "std_rmse": [], "rel_rmse": []},
            "R2_meanR2": {"rmse": [], "std_rmse": [], "rel_rmse": []},
        }

        for k, v in self.metrics.items():
            for metrics_data in v:
                run_id = list(metrics_data.keys())[0]

                for metric in ["e1", "e2", "R2_meanR2"]:
                    metric_rmse = metrics_data[run_id][0][plot_dataset][
                        "shape_results_dict"
                    ][f"rmse_{metric}"]
                    metric_std_rmse = metrics_data[run_id][0][plot_dataset][
                        "shape_results_dict"
                    ][f"std_rmse_{metric}"]

                    relative_metric_rmse = metric_rmse / (
                        e1_req_euclid
                        if metric == "e1"
                        else (e2_req_euclid if metric == "e2" else R2_req_euclid)
                    )

                    shape_metrics_data[metric]["rmse"].append(
                        {f"{k}-{run_id}": metric_rmse}
                    )
                    shape_metrics_data[metric]["std_rmse"].append(
                        {f"{k}-{run_id}": metric_std_rmse}
                    )
                    shape_metrics_data[metric]["rel_rmse"].append(
                        {f"{k}-{run_id}": relative_metric_rmse}
                    )

        return shape_metrics_data

    def make_shape_metrics_plot(
        self, rmse_data, std_rmse_data, rel_rmse_data, plot_dataset, metric
    ):
        """Make Shape Metrics Plot.

        A function to produce plots for the shape metrics.

        Parameters
        ----------
        rmse_data: list
            A list of dictionaries where each dictionary stores run as the key and the Root Mean Square Error (rmse).
        std_rmse_data: list
            A list of dictionaries where each dictionary stores run as the key and the Standard Deviation of the Root Mean Square Error (rmse) as the value.
        rel_rmse_data: list
            A list of dictionaries where each dictionary stores run as the key and the Root Mean Square Error (rmse) relative to the Euclid requirements as the value.
        plot_dataset: str
            A string denoting whether metrics are for the train or test datasets.
        metric: str
            A string representing the type of shape metric, i.e., e1, e2, or R2.

        """
        make_plot(
            x_axis=self.list_of_stars,
            y_axis=rmse_data,
            y_axis_err=std_rmse_data,
            y2_axis=rel_rmse_data,
            label=[key for item in rmse_data for key in item],
            plot_title=f"Stars {plot_dataset}. Shape {metric.upper()} RMSE",
            x_axis_label="Number of stars",
            y_left_axis_label="Absolute error",
            y_right_axis_label="Relative error [%]",
            filename=os.path.join(
                self.plots_dir,
                f"{plot_dataset}_nstars_{'_'.join(str(nstar) for nstar in self.list_of_stars)}_Shape_{metric.upper()}_RMSE.png",
            ),
            plot_show=self.plotting_params.plot_show,
        )


def get_number_of_stars(metrics):
    """Get Number of Stars.

    A function to get the number of stars used
    in training the model.

    Parameters
    ----------
    metrics: dict
        A dictionary containig the metrics results per run

    Returns
    -------
    list_of_stars: list
        A list containing the number of training stars per run.
    """
    list_of_stars = []

    try:
        for k, v in metrics.items():
            for run in v:
                run_id = list(run.keys())[0]
                list_of_stars.append(int(re.search(r"\d+", run_id).group()))
    except AttributeError:
        list_of_stars = np.arange(len(metrics.items())) + 1

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
            "rel_rmse": "rel_rmse",
            "plot_title": ".\nPolychromatic pixel RMSE @ Euclid resolution",
        },
        "opd_metric": {
            "rmse": "rmse_opd",
            "std_rmse": "rmse_std_opd",
            "rel_rmse": "rel_rmse_opd",
            "plot_title": ".\nOPD RMSE",
        },
        "shape_results_dict": {
            "rmse": "pix_rmse",
            "std_rmse": "pix_rmse_std",
            "rel_rmse": "rel_pix_rmse",
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
            v["rel_rmse"],
            v["plot_title"],
            plot_saving_path,
        )
        metrics_plot.plot()

    monochrom_metrics_plot = MonochromaticMetricsPlotHandler(
        plotting_params, metrics_confs, list_of_metrics, list_of_stars, plot_saving_path
    )
    monochrom_metrics_plot.plot()

    shape_metrics_plot = ShapeMetricsPlotHandler(
        plotting_params, list_of_metrics, list_of_stars, plot_saving_path
    )
    shape_metrics_plot.plot()
