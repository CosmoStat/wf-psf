"""Plot Metrics.

A module to produce plots of the various
PSF metrics.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>

"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
import seaborn as sns

def plot_metrics(plotting_params):
    r""" Plot model results.
    """
    define_plot_style()

    plot_saving_path = plotting_params.output_file

    # Define common data
    # Common data
    lambda_list = np.arange(0.55, 0.9, 0.01)
    star_list = np.array(plotting_params.star_numbers)
    e1_req_euclid = 2e-04
    e2_req_euclid = 2e-04
    R2_req_euclid = 1e-03

    # Define the number of datasets to test
    if not isinstance(args['suffix_id_name'], str):
        n_datasets = len(args['suffix_id_name'])
    else:
        n_datasets = 1

    # Run id without suffix
    run_id_no_suff = args['model'] + args['base_id_name']

    # Define the metric data paths
    if not isinstance(args['suffix_id_name'], str):
        model_paths = [
            args['metric_base_path'] + 'metrics-' + run_id_no_suff + _suff + '.npy'
            for _suff in args['suffix_id_name']
        ]
    else:
        model_paths = [
            args['metric_base_path'] + 'metrics-' + run_id_no_suff + args['suffix_id_name'] + '.npy'
        ]

    print('Model paths for performance plots: ', model_paths)

    # Load metrics
    try:
        metrics = [np.load(_path, allow_pickle=True)[()] for _path in model_paths]
    except FileNotFoundError:
        print('The required file for the plots was not found.')
        print('Probably I am not the last job for plotting the performance metrics.')
        return 0

    for plot_dataset in ['test', 'train']:

        try:
            ## Polychromatic results
            res = extract_poly_results(metrics, test_train=plot_dataset)
            model_polyc_rmse = res[0]
            model_polyc_std_rmse = res[1]
            model_polyc_rel_rmse = res[2]
            model_polyc_std_rel_rmse = res[3]

            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_subplot(111)
            ax1.errorbar(
                x=star_list,
                y=model_polyc_rmse,
                yerr=model_polyc_std_rmse,
                label=run_id_no_suff,
                alpha=0.75
            )
            plt.minorticks_on()
            ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            ax1.legend()
            ax1.set_title(
                'Stars ' + plot_dataset + '\n' + run_id_no_suff +
                '.\nPolychromatic pixel RMSE @ Euclid resolution'
            )
            ax1.set_xlabel('Number of stars')
            ax1.set_ylabel('Absolute error')
            ax2 = ax1.twinx()
            kwargs = dict(linewidth=2, linestyle='dashed', markersize=4, marker='^', alpha=0.5)
            ax2.plot(star_list, model_polyc_rel_rmse, **kwargs)
            ax2.set_ylabel('Relative error [%]')
            ax2.grid(False)
            plt.savefig(
                plot_saving_path + plot_dataset + '-metrics-' + run_id_no_suff +
                '_polyc_pixel_RMSE.png'
            )
            plt.show()
        except Exception:
            print('Problem with the performance metrics plot of pixel polychromatic errors.')

        ## Monochromatic
        if args['eval_mono_metric_rmse'] is True or 'eval_mono_metric_rmse' not in args:
            try:
                fig = plt.figure(figsize=(12, 8))
                ax1 = fig.add_subplot(111)
                for it in range(n_datasets):
                    ax1.errorbar(
                        x=lambda_list,
                        y=metrics[it]['test_metrics']['mono_metric']['rmse_lda'],
                        yerr=metrics[it]['test_metrics']['mono_metric']['std_rmse_lda'],
                        label=args['model'] + args['suffix_id_name'][it],
                        alpha=0.75
                    )
                plt.minorticks_on()
                ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                ax1.legend()
                ax1.set_title(
                    'Stars ' + plot_dataset + '\n' + run_id_no_suff +
                    '.\nMonochromatic pixel RMSE @ Euclid resolution'
                )
                ax1.set_xlabel('Wavelength [um]')
                ax1.set_ylabel('Absolute error')

                ax2 = ax1.twinx()
                kwargs = dict(linewidth=2, linestyle='dashed', markersize=8, marker='^', alpha=0.5)
                for it in range(n_datasets):
                    ax2.plot(
                        lambda_list, metrics[it]['test_metrics']['mono_metric']['rel_rmse_lda'],
                        **kwargs
                    )
                ax2.set_ylabel('Relative error [%]')
                ax2.grid(False)
                plt.savefig(
                    plot_saving_path + plot_dataset + '-metrics-' + run_id_no_suff +
                    '_monochrom_pixel_RMSE.png'
                )
                plt.show()
            except Exception:
                print('Problem with the performance metrics plot of pixel monochromatic errors.')

        ## OPD results
        if args['eval_opd_metric_rmse'] is True or 'eval_opd_metric_rmse' not in args:
            try:
                res = extract_opd_results(metrics, test_train=plot_dataset)
                model_opd_rmse = res[0]
                model_opd_std_rmse = res[1]
                model_opd_rel_rmse = res[2]
                model_opd_std_rel_rmse = res[3]

                fig = plt.figure(figsize=(12, 8))
                ax1 = fig.add_subplot(111)
                ax1.errorbar(
                    x=star_list,
                    y=model_opd_rmse,
                    yerr=model_opd_std_rmse,
                    label=run_id_no_suff,
                    alpha=0.75
                )
                plt.minorticks_on()
                ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                ax1.legend()
                ax1.set_title('Stars ' + plot_dataset + '\n' + run_id_no_suff + '.\nOPD RMSE')
                ax1.set_xlabel('Number of stars')
                ax1.set_ylabel('Absolute error')

                ax2 = ax1.twinx()
                kwargs = dict(linewidth=2, linestyle='dashed', markersize=8, marker='^', alpha=0.5)
                ax2.plot(star_list, model_opd_rel_rmse, **kwargs)
                ax2.set_ylabel('Relative error [%]')
                ax2.grid(False)
                plt.savefig(
                    plot_saving_path + plot_dataset + '-metrics-' + run_id_no_suff + '_OPD_RMSE.png'
                )
                plt.show()
            except Exception:
                print('Problem with the performance metrics plot of OPD errors.')

        ## Shape results
        if args['eval_train_shape_sr_metric_rmse'] is True or 'eval_train_shape_sr_metric_rmse' not in args or plot_dataset=='test':
            model_e1, model_e2, model_R2 = extract_shape_results(metrics, test_train=plot_dataset)
            model_e1_rmse = model_e1[0]
            model_e1_std_rmse = model_e1[1]
            model_e1_rel_rmse = model_e1[2]
            model_e1_std_rel_rmse = model_e1[3]
            model_e2_rmse = model_e2[0]
            model_e2_std_rmse = model_e2[1]
            model_e2_rel_rmse = model_e2[2]
            model_e2_std_rel_rmse = model_e2[3]
            model_rmse_R2_meanR2 = model_R2[0]
            model_std_rmse_R2_meanR2 = model_R2[1]

            # Compute Euclid relative error values
            model_e1_rel_euclid = model_e1_rmse / e1_req_euclid
            model_e2_rel_euclid = model_e2_rmse / e2_req_euclid
            model_R2_rel_euclid = model_rmse_R2_meanR2 / R2_req_euclid


            # Plot e1 and e2
            try:
                fig = plt.figure(figsize=(12, 8))
                ax1 = fig.add_subplot(111)
                ax1.errorbar(
                    x=star_list,
                    y=model_e1_rmse,
                    yerr=model_e1_std_rmse,
                    label='e1 ' + run_id_no_suff,
                    alpha=0.75
                )
                ax1.errorbar(
                    x=star_list,
                    y=model_e2_rmse,
                    yerr=model_e2_std_rmse,
                    label='e2 ' + run_id_no_suff,
                    alpha=0.75
                )
                plt.minorticks_on()
                ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                ax1.legend()
                ax1.set_title(
                    'Stars ' + plot_dataset + '\n' + run_id_no_suff +
                    '\ne1, e2 RMSE @ 3x Euclid resolution'
                )
                ax1.set_xlabel('Number of stars')
                ax1.set_ylabel('Absolute error')

                ax2 = ax1.twinx()
                kwargs = dict(linewidth=2, linestyle='dashed', markersize=8, marker='^', alpha=0.5)
                ax2.plot(star_list, model_e1_rel_euclid, **kwargs)
                ax2.plot(star_list, model_e2_rel_euclid, **kwargs)
                ax2.set_ylabel('Times over Euclid req.')
                ax2.grid(False)
                plt.savefig(
                    plot_saving_path + plot_dataset + '-metrics-' + run_id_no_suff +
                    '_shape_e1_e2_RMSE.png'
                )
                plt.show()
            except Exception:
                print('Problem with the performance metrics plot of e1/e2 errors.')

            # Plot R2
            try:
                fig = plt.figure(figsize=(12, 8))
                ax1 = fig.add_subplot(111)
                ax1.errorbar(
                    x=star_list,
                    y=model_rmse_R2_meanR2,
                    yerr=model_std_rmse_R2_meanR2,
                    label='R2 ' + run_id_no_suff,
                    alpha=0.75
                )
                plt.minorticks_on()
                ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                ax1.legend()
                ax1.set_title(
                    'Stars ' + plot_dataset + '\n' + run_id_no_suff +
                    '\nR2/<R2> RMSE @ 3x Euclid resolution'
                )
                ax1.set_xlabel('Number of stars')
                ax1.set_ylabel('Absolute error')

                ax2 = ax1.twinx()
                kwargs = dict(linewidth=2, linestyle='dashed', markersize=8, marker='^', alpha=0.5)
                ax2.plot(star_list, model_R2_rel_euclid, **kwargs)
                ax2.set_ylabel('Times over Euclid req.')
                ax2.grid(False)
                plt.savefig(
                    plot_saving_path + plot_dataset + '-metrics-' + run_id_no_suff +
                    '_shape_R2_RMSE.png'
                )
                plt.show()
            except Exception:
                print('Problem with the performance metrics plot of R2 errors.')

            ## Polychromatic pixel residual at shape measurement resolution
            try:
                res = extract_shape_pix_results(metrics, test_train=plot_dataset)
                model_polyc_shpix_rmse = res[0]
                model_polyc_shpix_std_rmse = res[1]
                model_polyc_shpix_rel_rmse = res[2]
                model_polyc_shpix_std_rel_rmse = res[3]

                fig = plt.figure(figsize=(12, 8))
                ax1 = fig.add_subplot(111)
                ax1.errorbar(
                    x=star_list,
                    y=model_polyc_shpix_rmse,
                    yerr=model_polyc_shpix_std_rmse,
                    label=run_id_no_suff,
                    alpha=0.75
                )
                plt.minorticks_on()
                ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
                ax1.legend()
                ax1.set_title(
                    'Stars ' + plot_dataset + '\n' + run_id_no_suff +
                    '\nPixel RMSE @ 3x Euclid resolution'
                )
                ax1.set_xlabel('Number of stars')
                ax1.set_ylabel('Absolute error')

                ax2 = ax1.twinx()
                kwargs = dict(linewidth=2, linestyle='dashed', markersize=8, marker='^', alpha=0.5)
                ax2.plot(star_list, model_polyc_shpix_rel_rmse, **kwargs)
                ax2.set_ylabel('Relative error [%]')
                ax2.grid(False)
                plt.savefig(
                    plot_saving_path + plot_dataset + '-metrics-' + run_id_no_suff +
                    '_poly_pixel_3xResolution_RMSE.png'
                )
                plt.show()
            except Exception:
                print(
                    'Problem with the performance metrics plot of super resolution pixel polychromatic errors.'
                )


def plot_optimisation_metrics(**args):
    r""" Plot optimisation results.
    """
    define_plot_style()

    # Define saving path
    plot_saving_path = args['base_path'] + args['plots_folder']

    # Define the number of datasets to test
    if not isinstance(args['suffix_id_name'], str):
        n_datasets = len(args['suffix_id_name'])
    else:
        n_datasets = 1

    optim_hist_file = args['base_path'] + args['optim_hist_folder']
    run_id_no_suff = args['model'] + args['base_id_name']

    # Define the metric data paths
    if not isinstance(args['suffix_id_name'], str):
        model_paths = [
            optim_hist_file + 'optim_hist_' + run_id_no_suff + _suff + '.npy'
            for _suff in args['suffix_id_name']
        ]
    else:
        model_paths = [
            optim_hist_file + 'optim_hist_' + run_id_no_suff + str(args['suffix_id_name']) + '.npy'
        ]

    print('Model paths for optimisation plots: ', model_paths)

    try:
        # Load metrics
        metrics = [np.load(_path, allow_pickle=True)[()] for _path in model_paths]
    except FileNotFoundError:
        print('The required file for the plots was not found.')
        print('Probably I am not the last job for plotting the optimisation metrics.')
        raise 0

    ## Plot the first parametric cycle
    cycle_str = 'param_cycle1'
    metric_str = 'mean_squared_error'
    val_mertric_str = 'val_mean_squared_error'

    try:
        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(111)
        for it in range(n_datasets):
            try:
                ax1.plot(
                    metrics[it][cycle_str][metric_str],
                    label=args['model'] + args['suffix_id_name'][it],
                    alpha=0.75
                )
            except KeyError as KE:
                print('Error with Key: ', KE)
        plt.yscale('log')
        plt.minorticks_on()
        ax1.legend()
        ax1.set_title('Parametric cycle 1.\n' + run_id_no_suff + '_' + cycle_str)
        ax1.set_xlabel('Number of epoch')
        ax1.set_ylabel('Training MSE')

        ax2 = ax1.twinx()
        kwargs = dict(linewidth=2, linestyle='dashed', markersize=2, marker='^', alpha=0.5)
        for it in range(n_datasets):
            try:
                ax2.plot(metrics[it][cycle_str][val_mertric_str], **kwargs)
            except KeyError as KE:
                print('Error with Key: ', KE)
        ax2.set_ylabel('Validation MSE')
        ax2.grid(False)
        plt.savefig(plot_saving_path + 'optim_' + run_id_no_suff + '_' + cycle_str + '.png')
        plt.show()
    except Exception:
        print('Problem with the plot of the optimisation metrics of the first parametric cycle.')

    # Plot the first non-parametric cycle
    if args['model'] != 'param':
        try:
            cycle_str = 'nonparam_cycle1'
            metric_str = 'mean_squared_error'
            val_mertric_str = 'val_mean_squared_error'

            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_subplot(111)
            for it in range(n_datasets):
                try:
                    ax1.plot(
                        metrics[it][cycle_str][metric_str],
                        label=args['model'] + args['suffix_id_name'][it],
                        alpha=0.75
                    )
                except KeyError as KE:
                    print('Error with Key: ', KE)
            plt.yscale('log')
            plt.minorticks_on()
            ax1.legend()
            ax1.set_title('Non-parametric cycle 1.\n' + run_id_no_suff + '_' + cycle_str)
            ax1.set_xlabel('Number of epoch')
            ax1.set_ylabel('Training MSE')

            ax2 = ax1.twinx()
            kwargs = dict(linewidth=2, linestyle='dashed', markersize=2, marker='^', alpha=0.5)
            for it in range(n_datasets):
                try:
                    ax2.plot(metrics[it][cycle_str][val_mertric_str], **kwargs)
                except KeyError as KE:
                    print('Error with Key: ', KE)
            ax2.set_ylabel('Validation MSE')
            ax2.grid(False)
            plt.savefig(plot_saving_path + 'optim_' + run_id_no_suff + '_' + cycle_str + '.png')
            plt.show()
        except Exception:
            print(
                'Problem with the plot of the optimisation metrics of the first non-parametric cycle.'
            )

    ## Plot the second parametric cycle
    if cycle_str in metrics[0]:
        cycle_str = 'param_cycle2'
        metric_str = 'mean_squared_error'
        val_mertric_str = 'val_mean_squared_error'

        try:
            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_subplot(111)
            for it in range(n_datasets):
                try:
                    ax1.plot(
                        metrics[it][cycle_str][metric_str],
                        label=args['model'] + args['suffix_id_name'][it],
                        alpha=0.75
                    )
                except KeyError as KE:
                    print('Error with Key: ', KE)
            plt.yscale('log')
            plt.minorticks_on()
            ax1.legend()
            ax1.set_title('Parametric cycle 2.\n' + run_id_no_suff + '_' + cycle_str)
            ax1.set_xlabel('Number of epoch')
            ax1.set_ylabel('Training MSE')

            ax2 = ax1.twinx()
            kwargs = dict(linewidth=2, linestyle='dashed', markersize=2, marker='^', alpha=0.5)
            for it in range(n_datasets):
                try:
                    ax2.plot(metrics[it][cycle_str][val_mertric_str], **kwargs)
                except KeyError as KE:
                    print('Error with Key: ', KE)
            ax2.set_ylabel('Validation MSE')
            ax2.grid(False)
            plt.savefig(plot_saving_path + 'optim_' + run_id_no_suff + '_' + cycle_str + '.png')
            plt.show()
        except Exception:
            print(
                'Problem with the plot of the optimisation metrics of the second parametric cycle.'
            )

    ## Plot the second non-parametric cycle
    if cycle_str in metrics[0]:
        cycle_str = 'nonparam_cycle2'
        metric_str = 'mean_squared_error'
        val_mertric_str = 'val_mean_squared_error'

        try:
            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_subplot(111)
            for it in range(n_datasets):
                try:
                    ax1.plot(
                        metrics[it][cycle_str][metric_str],
                        label=args['model'] + args['suffix_id_name'][it],
                        alpha=0.75
                    )
                except KeyError as KE:
                    print('Error with Key: ', KE)
            plt.yscale('log')
            plt.minorticks_on()
            ax1.legend()
            ax1.set_title('Non-parametric cycle 2.\n' + run_id_no_suff + '_' + cycle_str)
            ax1.set_xlabel('Number of epoch')
            ax1.set_ylabel('Training MSE')

            ax2 = ax1.twinx()
            kwargs = dict(linewidth=2, linestyle='dashed', markersize=2, marker='^', alpha=0.5)
            for it in range(n_datasets):
                try:
                    ax2.plot(metrics[it][cycle_str][val_mertric_str], **kwargs)
                except KeyError as KE:
                    print('Error with Key: ', KE)
            ax2.set_ylabel('Validation MSE')
            ax2.grid(False)
            plt.savefig(plot_saving_path + 'optim_' + run_id_no_suff + '_' + cycle_str + '.png')
            plt.show()
        except Exception:
            print(
                'Problem with the plot of the optimisation metrics of the second non-parametric cycle.'
            )


def define_plot_style():
    # Define plot paramters
    plot_style = {
        'figure.figsize': (12, 8),
        'figure.dpi': 200,
        'figure.autolayout': True,
        'lines.linewidth': 2,
        'lines.linestyle': '-',
        'lines.marker': 'o',
        'lines.markersize': 10,
        'legend.fontsize': 20,
        'legend.loc': 'best',
        'axes.titlesize': 24,
        'font.size': 16
    }
    mpl.rcParams.update(plot_style)
    # Use seaborn style
    sns.set()


def extract_poly_results(metrics_dicts, test_train='test'):

    if test_train == 'test':
        first_key = 'test_metrics'
    elif test_train == 'train':
        first_key = 'train_metrics'
    else:
        raise ValueError

    n_dicts = len(metrics_dicts)

    polyc_rmse = np.zeros(n_dicts)
    polyc_std_rmse = np.zeros(n_dicts)
    polyc_rel_rmse = np.zeros(n_dicts)
    polyc_std_rel_rmse = np.zeros(n_dicts)

    for it in range(n_dicts):
        polyc_rmse[it] = metrics_dicts[it][first_key]['poly_metric']['rmse']
        polyc_std_rmse[it] = metrics_dicts[it][first_key]['poly_metric']['std_rmse']
        polyc_rel_rmse[it] = metrics_dicts[it][first_key]['poly_metric']['rel_rmse']
        polyc_std_rel_rmse[it] = metrics_dicts[it][first_key]['poly_metric']['std_rel_rmse']

    return polyc_rmse, polyc_std_rmse, polyc_rel_rmse, polyc_std_rel_rmse


def extract_opd_results(metrics_dicts, test_train='test'):

    if test_train == 'test':
        first_key = 'test_metrics'
    elif test_train == 'train':
        first_key = 'train_metrics'
    else:
        raise ValueError

    n_dicts = len(metrics_dicts)

    opd_rmse = np.zeros(n_dicts)
    opd_std_rmse = np.zeros(n_dicts)
    opd_rel_rmse = np.zeros(n_dicts)
    opd_std_rel_rmse = np.zeros(n_dicts)

    for it in range(n_dicts):
        opd_rmse[it] = metrics_dicts[it][first_key]['opd_metric']['rmse_opd']
        opd_std_rmse[it] = metrics_dicts[it][first_key]['opd_metric']['rmse_std_opd']
        opd_rel_rmse[it] = metrics_dicts[it][first_key]['opd_metric']['rel_rmse_opd']
        opd_std_rel_rmse[it] = metrics_dicts[it][first_key]['opd_metric']['rel_rmse_std_opd']

    return opd_rmse, opd_std_rmse, opd_rel_rmse, opd_std_rel_rmse


def extract_shape_results(metrics_dicts, test_train='test'):

    if test_train == 'test':
        first_key = 'test_metrics'
    elif test_train == 'train':
        first_key = 'train_metrics'
    else:
        raise ValueError

    n_dicts = len(metrics_dicts)

    e1_rmse = np.zeros(n_dicts)
    e1_std_rmse = np.zeros(n_dicts)
    e1_rel_rmse = np.zeros(n_dicts)
    e1_std_rel_rmse = np.zeros(n_dicts)

    e2_rmse = np.zeros(n_dicts)
    e2_std_rmse = np.zeros(n_dicts)
    e2_rel_rmse = np.zeros(n_dicts)
    e2_std_rel_rmse = np.zeros(n_dicts)

    rmse_R2_meanR2 = np.zeros(n_dicts)
    std_rmse_R2_meanR2 = np.zeros(n_dicts)

    for it in range(n_dicts):
        e1_rmse[it] = metrics_dicts[it][first_key]['shape_results_dict']['rmse_e1']
        e1_std_rmse[it] = metrics_dicts[it][first_key]['shape_results_dict']['std_rmse_e1']
        e1_rel_rmse[it] = metrics_dicts[it][first_key]['shape_results_dict']['rel_rmse_e1']
        e1_std_rel_rmse[it] = metrics_dicts[it][first_key]['shape_results_dict']['std_rel_rmse_e1']

        e2_rmse[it] = metrics_dicts[it][first_key]['shape_results_dict']['rmse_e2']
        e2_std_rmse[it] = metrics_dicts[it][first_key]['shape_results_dict']['std_rmse_e2']
        e2_rel_rmse[it] = metrics_dicts[it][first_key]['shape_results_dict']['rel_rmse_e2']
        e2_std_rel_rmse[it] = metrics_dicts[it][first_key]['shape_results_dict']['std_rel_rmse_e2']

        rmse_R2_meanR2[it] = metrics_dicts[it][first_key]['shape_results_dict']['rmse_R2_meanR2']
        std_rmse_R2_meanR2[it] = metrics_dicts[it][first_key]['shape_results_dict'][
            'std_rmse_R2_meanR2']

    e1 = [e1_rmse, e1_std_rmse, e1_rel_rmse, e1_std_rel_rmse]
    e2 = [e2_rmse, e2_std_rmse, e2_rel_rmse, e2_std_rel_rmse]
    R2 = [rmse_R2_meanR2, std_rmse_R2_meanR2]

    return e1, e2, R2


def extract_shape_pix_results(metrics_dicts, test_train='test'):

    if test_train == 'test':
        first_key = 'test_metrics'
    elif test_train == 'train':
        first_key = 'train_metrics'
    else:
        raise ValueError

    n_dicts = len(metrics_dicts)

    polyc_rmse = np.zeros(n_dicts)
    polyc_std_rmse = np.zeros(n_dicts)
    polyc_rel_rmse = np.zeros(n_dicts)
    polyc_std_rel_rmse = np.zeros(n_dicts)

    for it in range(n_dicts):
        polyc_rmse[it] = metrics_dicts[it][first_key]['shape_results_dict']['pix_rmse']
        polyc_std_rmse[it] = metrics_dicts[it][first_key]['shape_results_dict']['pix_rmse_std']
        polyc_rel_rmse[it] = metrics_dicts[it][first_key]['shape_results_dict']['rel_pix_rmse']
        polyc_std_rel_rmse[it] = metrics_dicts[it][first_key]['shape_results_dict'][
            'rel_pix_rmse_std']

    return polyc_rmse, polyc_std_rmse, polyc_rel_rmse, polyc_std_rel_rmse