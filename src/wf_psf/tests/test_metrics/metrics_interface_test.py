from unittest.mock import patch, MagicMock
import pytest
from wf_psf.metrics.metrics_interface import evaluate_model
from wf_psf.data.training_preprocessing import DataHandler


@pytest.fixture
def mock_metrics_params():
    return MagicMock(
        eval_mono_metric=True,
        eval_opd_metric=False,
        eval_test_shape_results_dict=True,
        eval_train_shape_results_dict=False,
    )


@pytest.fixture
def mock_trained_model_params():
    return MagicMock(model_params=MagicMock(model_name="mock_model"), id_name="mock_id")


@pytest.fixture
def mock_data():
    # Create mock instances of the required attributes
    mock_data_params = MagicMock()
    mock_simPSF = MagicMock()

    # Mock the `data_params` dictionary for "train" and "test" data
    mock_data_params.train = MagicMock()
    mock_data_params.test = MagicMock()

    # Initialize the DataHandler object with the mocked attributes
    mock_data_handler = MagicMock(spec=DataHandler)
    mock_data_handler.data_params = mock_data_params
    mock_data_handler.simPSF = mock_simPSF
    mock_data_handler.n_bins_lambda = 10  # Example number of bins

    # Mock the `training_data` and `test_data` attributes
    mock_data_handler.training_data = MagicMock()
    mock_data_handler.test_data = MagicMock()

    mock_data_handler.training_data.dataset = {
        "positions": "train_positions",
        "noisy_stars": "train_noisy_stars",
        "SEDs": "train_SEDs",
        "C_poly": "train_C_poly",
    }

    mock_data_handler.test_data.dataset = {
        "positions": "test_positions",
        "noisy_stars": "test_noisy_stars",
        "SEDs": "test_SEDs",
        "C_poly": "test_C_poly",
    }
    mock_data_handler.sed_data = "mock_sed_data"

    # Return the mocked DataHandler instance
    return mock_data_handler


@pytest.mark.parametrize(
    "mono_flag,opd_flag,train_shape_flag,test_shape_flag,expected_calls",
    [
        (True, False, False, False, ["mono"]),
        (False, True, False, False, ["opd"]),
        (False, False, True, True, ["shape"]),
        (True, True, True, True, ["mono", "opd", "shape"]),
        (False, False, False, False, []),
    ],
)
def test_evaluate_model_flags(
    mono_flag,
    opd_flag,
    train_shape_flag,
    test_shape_flag,
    expected_calls,
    mock_trained_model_params,
    mock_data,
    mock_psf_model,
):
    metrics_params = MagicMock(
        eval_mono_metric=mono_flag,
        eval_opd_metric=opd_flag,
        eval_train_shape_results_dict=train_shape_flag,
        eval_test_shape_results_dict=test_shape_flag,
        plotting_config=None,
    )

    with (
        patch(
            "wf_psf.metrics.metrics_interface.MetricsParamsHandler.evaluate_metrics_polychromatic_lowres"
        ) as mock_poly,
        patch(
            "wf_psf.metrics.metrics_interface.MetricsParamsHandler.evaluate_metrics_mono_rmse"
        ) as mock_mono,
        patch(
            "wf_psf.metrics.metrics_interface.MetricsParamsHandler.evaluate_metrics_opd"
        ) as mock_opd,
        patch(
            "wf_psf.metrics.metrics_interface.MetricsParamsHandler.evaluate_metrics_shape"
        ) as mock_shape,
        patch("numpy.save") as mock_np_save,
    ):
        evaluate_model(
            metrics_params=metrics_params,
            trained_model_params=mock_trained_model_params,
            data=mock_data,
            psf_model=mock_psf_model,
            weights_path="/mock/path",
            metrics_output="/mock/output",
        )

        # Assertions
        assert mock_poly.call_count == 2  # Called twice, once for each dataset
        assert (mock_mono.called) == ("mono" in expected_calls)
        assert (mock_opd.called) == ("opd" in expected_calls)
        assert (mock_shape.called) == ("shape" in expected_calls)
        assert mock_np_save.called


def test_missing_ground_truth_model_raises(
    mock_metrics_params, mock_trained_model_params, mock_data, mock_psf_model
):
    # Set ground_truth_model to None to simulate missing config
    mock_metrics_params.ground_truth_model = None

    from wf_psf.metrics.metrics_interface import evaluate_model

    import pytest

    with pytest.raises(AttributeError):
        evaluate_model(
            metrics_params=mock_metrics_params,
            trained_model_params=mock_trained_model_params,
            data=mock_data,
            psf_model=mock_psf_model,
            weights_path="/mock/weights/path",
            metrics_output="/mock/metrics/output",
        )


@pytest.mark.parametrize("plotting_config", [None, "mock_plot_config.yaml"])
def test_plotting_config_passed(
    plotting_config, mock_trained_model_params, mock_data, mock_psf_model
):
    metrics_params = MagicMock(
        eval_mono_metric=False,
        eval_opd_metric=False,
        eval_train_shape_results_dict=False,
        eval_test_shape_results_dict=True,
        plotting_config=plotting_config,
        ground_truth_model=MagicMock(
            model_params=MagicMock(model_name="mock_gt_model")
        ),
    )

    with (
        patch(
            "wf_psf.metrics.metrics_interface.MetricsParamsHandler.evaluate_metrics_polychromatic_lowres"
        ) as mock_poly,
        patch(
            "wf_psf.metrics.metrics_interface.MetricsParamsHandler.evaluate_metrics_shape"
        ) as mock_shape,
        patch("numpy.save") as mock_np_save,
    ):
        evaluate_model(
            metrics_params=metrics_params,
            trained_model_params=mock_trained_model_params,
            data=mock_data,
            psf_model=mock_psf_model,
            weights_path="/mock/path",
            metrics_output="/mock/output",
        )

        assert mock_poly.call_count == 2  # Called twice, once for each dataset
        mock_shape.assert_called_once()
        mock_np_save.assert_called_once()


def test_evaluate_model(
    mock_metrics_params, mock_trained_model_params, mock_data, mock_psf_model, mocker
):
    # Mock the metric functions
    with (
        patch(
            "wf_psf.metrics.metrics_interface.MetricsParamsHandler.evaluate_metrics_polychromatic_lowres",
            new_callable=MagicMock,
        ) as mock_evaluate_poly_metric,
        patch(
            "wf_psf.metrics.metrics_interface.MetricsParamsHandler.evaluate_metrics_mono_rmse",
            new_callable=MagicMock,
        ) as mock_evaluate_mono_metric,
        patch(
            "wf_psf.metrics.metrics_interface.MetricsParamsHandler.evaluate_metrics_opd",
            new_callable=MagicMock,
        ) as mock_evaluate_opd_metric,
        patch(
            "wf_psf.metrics.metrics_interface.MetricsParamsHandler.evaluate_metrics_shape",
            new_callable=MagicMock,
        ) as mock_evaluate_shape_results_dict,
        patch("numpy.save", new_callable=MagicMock) as mock_np_save,
    ):
        # Mock the logger
        _ = mocker.patch("wf_psf.metrics.metrics_interface.logger")

        # Call evaluate_model
        evaluate_model(
            metrics_params=mock_metrics_params,
            trained_model_params=mock_trained_model_params,
            data=mock_data,
            psf_model=mock_psf_model,
            weights_path="/mock/weights/path",
            metrics_output="/mock/metrics/output",
        )

        # Assertions for metric functions
        assert (
            mock_evaluate_poly_metric.call_count == 2
        )  # Called twice, once for each dataset
        assert (
            mock_evaluate_mono_metric.call_count == 2
        )  # Called twice, once for each dataset
        mock_evaluate_opd_metric.assert_not_called()  # Should not be called because the flag is False
        mock_evaluate_shape_results_dict.assert_called_once()  # Should be called only for the test dataset

        # Assertion for np.save (should be called **once**)
        mock_np_save.assert_called_once()

        # Validate the np.save call arguments
        output_path, saved_data = mock_np_save.call_args[0]  # Extract arguments
        assert (
            "/mock/metrics/output/metrics-mock_modelmock_id" in output_path
        )  # Ensure correct path format
        assert isinstance(saved_data, dict)  # Ensure data being saved is a dictionary
