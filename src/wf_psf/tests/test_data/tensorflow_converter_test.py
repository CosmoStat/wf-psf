import pytest
import numpy as np
import tensorflow as tf
from wf_psf.data.tensorflow_converter import TensorFlowDatasetConverter
from wf_psf.tests.test_data.test_data_utils import assert_tensor


@pytest.fixture
def mock_simPSF(mocker):
    """Mock simPSF instance to avoid real SED processing."""
    mock = mocker.Mock(name="SimPSFToolkit")
    return mock


@pytest.fixture
def converter(mock_simPSF):
    """TensorFlowDatasetConverter instance with mock simPSF."""
    return TensorFlowDatasetConverter(simPSF=mock_simPSF, n_bins_lambda=10)


@pytest.fixture
def mock_process_seds(mocker, converter):
    """
    Mock process_seds to avoid real SED processing in converter tests.

    Returns a tensor of shape (N, n_bins_lambda, 3) to mimic real output.
    """

    def _mock_process_seds(sed_data):
        n_sources = len(sed_data)
        return tf.zeros((n_sources, 10, 3), dtype=tf.float32)

    return mocker.patch.object(
        converter, "process_seds", side_effect=_mock_process_seds
    )


@pytest.fixture
def simulation_train_dict():
    """Mock simulation training dataset dict."""
    return {
        "positions": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "noisy_stars": np.random.randn(2, 32, 32).astype(np.float32),
        "SEDs": np.random.randn(2, 3, 2).astype(np.float32),
        "masks": np.ones((2, 32, 32), dtype=np.float32),
    }


@pytest.fixture
def simulation_test_dict():
    """Mock simulation test dataset dict."""
    return {
        "positions": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "stars": np.random.randn(2, 32, 32).astype(np.float32),
        "SEDs": np.random.randn(2, 3, 2).astype(np.float32),
        "masks": np.ones((2, 32, 32), dtype=np.float32),
    }


@pytest.fixture
def inference_dict():
    """Mock inference dataset dict."""
    return {
        "positions": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "sources": np.random.randn(2, 32, 32).astype(np.float32),
        "masks": np.ones((2, 32, 32), dtype=np.float32),
    }


class TestConvertDict:
    """Tests for TensorFlowDatasetConverter.convert_dict()."""

    @pytest.mark.parametrize(
        "dataset_type,input_fixture,expected_keys,forbidden_keys",
        [
            (
                "training",
                "simulation_train_dict",
                {"positions", "noisy_stars", "SEDs"},
                {"stars"},
            ),
            (
                "test",
                "simulation_test_dict",
                {"positions", "stars", "SEDs"},
                {"noisy_stars"},
            ),
            (
                "inference",
                "inference_dict",
                {"positions", "sources"},
                set(),
            ),
        ],
    )
    def test_required_and_forbidden_keys(
        self,
        request,
        converter,
        dataset_type,
        input_fixture,
        expected_keys,
        forbidden_keys,
        mock_process_seds,
    ):
        dataset_dict = request.getfixturevalue(input_fixture)

        result = converter.convert_dict(dataset_dict, dataset_type=dataset_type)

        assert expected_keys.issubset(result.keys())
        assert forbidden_keys.isdisjoint(result.keys())

    def test_training_tensor_properties(
        self, converter, simulation_train_dict, mock_process_seds
    ):
        result = converter.convert_dict(simulation_train_dict, dataset_type="training")

        # All values are tensors
        assert all(tf.is_tensor(v) for v in result.values())

        # Positions dtype and shape
        assert_tensor(
            result["positions"],
            expected_shape=(2, 2),
            expected_dtype=tf.float32,
        )

        # Data preserved
        np.testing.assert_array_almost_equal(
            result["positions"].numpy(),
            simulation_train_dict["positions"],
        )
        np.testing.assert_array_almost_equal(
            result["noisy_stars"].numpy(),
            simulation_train_dict["noisy_stars"],
        )

    @pytest.mark.parametrize("remove_masks", [False, True])
    def test_mask_inclusion_behavior(
        self,
        converter,
        simulation_train_dict,
        mock_process_seds,
        remove_masks,
    ):
        if remove_masks:
            simulation_train_dict.pop("masks", None)

        result = converter.convert_dict(simulation_train_dict, dataset_type="training")

        if remove_masks:
            assert "masks" not in result
        else:
            assert "masks" in result
            assert_tensor(result["masks"])

    @pytest.mark.parametrize(
        "external_seds,expect_call,expect_seds_key",
        [
            (None, True, True),  # from dict
            ("external", True, True),  # override
            ("missing", False, False),  # none anywhere
        ],
    )
    def test_seds_resolution(
        self,
        converter,
        simulation_train_dict,
        mock_process_seds,
        external_seds,
        expect_call,
        expect_seds_key,
    ):
        if external_seds == "external":
            external_seds = np.random.randn(2, 5, 2).astype(np.float32)
        elif external_seds == "missing":
            simulation_train_dict.pop("SEDs", None)
            external_seds = None

        result = converter.convert_dict(
            simulation_train_dict,
            dataset_type="training",
            seds=external_seds,
        )

        if expect_call:
            if external_seds is not None:
                mock_process_seds.assert_called_once_with(external_seds)
            else:
                mock_process_seds.assert_called_once_with(simulation_train_dict["SEDs"])
        else:
            mock_process_seds.assert_not_called()

        assert ("SEDs" in result) is expect_seds_key

    def test_missing_positions_raises_error(self, converter, mock_process_seds):
        dataset_dict = {
            "noisy_stars": np.random.randn(2, 32, 32).astype(np.float32),
        }

        with pytest.raises(KeyError):
            converter.convert_dict(dataset_dict, dataset_type="training")


class TestTensorFlowDatasetConverter:
    """Unit tests for TensorFlowDatasetConverter class."""

    def test_process_seds(self, simPSF):
        """
        Test SED processing converts raw SEDs to correct TensorFlow format.

        Note: process_seds() does NOT return raw SED values.
        It returns processed components via generate_SED_elems_in_tensorflow:
        - feasible_N:   integer N values for diffraction computation
        - feasible_wv:  feasible wavelengths in [um]
        - SED_norm:     normalized, weighted SED values

        Shape: (N_sources, n_bins_lambda, 3)
        """

        converter = TensorFlowDatasetConverter(simPSF, n_bins_lambda=10)

        # Raw SED input: (N_sources, n_wavelengths, 2) where 2 = s[wavelength_nm, flux]
        raw_seds = np.array(
            [
                [[400.0, 0.1], [500.0, 0.2], [600.0, 0.3]],  # Source 1
                [[400.0, 0.4], [500.0, 0.5], [600.0, 0.6]],  # Source 2
            ]
        )

        processed = converter.process_seds(raw_seds)

        # Test shape only - values are transformed, not preserved
        assert isinstance(processed, tf.Tensor)
        assert processed.dtype == tf.float32
        assert processed.shape == (2, 10, 3)  # (N_sources, n_bins_lambda, 3 components)

        # Test each component has physically meaningful values
        feasible_N = processed[:, :, 0]  # Integer N values
        feasible_wv = processed[:, :, 1]  # Wavelengths in [um]
        SED_norm = processed[:, :, 2]  # Normalized SED values

        # feasible_N should be positive even integers
        assert tf.reduce_all(feasible_N > 0)
        assert tf.reduce_all(feasible_N % 2 == 0)

        # feasible_wv should be in [um] range (visible light ~0.4-0.9 um)
        assert tf.reduce_all(feasible_wv > 0.3)
        assert tf.reduce_all(feasible_wv < 1.0)

        # SED_norm should sum to ~1.0 per source (normalized)
        sed_sums = tf.reduce_sum(SED_norm, axis=1)  # Sum over wavelength bins
        np.testing.assert_allclose(
            sed_sums.numpy(),
            np.ones(2),  # One per source
            rtol=1e-5,
            err_msg="SED_norm should sum to 1.0 per source",
        )

        # Test deterministic - same input gives same output
        processed_again = converter.process_seds(raw_seds)
        np.testing.assert_array_equal(processed.numpy(), processed_again.numpy())

    def test_process_seds_none_raises_error(self, simPSF):
        """Test that None SEDs raise ValueError."""
        converter = TensorFlowDatasetConverter(simPSF, n_bins_lambda=10)

        with pytest.raises(ValueError, match="SED data must be provided"):
            converter.process_seds(None)

    def test_process_seds_single_source(self, simPSF):
        """Test SED processing works for a single source."""
        converter = TensorFlowDatasetConverter(simPSF, n_bins_lambda=10)

        raw_seds = np.array(
            [
                [[400.0, 0.1], [500.0, 0.2], [600.0, 0.3]],  # Single source
            ]
        )

        processed = converter.process_seds(raw_seds)

        assert processed.shape == (1, 10, 3)

    def test_process_seds_output_dtype(self, simPSF):
        """Test output is always float32 regardless of input dtype."""
        converter = TensorFlowDatasetConverter(simPSF, n_bins_lambda=10)

        # Input as float64
        raw_seds = np.array(
            [
                [[400.0, 0.1], [500.0, 0.2]],
            ],
            dtype=np.float64,
        )

        processed = converter.process_seds(raw_seds)

        # Should always return float32 for training efficiency
        assert processed.dtype == tf.float32


class TestConvertInferenceData:
    """Tests for TensorFlowDatasetConverter.convert_inference_data()."""

    @pytest.fixture
    def positions(self):
        """Sample positions array."""
        return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    @pytest.fixture
    def full_inference_data(self):
        """Full set of inference data arrays."""
        return {
            "positions": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "sources": np.random.randn(2, 32, 32).astype(np.float32),
            "masks": np.ones((2, 32, 32), dtype=np.float32),
            "seds": np.random.randn(2, 3, 2).astype(np.float32),
        }

    @pytest.mark.parametrize(
        "sources, masks, seds, expected_keys",
        [
            # Positions only
            (
                None,
                None,
                None,
                {"positions"},
            ),
            # Positions + sources
            (
                np.random.randn(2, 32, 32).astype(np.float32),
                None,
                None,
                {"positions", "sources"},
            ),
            # Positions + masks
            (
                None,
                np.ones((2, 32, 32), dtype=np.float32),
                None,
                {"positions", "masks"},
            ),
            # Positions + seds
            (
                None,
                None,
                np.random.randn(2, 3, 2).astype(np.float32),
                {"positions", "seds"},
            ),
            # All fields
            (
                np.random.randn(2, 32, 32).astype(np.float32),
                np.ones((2, 32, 32), dtype=np.float32),
                np.random.randn(2, 3, 2).astype(np.float32),
                {"positions", "sources", "masks", "seds"},
            ),
        ],
        ids=[
            "positions_only",
            "positions_and_sources",
            "positions_and_masks",
            "positions_and_seds",
            "all_fields",
        ],
    )
    def test_output_keys(
        self,
        converter,
        positions,
        sources,
        masks,
        seds,
        expected_keys,
        mock_process_seds,
    ):
        """Test result contains exactly the expected keys for each input combination."""
        result = converter.convert_inference_data(
            positions=positions,
            sources=sources,
            masks=masks,
            seds=seds,
        )

        assert set(result.keys()) == expected_keys

    @pytest.mark.parametrize(
        "field, array, expected_shape",
        [
            ("positions", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), (2, 2)),
            ("sources", np.random.randn(2, 32, 32).astype(np.float32), (2, 32, 32)),
            ("masks", np.ones((2, 32, 32), dtype=np.float32), (2, 32, 32)),
        ],
        ids=["positions", "sources", "masks"],
    )
    def test_tensor_shapes_and_dtype(
        self, converter, field, array, expected_shape, mock_process_seds
    ):
        """Test each field produces correct tensor shape and float32 dtype."""
        positions = (
            array
            if field == "positions"
            else np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        )
        kwargs = {
            "positions": positions,
            **({field: array} if field != "positions" else {}),
        }

        result = converter.convert_inference_data(**kwargs)

        assert_tensor(
            result[field], expected_shape=expected_shape, expected_dtype=tf.float32
        )

    @pytest.mark.parametrize(
        "field, value",
        [
            ("sources", None),
            ("masks", None),
            ("seds", None),
        ],
        ids=["sources_none", "masks_none", "seds_none"],
    )
    def test_optional_fields_absent_when_none(
        self, converter, positions, field, value, mock_process_seds
    ):
        """Test optional fields are excluded from result when None."""
        result = converter.convert_inference_data(positions=positions, **{field: value})

        assert field not in result

    @pytest.mark.parametrize(
        "provide_seds, expected_call_count",
        [
            (True, 1),
            (False, 0),
        ],
        ids=["seds_provided", "no_seds"],
    )
    def test_process_seds_call_count(
        self, converter, positions, provide_seds, expected_call_count, mock_process_seds
    ):
        """Test process_seds is called only when SEDs are provided."""
        seds = np.random.randn(2, 3, 2).astype(np.float32) if provide_seds else None

        converter.convert_inference_data(positions=positions, seds=seds)

        assert mock_process_seds.call_count == expected_call_count

    def test_positions_values_preserved(self, converter, positions):
        """Test position values are numerically preserved after conversion."""
        result = converter.convert_inference_data(positions=positions)

        np.testing.assert_array_almost_equal(result["positions"].numpy(), positions)

    def test_positions_required(self, converter):
        """Test that omitting positions raises TypeError."""
        with pytest.raises(TypeError):
            converter.convert_inference_data()
