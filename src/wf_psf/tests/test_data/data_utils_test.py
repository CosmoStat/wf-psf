import numpy as np
import pytest

from wf_psf.data.data_utils import DatasetContainer


class TestDatasetContainer:
    """Tests for DatasetContainer."""

    def test_dict_style_access(self):
        data = {"x": np.array([1, 2, 3])}
        container = DatasetContainer(data)

        assert np.array_equal(container["x"], data["x"])

    def test_attribute_style_access(self):
        data = {"x": np.array([1, 2, 3])}
        container = DatasetContainer(data)

        assert np.array_equal(container.x, data["x"])

    def test_len_and_iter(self):
        data = {"a": 1, "b": 2}
        container = DatasetContainer(data)

        assert len(container) == 2
        assert set(iter(container)) == {"a", "b"}

    def test_missing_attribute_raises(self):
        container = DatasetContainer({})

        with pytest.raises(AttributeError):
            _ = container.missing_key

    def test_set_and_delete_item(self):
        container = DatasetContainer({})

        container["a"] = 42
        assert container["a"] == 42

        del container["a"]
        assert "a" not in container.to_dict()
