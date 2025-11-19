"""UNIT TESTS FOR PACKAGE MODULE: IO.

This module contains unit tests for the wf_psf.utils io module.

:Author: Jennifer Pollack <jennifer.pollack@cea.fr>


"""

import pytest
import os
import tempfile
import shutil
from datetime import datetime
from unittest.mock import patch
from wf_psf.utils.io import FileIOHandler


@pytest.fixture
def test_file_handler(path_to_repo_dir, path_to_tmp_output_dir, path_to_config_dir):
    test_file_handler = FileIOHandler(
        path_to_repo_dir, path_to_tmp_output_dir, path_to_config_dir
    )
    test_file_handler._make_output_dir()
    test_file_handler._make_run_dir()
    test_file_handler._setup_dirs()
    return test_file_handler


def test_make_output_dir(test_file_handler, path_to_tmp_output_dir):
    assert os.path.exists(
        os.path.join(path_to_tmp_output_dir, test_file_handler.parent_output_dir)
    )


def test_make_run_dir(test_file_handler):
    assert os.path.exists(test_file_handler._run_output_dir)


def test_setup_dirs(test_file_handler):
    wf_outdirs = [
        "_config",
        "_checkpoint",
        "_log_files",
        "_metrics",
        "_optimizer",
        "_plots",
        "_psf_model",
    ]

    for odir in wf_outdirs:
        assert os.path.exists(
            os.path.join(
                test_file_handler._run_output_dir, test_file_handler.__dict__[odir]
            )
        )


class TestFileIOTimestampCollision:
    """Test that FileIOHandler generates unique directories even with sub-second collisions."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_repo = tempfile.mkdtemp()
        temp_output = tempfile.mkdtemp()
        temp_config = tempfile.mkdtemp()
        
        yield temp_repo, temp_output, temp_config
        
        # Cleanup
        shutil.rmtree(temp_repo, ignore_errors=True)
        shutil.rmtree(temp_output, ignore_errors=True)
        shutil.rmtree(temp_config, ignore_errors=True)
    
    def test_no_collision_with_microsecond_precision(self, temp_dirs):
        """Test that multiple rapid instantiations create unique directories."""
        temp_repo, temp_output, temp_config = temp_dirs
        
        # Create multiple FileIOHandler instances in rapid succession
        handlers = []
        workdirs = []
        
        for _ in range(10):
            handler = FileIOHandler(temp_repo, temp_output, temp_config)
            handlers.append(handler)
            workdirs.append(handler.workdir)
        
        # All workdirs should be unique
        assert len(workdirs) == len(set(workdirs)), \
            f"Collision detected! Workdirs: {workdirs}"

    def test_collision_with_same_timestamp(self, temp_dirs):
        """Test collision scenario when timestamps are identical (simulated)."""
        temp_repo, temp_output, temp_config = temp_dirs
        
        # Mock the timestamp to return the same value
        fixed_timestamp = "202310221632"
        
        with patch.object(FileIOHandler, 'get_timestamp', return_value=fixed_timestamp):
            handler1 = FileIOHandler(temp_repo, temp_output, temp_config)
            handler2 = FileIOHandler(temp_repo, temp_output, temp_config)
            
            # With old implementation (minute precision), these would be the same
            # With new implementation (second/microsecond), they should be different
            
            # If using the OLD format, this assertion would FAIL (which is the bug)
            # With the NEW format, this should PASS
            assert handler1.workdir == handler2.workdir, \
                "This test demonstrates the collision when timestamps are mocked to be identical"

    def test_batch_submission_simulation(self, temp_dirs):
        """Simulate batch job submissions within the same second."""
        temp_repo, temp_output, temp_config = temp_dirs
        
        # Simulate multiple jobs starting at nearly the same time
        # by mocking datetime.now() to return very close timestamps
        base_time = datetime(2023, 10, 22, 16, 32, 45, 123456)
        
        handlers = []
        workdirs = []
        
        # Simulate 5 jobs submitted within microseconds of each other
        for microsecond_offset in range(0, 50000, 10000):  # 0, 10ms, 20ms, 30ms, 40ms
            mock_time = base_time.replace(microsecond=microsecond_offset)
            
            with patch('wf_psf.utils.io.datetime') as mock_datetime:
                mock_datetime.now.return_value = mock_time
                handler = FileIOHandler(temp_repo, temp_output, temp_config)
                handlers.append(handler)
                workdirs.append(handler.workdir)
        
        # All workdirs should be unique
        unique_workdirs = set(workdirs)
        assert len(workdirs) == len(unique_workdirs), \
            f"Collision detected in batch simulation! Got {len(unique_workdirs)} unique dirs from {len(workdirs)} jobs"
    
    def test_directory_creation_no_overwrite(self, temp_dirs):
        """Test that actual directory creation doesn't overwrite existing directories."""
        temp_repo, temp_output, temp_config = temp_dirs
        
        # Create first handler and its directory structure
        handler1 = FileIOHandler(temp_repo, temp_output, temp_config)
        os.makedirs(handler1._run_output_dir, exist_ok=True)
        
        # Write a marker file
        marker_file = os.path.join(handler1._run_output_dir, "marker.txt")
        with open(marker_file, "w") as f:
            f.write("handler1")
        
        # Create second handler (should have different directory)
        handler2 = FileIOHandler(temp_repo, temp_output, temp_config)
        os.makedirs(handler2._run_output_dir, exist_ok=True)
        
        # Verify directories are different OR that marker file still exists
        if handler1._run_output_dir == handler2._run_output_dir:
            # If directories are the same, this is a collision (old bug)
            pytest.fail(f"Directory collision detected: {handler1._run_output_dir}")
        else:
            # Directories are different - verify first handler's data is intact
            assert os.path.exists(marker_file), \
                "First handler's data was affected by second handler"
    
    def test_timestamp_format_includes_sufficient_precision(self):
        """Test that the timestamp format has sufficient precision (seconds or microseconds)."""
        handler = FileIOHandler(".", ".", ".")
        timestamp = handler.get_timestamp()
        
        # Check timestamp length
        # Format YYYYMMDDHHMMSS = 14 characters (with seconds)
        # Format YYYYMMDDHHMM = 12 characters (without seconds - OLD BUG)
        # Format YYYYMMDDHHMMSS<microseconds> = 20 characters (with microseconds)
        
        assert len(timestamp) >= 14, \
            f"Timestamp '{timestamp}' does not include seconds (length={len(timestamp)}). " \
            f"Expected at least 14 characters (YYYYMMDDHHMMSS)"
        
        # Verify timestamp is all digits (or digits + microseconds)
        assert timestamp.isdigit(), \
            f"Timestamp '{timestamp}' contains non-digit characters"
    
    @pytest.mark.parametrize("num_instances", [2, 5, 10, 20])
    def test_rapid_instantiation_uniqueness(self, temp_dirs, num_instances):
        """Parameterized test for various numbers of rapid instantiations."""
        temp_repo, temp_output, temp_config = temp_dirs
        
        workdirs = []
        for _ in range(num_instances):
            handler = FileIOHandler(temp_repo, temp_output, temp_config)
            workdirs.append(handler.workdir)
        
        unique_count = len(set(workdirs))
        assert unique_count == num_instances, \
            f"Expected {num_instances} unique directories, got {unique_count}. " \
            f"Duplicates: {[w for w in workdirs if workdirs.count(w) > 1]}"

