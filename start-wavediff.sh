#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.

# Enable strict mode for robust error handling
set -euo pipefail

# Log the script start
echo "Starting entrypoint script..."

# Temporarily disable strict mode for Conda activation
set +euo pipefail
echo "Activating Conda environment..."
conda activate wavedenv || { echo "Failed to activate Conda environment 'wavediff-env'"; exit 1; }

# Re-enable strict mode after activation
set -euo pipefail
echo "Conda environment 'wavediff-env' activated."

# Execute the main application command
echo "Running the application..."
exec wavediff -c "${WORK}/config_runs/configs.yaml" -r "${WORK}/wf-psf" -o "${WORK}/config_runs"
