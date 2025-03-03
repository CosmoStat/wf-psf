FROM continuumio/miniconda3:latest

WORKDIR /workdir

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml \
    && conda clean -a \
    && rm -rf /opt/conda/pkgs/*

# Make sure conda is activated by default
RUN echo "conda activate wavediff-env" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Demonstrate the environment is activated and TensorFlow is installed
RUN conda activate wavediff-env

# Activate the environment and set it to be the default for the container
SHELL ["conda", "run", "-n", "wavediff-env", "/bin/bash", "-c"]
RUN conda init bash

# Set the default environment for the container to the created environment
ENV PATH /opt/conda/envs/wavediff-env/bin:$PATH

# Optionally, include a step to make sure TensorFlow isn't installed if you don't want it on CPU
# RUN pip uninstall -y tensorflow

# Specify an entrypoint that does not execute until the GPU setup is complete.
# The container won't run anything until the user explicitly installs the GPU dependencies.
CMD ["echo", "TensorFlow GPU not installed. Please install TensorFlow GPU and related dependencies before running the application."]


