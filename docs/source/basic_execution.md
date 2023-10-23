# Basic Execution

The WaveDiff pipeline is launched by the Python script: `src/wf_psf/run.py` which is referenced by the command `wavediff`.

A list of command-line arguments can be displayed using the `--help` option:

```
> wavediff --help
usage: run.py [-h] --conffile CONFFILE --repodir REPODIR --outputdir OUTPUTDIR

optional arguments:
  -h, --help            show this help message and exit
  --conffile CONFFILE, -c CONFFILE
                        a configuration file containing program settings.
  --repodir REPODIR, -r REPODIR
                        the path of the code repository directory.
  --outputdir OUTPUTDIR, -o OUTPUTDIR
                        the path of the output directory.
```

There are three arguments, which the user should specify when launching the pipeline.

The first argument: `--confile CONFFILE` specifies the path to the configuration file storing the pipeline tasks to be executed at runtime.

The second argument: `--repodir REPODIR` is the path to the `wf-psf` repository.

The third argument: `--outputdir OUTPUTDIR` is used to set the path to the main output directory, which stores the `WaveDiff` results.

To run `WaveDiff`, use the following command:

```
> wavediff -c /path/to/config/file -r /path/to/wf-psf -o /path/to/output/dir
```

WaveDiff begins with the input/output (i.e. retrieving and parsing the configuration file and creating output subdirectories in the main output directory).  The subdirectory name is composed of `wf-outputs-` and the timestamp of the run, i.e. `wf-outputs-202310221632`. Each run will produce its own unique subdirectory. Then within this subdirectory, further subdirectories are generated to store the corresponding output.

Below is an example of the all directories generated during each execution of the WaveDiff pipeline.

(wf-outputs)=
```
wf-outputs-202310211641
├── checkpoint
├── config
├── log-files
├── metrics
├── optim-hist
├── plots
└── psf_model
```
A description of each subdirectory is provided in the table below.

| Sub-directory    | Purpose                                       |
|--------------|---------------------------------------------------|
| checkpoint   | Stores the checkpoint weights produced during the training pipeline task. |
| config       | Stores all of the configuration files (see [Configuration](Configuration)) provided as input during the run. |
| log-files    | Stores the log-file of the run.  |
| metrics       | Stores the metrics results generated during the metrics pipeline task.  |
| optim-hist    |  Stores the training history of the model parameters.  |
| plots         | Stores metrics plots generated during the plotting pipeline task.   | 
| psf_models    |  Stores the final trained psf models generated for each training cycle. |


Next, we describe to some detail the configuration file structures and content.


