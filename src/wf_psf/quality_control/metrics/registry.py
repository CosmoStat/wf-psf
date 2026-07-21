"""Quality metric registry.

Provides the registry responsible for constructing quality metric
implementations from configuration.

The registry decouples configuration parsing from metric
instantiation, allowing new quality metrics to be registered without
modifying the quality control pipeline.

:Authors:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""