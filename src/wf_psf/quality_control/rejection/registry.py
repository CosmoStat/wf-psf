"""Quality control rejection policy registry.

Provides the registry responsible for constructing rejection policies from configuration.

The registry decouples configuration parsing from rejection policy
instantiation, allowing new rejection policies to be registered without
modifying the quality control pipeline.

:Authors:
    Jennifer Pollack <jennifer.pollack@cea.fr>
"""