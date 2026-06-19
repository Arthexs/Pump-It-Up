"""
Registered preprocessing transformers.

Each transformer follows the sklearn protocol: fit / transform / fit_transform.
Registered via @TRANSFORMERS.register so Pipeline.from_config() can build them
from a config dict.
"""
