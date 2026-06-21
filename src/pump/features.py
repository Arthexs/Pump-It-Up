"""
Feature engineering and feature selection.

Engineering: derive new columns (pump_age, geo_cluster, frequency encodings).
Selection: reduce to most informative subset (variance, correlation, XGB importance,
mutual information).

Both groups are registered so Pipeline.from_config() can compose them.
"""
