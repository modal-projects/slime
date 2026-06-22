"""Dataset converters: one per env, paired by filename (see environment/base.py).

``environment/convert2slime/<name>.py`` is the only writer of the ``metadata`` schema
``environment/<name>.py`` reads. Run offline; may carry heavy deps the rollout never imports.
"""
