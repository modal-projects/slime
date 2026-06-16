"""Dataset converters: one per env, paired by filename (see env/base.py).

``env/convert2slime/<name>.py`` is the only writer of the ``metadata`` schema
``env/<name>.py`` reads. Run offline; may carry heavy deps the rollout never imports.
"""
