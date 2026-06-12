"""Dataset converters: one per env, paired by filename (see env/base.py).

``env/convert2slime/<name>.py`` is the only writer of the ``metadata`` schema
that ``env/<name>.py`` reads. Converters run offline (laptop or head node)
and may carry heavy/optional dependencies (``datasets``, ``harbor``) that the
rollout runtime never imports.
"""
