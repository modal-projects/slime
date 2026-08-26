"""Task families, one directory each (RUNBOOK §7).

A family dir owns everything family-specific: its converter (the only writer
of the ``metadata`` schema its env reads), its env subclass, its infra (e.g.
the Frontier-CS judge), and its eval protocol. The shared substrate (harbor
runtime, sandbox, model seam, rewards) stays outside the family dirs.
``base.py`` holds the RolloutEnv contract + the ``ENVS`` registry.
"""
