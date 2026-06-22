"""Verifier server: a grading server the agent queries DURING rollout.

Frontier-CS (competitive programming) grades a C++ submission with a Node +
go-judge server (vendored under ``server/``). One server per worker mounts the
problems Volume and serves every problem (the caller passes ``PROBLEM_ID``);
both the agent's ``submit.sh`` loop and the verifier's ``evaluate.py`` POST to it.

``autostart.ensure_started()`` boots it once per worker as a ``vm_runtime`` Modal
Sandbox, mounts the ``slime-data`` Volume at ``/data``, and points the go-judge's
``problemsRoot`` at ``/data/frontier_cs/problems`` — the testdata that the config's
``download_data`` pulls there (``frontier_cs/problems/**``). No separate problems
Volume or populate step. Replaces the old ``judges/frontier_cs/{autostart,judge_server}``.
"""
