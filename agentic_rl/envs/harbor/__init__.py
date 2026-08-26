"""The shared substrate family: HarborEnv runtime + the canonical converter.

Multi-step episodes with in-place grading (``test.sh`` → ``reward.json``).
Other families layer on top: frontier_cs subclasses the env; swe_rebench and
(planned) terminal_bench / swebenchpro reuse the converter.
"""
