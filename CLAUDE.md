# slime

For debugging and testing, use Modal. Helper scripts for running slime on Modal exist in ~/multinode-training-guide/slime. You may need to write tests, refer to `test_lora_verification.py` in that direcotry for an example.

If you just need to browse dependencies or test code changes, you can run commands in a temp shell via something like `uv run modal shell some_modal_script.py:some_modal_func -c CMD`. You can also do so against a running container via `modal container exec ta-... -- /bin/bash -c 'echo hi'`, or `uv run modal shell ta-... -c "/bin/bash -c 'echo hi'"`.

When in doubt, use `uv` to interact with modal.
