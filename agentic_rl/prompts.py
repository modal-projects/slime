"""Agent prompts and the tool-call protocol, pinned here so the rollout's prompt
distribution is version-controlled rather than loaded from the installed package.

One universal scaffold for every task family: response format, subshell
semantics, and how to finish. The task's *deliverable* (patch vs artifacts vs
stdout) lives in the per-row instruction text rendered as ``{{task}}``.

The harness drives bash via native tool-calls, so the served model must expose
``--sglang-tool-call-parser`` and ``--sglang-reasoning-parser``. ``BASH_TOOL`` is
mini-swe's single bash tool schema.
"""

from minisweagent.models.utils.actions_toolcall import BASH_TOOL  # noqa: F401

SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

SYSTEM_TEMPLATE = "You are a helpful assistant that can interact with a computer.\n"

INSTANCE_TEMPLATE = """\
Please solve the following task:

<task>
{{task}}
</task>

You can execute bash commands and edit files to accomplish it. The task
description above defines what you must produce and any task-specific submission
steps; follow it exactly.

## Command Execution Rules

You are operating in an environment where

1. You issue at least one command
2. The system executes the command(s) in a subshell
3. You see the result(s)
4. You write your next command(s)

Each response should include:

1. Reasoning text where you explain your analysis and plan
2. At least one call to the `bash` tool with the shell command to run

**CRITICAL REQUIREMENTS:**

- Your response SHOULD include reasoning text explaining what you're doing
- Your response MUST include AT LEAST ONE call to the `bash` tool
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
- However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/working/dir && ...` or write/load environment variables from files

## Environment Details

- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- You can create new tools or scripts to help you; if a tool isn't available, you can install it

<system_information>
{{system}} {{release}} {{version}} {{machine}}
</system_information>

## Useful shell idioms

The following are commands you pass to the `bash` tool (its `command` argument);
they are NOT a response format. Adapt as needed.

- Create a file with a heredoc:
  cat <<'EOF' > newfile.py
  import numpy as np
  print("hello")
  EOF
- Edit in place with sed: `sed -i 's/old/new/g' filename.py`.
- View specific lines with numbers: `nl -ba filename.py | sed -n '10,20p'`.

## Finishing

Once you have verified your work and completed any submission steps the task
description requires, finish by issuing the following command:
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`.
Do not combine it with any other command. <important>After this command, you
cannot continue working on this task.</important>
"""

OBSERVATION_TEMPLATE = """\
{% if output.exception_info -%}
<exception>{{output.exception_info}}</exception>
{% endif -%}
<returncode>{{output.returncode}}</returncode>
{% if output.output | length < 10000 -%}
<output>
{{ output.output -}}
</output>
{%- else -%}
<warning>
The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you're using grep or find and it produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.
</warning>
{%- set elided_chars = output.output | length - 10000 -%}
<output_head>
{{ output.output[:5000] }}
</output_head>
<elided_chars>
{{ elided_chars }} characters elided
</elided_chars>
<output_tail>
{{ output.output[-5000:] }}
</output_tail>
{%- endif -%}
"""

FORMAT_ERROR_TEMPLATE = """\
<tool_response>
Tool call error:

<error>
{{error}}
</error>

Here is general guidance on how to submit correct toolcalls:

Every response needs to use the 'bash' tool at least once to execute commands.

Call the bash tool with your command as the argument:
- Tool: bash
- Arguments: {"command": "your_command_here"}

If you want to end the task, please issue the following command: `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
without any other command.
</tool_response>
"""
