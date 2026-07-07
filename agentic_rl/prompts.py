"""Agent prompts + the bash tool schema.

Qwen3.x is tool-call-trained (reasoning_parser=qwen3, tool_call_parser=qwen). Asked to emit a
```bash text fence (mini-swe's default), it often emits its native <tool_call>…</tool_call> instead,
which fails the fence parser and kills episodes on RepeatedFormatError. So (following Tmax, Qwen3.6-27B
on open-instruct) we switch the action channel to native tool-calling: render the `bash` tool schema
into the prompt (apply_chat_template tools=…) so the model emits <tool_call>, which model.py parses.
The mini-swe THOUGHT / workflow / submit protocol is otherwise preserved. GLM (clear_thinking path)
keeps the ```bash fence via ACTION_REGEX below.
"""

# The single tool we expose. Standard OpenAI function schema; Qwen's chat template renders it into the
# system section and the model replies with a native <tool_call> block (parsed in model.py).
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command. Each command runs in a new subshell. "
        "Commands chained with && or || count as a single command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute."}},
            "required": ["command"],
        },
    },
}

SYSTEM_TEMPLATE = """\
You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.

Include a THOUGHT section before your action where you explain your reasoning. After the THOUGHT, call the
`bash` tool with EXACTLY ONE command (commands chained with && or || count as one command).

Failure to follow these rules — calling no tool, calling a tool other than `bash`, or omitting the THOUGHT —
will cause your response to be rejected.
"""

INSTANCE_TEMPLATE = """\
<pr_description>
Consider the following PR description:
{{task}}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by calling the `bash` tool.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.
<IMPORTANT>This is an interactive process where you will think and call the `bash` tool ONCE, see its result, then think and issue your next call.</IMPORTANT>

## Important Boundaries

- MODIFY: Regular source code files in {{cwd}} (this is the repository, and the working directory for all your subsequent commands)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Command Execution Rules

You are operating in an environment where
1. You call the `bash` tool with a single command
2. The system executes that command in a subshell
3. You see the result
4. You call the `bash` tool again with your next command

Each response must include:
1. A **THOUGHT** section where you explain your reasoning and plan
2. EXACTLY ONE call to the `bash` tool with a single command

**CRITICAL REQUIREMENTS:**
- Your response MUST include a THOUGHT section explaining your reasoning
- Your response MUST make EXACTLY ONE call to the `bash` tool
- The call MUST contain EXACTLY ONE command (or a set of commands connected with && or ||)
- Do NOT call any tool other than `bash`, and do NOT make zero or multiple calls — your response WILL FAIL
- Do NOT try to run multiple independent commands in one response; chain with && / || or wait for the result and issue the next call

If you need to run multiple commands, either combine them with && or ||, or wait for the
first command's output and issue the next command in your following response.

## Environment Details

- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- Directory or environment variable changes are not persistent. Every command is executed in a new subshell.
  However, you can prefix any command with `MY_ENV_VAR=MY_VALUE cd /path/to/dir && ...` to chain context within one command.

## Submission

When you've completed your work, you MUST submit your changes as a git patch.
Follow these steps IN ORDER, with SEPARATE `bash` calls:

Step 1: Create the patch file
Call `bash` with `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only the source files you modified.
Do NOT commit your changes.

<IMPORTANT>
The patch must only contain changes to the specific source files you modified to fix the issue.
Do not submit file creations or changes to any of the following files:
- test and reproduction files
- helper scripts, tests, or tools that you created
- installation, build, packaging, configuration, or setup scripts unless directly part of the fix
- binary or compiled files
</IMPORTANT>

Step 2: Verify your patch
Inspect patch.txt to confirm it only contains your intended changes and headers show `--- a/` and `+++ b/` paths.

Step 3: Submit (EXACT command required)
Call the `bash` tool with this EXACT command to submit:

`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`

If the command fails (nonzero exit status), it will not submit.

<CRITICAL>
- Creating/viewing the patch and submitting it MUST be separate `bash` calls (not combined with &&).
- You CANNOT continue working (reading, editing, testing) in any way on this task after submitting.
</CRITICAL>
</instructions>
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

# Tool-call protocol (Qwen): model.py parses the native qwen3_xml
# <tool_call><function=bash><parameter=command>…</parameter></function></tool_call> from the generated
# text. The submit sentinel is echoed via the bash tool; Sandbox.execute detects it and raises Submitted.
SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# Legacy ```bash text fence + its jinja format-error template — retained for the GLM (clear_thinking)
# path, which emits the fence reliably. parse_regex_actions renders this with jinja. The Qwen tool-call
# path builds its own format-error message inline in model.py.
ACTION_REGEX = r"```bash\s*\n(.*?)\n```"
FORMAT_ERROR_TEMPLATE = """\
{% if finish_reason is defined and finish_reason in ["length", "tool_calls"] -%}
Your previous response reached the output token limit (finish_reason={{ finish_reason }}) before you produced a complete action, so it was cut off. Respond more concisely and provide exactly one action in the required format.
{%- else -%}
Format error: {{ error }}

Please always provide EXACTLY ONE action in triple backticks (```bash ... ```), found {{ actions|length }} actions.
{%- endif %}
"""
