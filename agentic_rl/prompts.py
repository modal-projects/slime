"""Agent prompts — adapted from mini-swe-agent's `config/benchmarks/swebench_backticks.yaml`
(the SWE-bench config for text-completion models, not the generic default.yaml). Key points it
carries that the generic config lacks: the working directory ({{cwd}}), the MODIFY-source /
DO-NOT-MODIFY-tests boundary, the full single-action format spec, and the *curated patch*
submission protocol (the agent assembles a source-only `git diff` and submits it via the
sentinel, so we grade a clean patch instead of a `git add -A` dump). cwd is templated to our
per-repo workdir (SWE-rebench checks the repo out at /<reponame>, not /testbed).

Action fence is the standard ```bash (like `swebench.yaml`), NOT swebench_backticks's exotic
```mswea_bash_command. GLM-4.7-Flash emits ```bash reliably but whiffs the unusual fence,
which caused ~27% of episodes to die at turn 0 with RepeatedFormatError (zero-signal rollouts).
"""

SYSTEM_TEMPLATE = """\
You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
THOUGHT: Your reasoning and analysis here

```bash
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected.
"""

INSTANCE_TEMPLATE = """\
<pr_description>
Consider the following PR description:
{{task}}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.
<IMPORTANT>This is an interactive process where you will think and issue ONE command, see its result, then think and issue your next command.</IMPORTANT>

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
1. You write a single command
2. The system executes that command in a subshell
3. You see the result
4. You write your next command

Each response should include:
1. A **THOUGHT** section where you explain your reasoning and plan
2. A single bash code block with your command

Format your responses like demonstrated within the <format_example> block:

<format_example>
THOUGHT: Here I explain my reasoning process, analysis of the current situation,
and what I'm trying to accomplish with the command below.

```bash
your_command_here
```
</format_example>

**CRITICAL REQUIREMENTS:**
- Your response SHOULD include a THOUGHT section explaining your reasoning
- Your response MUST include EXACTLY ONE bash code block
- This bash block MUST contain EXACTLY ONE command (or a set of commands connected with && or ||)
- If you include zero or multiple bash blocks, or no command at all, YOUR RESPONSE WILL FAIL
- Do NOT try to run multiple independent commands in separate blocks in one response

Example of a CORRECT response:
<example_response>
THOUGHT: I need to understand the structure of the repository first. Let me check what files are in the current directory.

```bash
ls -la
```
</example_response>

If you need to run multiple commands, either combine them in one block with && or ||, or wait for the
first command's output and issue the next command in your following response.

## Environment Details

- You have a full Linux shell environment
- Always use non-interactive flags (-y, -f) for commands
- Avoid interactive tools like vi, nano, or any that require user input
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell.
  However, you can prefix any action with `MY_ENV_VAR=MY_VALUE cd /path/to/dir && ...` to chain context within one action.

## Submission

When you've completed your work, you MUST submit your changes as a git patch.
Follow these steps IN ORDER, with SEPARATE commands:

Step 1: Create the patch file
Run `git diff -- path/to/file1 path/to/file2 > patch.txt` listing only the source files you modified.
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
You MUST use this EXACT command to submit:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
```

If the command fails (nonzero exit status), it will not submit.

<CRITICAL>
- Creating/viewing the patch and submitting it MUST be separate commands (not combined with &&).
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

# Text protocol: the model emits one ```bash block; the env detects the submit
# sentinel and carries the curated patch (everything after the sentinel line) as the submission.
# (Fence is ```bash, not the exotic ```mswea_bash_command — see module docstring.)
ACTION_REGEX = r"```bash\s*\n(.*?)\n```"
FORMAT_ERROR_TEMPLATE = """\
{% if finish_reason is defined and finish_reason in ["length", "tool_calls"] -%}
Your previous response reached the output token limit (finish_reason={{ finish_reason }}) before you produced a complete action, so it was cut off. Respond more concisely and provide exactly one action in the required format. If you need to think more, do so briefly.
{%- else -%}
Format error: {{ error }}

Please always provide EXACTLY ONE action in triple backticks, found {{ actions|length }} actions.

Please format your action in triple backticks as shown in <response_example>.

<response_example>
THOUGHT: Here are some thoughts about why you want to perform the action.

```bash
your_command_here
```
</response_example>
{%- endif %}
"""
SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
