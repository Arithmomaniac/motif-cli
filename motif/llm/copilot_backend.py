"""
Copilot SDK backend for automated analysis.

Uses the GitHub Copilot Python SDK (github-copilot-sdk) to send analysis
prompts to an LLM and collect structured JSON responses, replacing the
manual "paste into your IDE agent" step.
"""

import asyncio
import json
import re
from typing import Optional

from rich.console import Console


DEFAULT_MODEL = "claude-sonnet-4.6"


def _check_sdk_available():
    """Raise a helpful error if the Copilot SDK is not installed."""
    try:
        import copilot  # noqa: F401
        return True
    except ImportError:
        raise ImportError(
            "The Copilot SDK is required for --auto mode.\n"
            "Install it with: pip install motif-cli[copilot]\n"
            "Or: pip install github-copilot-sdk"
        )


def _parse_json_response(text: str) -> dict:
    """Extract JSON from an LLM response that may include markdown fences.

    Handles:
    - Raw JSON: {"key": "value"}
    - Fenced JSON: ```json\n{...}\n```
    - JSON with surrounding text
    """
    # Try raw JSON first
    text = text.strip()
    if text.startswith("{"):
        # Find the matching closing brace
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Maybe there's trailing text after the JSON
            pass

    # Try extracting from code fences
    fence_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    matches = re.findall(fence_pattern, text)
    for match in matches:
        match = match.strip()
        if match.startswith("{"):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

    # Last resort: find first { to last } and try parsing
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract valid JSON from LLM response.\n"
        f"Response starts with: {text[:200]}..."
    )


async def _send_and_collect(session, prompt: str) -> str:
    """Send a prompt to a Copilot session and collect the full response."""
    done = asyncio.Event()
    response_parts = []

    def on_event(event):
        event_type = event.type.value if hasattr(event.type, 'value') else str(event.type)
        if event_type == "assistant.message":
            response_parts.append(event.data.content)
        elif event_type == "session.idle":
            done.set()

    session.on(on_event)
    await session.send(prompt)
    await done.wait()

    if not response_parts:
        raise RuntimeError("No response received from Copilot session")

    return response_parts[-1]  # Final complete message


async def _run_full_analysis(
    client,
    prepared_data: str,
    model: str,
    console: Console,
) -> dict:
    """Run single-shot analysis for 'full' mode (Personalize AI)."""
    from copilot.session import PermissionHandler

    console.print(f"  Creating session with [cyan]{model}[/cyan]...")

    async with await client.create_session(
        on_permission_request=PermissionHandler.deny_all,
        model=model,
        infinite_sessions={"enabled": False},
    ) as session:
        console.print("  Sending analysis data to LLM...")
        with console.status("[bold green]Analyzing conversations..."):
            response = await _send_and_collect(session, prepared_data)

    console.print("  Parsing JSON response...")
    return _parse_json_response(response)


async def _run_vibe_report_analysis(
    client,
    output_parts: list,
    model: str,
    console: Console,
) -> dict:
    """Run vibe-report analysis in a single consolidated request.

    The original multi-batch workflow was designed for manual IDE delegation
    where agents had limited context. With the Copilot SDK and models like
    claude-sonnet-4.6 (1M context), we consolidate everything into one
    request to minimize premium request consumption.
    """
    from copilot.session import PermissionHandler

    # Reassemble instructions + all batch data into one prompt
    instructions_content = None
    all_batch_data = []

    for suffix, content in output_parts:
        if suffix == "instructions":
            instructions_content = content
        elif suffix.startswith("batch-"):
            all_batch_data.append(content)
        # analysis-brief and session-index are for the multi-batch workflow, skip

    if not instructions_content:
        raise ValueError("Missing instructions file in prepared output")
    if not all_batch_data:
        raise ValueError("No batch files found in prepared output")

    # Single consolidated prompt
    combined_data = "\n\n---\n\n".join(all_batch_data)
    prompt = (
        f"{instructions_content}\n\n---\n\n"
        f"## Conversation Data\n\n{combined_data}"
    )

    batch_count = len(all_batch_data)
    console.print(
        f"  Consolidated {batch_count} batches into single request "
        f"(~{len(prompt) // 4:,} tokens)"
    )

    async with await client.create_session(
        on_permission_request=PermissionHandler.deny_all,
        model=model,
        infinite_sessions={"enabled": False},
    ) as session:
        console.print(f"  Creating session with [cyan]{model}[/cyan]...")
        with console.status("[bold green]Analyzing conversations (this may take a minute)..."):
            response = await _send_and_collect(session, prompt)

    console.print("  Parsing JSON response...")
    return _parse_json_response(response)


def send_to_copilot(
    prepared_output,
    mode: str,
    model: str = DEFAULT_MODEL,
    console: Optional[Console] = None,
) -> dict:
    """Send pre-prepared analysis data to Copilot SDK and return parsed JSON.

    Args:
        prepared_output: Either a string (full mode) or list of (suffix, content)
                        tuples (vibe-report mode), as returned by prepare_analysis().
        mode: "full" or "vibe-report".
        model: Copilot model name.
        console: Rich console for progress display.

    Returns:
        Parsed analysis JSON dict.
    """
    _check_sdk_available()

    if console is None:
        console = Console()

    console.print(f"\n[bold]Sending to Copilot ({model})...[/bold]")

    async def _run():
        from copilot import CopilotClient

        async with CopilotClient() as client:
            if mode == "vibe-report" and isinstance(prepared_output, list):
                return await _run_vibe_report_analysis(
                    client, prepared_output, model, console,
                )
            else:
                return await _run_full_analysis(
                    client, prepared_output, model, console,
                )

    return asyncio.run(_run())
