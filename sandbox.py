"""Sandboxed code execution via Daytona."""

import asyncio

from config import DAYTONA_API_KEY

_daytona_client = None
_daytona_sandbox = None
_daytona_lock = asyncio.Lock()

_LANGUAGE_ALIASES = {
    "python": "python",
    "python3": "python",
    "py": "python",
    "javascript": "javascript",
    "js": "javascript",
    "node": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "bash": "bash",
    "sh": "bash",
    "shell": "bash",
}


async def _get_daytona_sandbox():
    """Return a live Daytona sandbox, creating one if needed."""
    global _daytona_client, _daytona_sandbox

    if _daytona_client is None:
        from daytona import AsyncDaytona, DaytonaConfig

        _daytona_client = AsyncDaytona(DaytonaConfig())

    if _daytona_sandbox is None:
        _daytona_sandbox = await _daytona_client.create()

    return _daytona_sandbox


async def _reset_daytona_sandbox():
    """Discard the current sandbox so the next call builds a fresh one."""
    global _daytona_sandbox
    sandbox, _daytona_sandbox = _daytona_sandbox, None
    if sandbox is not None:
        try:
            await _daytona_client.delete(sandbox)
        except Exception:
            pass


async def run_code_in_sandbox(code: str, language: str = "python") -> str:
    """Execute code in an isolated Daytona sandbox and return its output."""
    if not DAYTONA_API_KEY:
        return "Error: DAYTONA_API_KEY is not configured; code execution is unavailable."

    lang = _LANGUAGE_ALIASES.get((language or "python").lower())
    if lang is None:
        return (
            f"Error: unsupported language '{language}'. "
            "Supported: python, javascript, typescript, bash."
        )

    async with _daytona_lock:
        for attempt in (1, 2):
            try:
                sandbox = await _get_daytona_sandbox()
            except Exception as exc:
                if attempt == 2:
                    return f"Error: could not start sandbox: {exc}"
                continue

            try:
                if lang == "python":
                    response = await sandbox.process.code_run(code, timeout=60)
                else:
                    # For non-Python runtimes, pipe the source through a shell
                    # heredoc into the interpreter.
                    interpreter = {
                        "javascript": "node",
                        "typescript": "ts-node",
                        "bash": "bash",
                    }[lang]
                    heredoc = f"{interpreter} <<'__CODE_EOF__'\n{code}\n__CODE_EOF__"
                    response = await sandbox.process.exec(heredoc, timeout=60)

                output = (response.result or "").rstrip()
                exit_code = getattr(response, "exit_code", 0)
                if exit_code:
                    output += f"\n[exit code: {exit_code}]"
                return output or "(no output)"
            except Exception as exc:
                # The sandbox may have died between calls; rebuild it and retry.
                await _reset_daytona_sandbox()
                if attempt == 2:
                    return f"Code execution error: {exc}"
        return "Code execution error: unknown failure"


# Backwards-compatible alias for the old Piston-based name.
run_code_in_piston = run_code_in_sandbox
