"""Exercise an installed artifact, never the checkout. Run with its Python."""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request


def main():
    # Isolated mode also excludes the checkout/script directory from sys.path.
    assert sys.flags.isolated, "run with python -I scripts/smoke_installed.py"
    import indexedcode
    from importlib.metadata import version
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    assert Path(indexedcode.__file__).is_relative_to(Path(sys.prefix))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cli = str(Path(sys.executable).parent / "indexedcode")
    mcp_cli = str(Path(sys.executable).parent / "indexedcode-mcp")
    with tempfile.TemporaryDirectory() as directory:
        cwd = Path(directory)
        root = cwd / "repo"
        root.mkdir()
        (root / "example.py").write_text(
            "def greet(name):\n    return 'hello ' + name\n\n"
            "def welcome():\n    return greet('world')\n"
        )
        common = ["--root", str(root), "--db", str(cwd / "index.sqlite")]

        def run(*args):
            return subprocess.check_output(
                args, cwd=cwd, env=env, text=True, timeout=60,
            )

        assert run(cli, "--version").strip() == f"indexedcode {version('indexedcode')}"
        for command in ([cli], [mcp_cli], [sys.executable, "-m", "indexedcode"]):
            assert "usage:" in run(*command, "--help")
        assert json.loads(run(cli, "index", "once", *common))["files_ready"] == 1
        result = json.loads(run(cli, "search", "greet", *common))
        assert result["results"][0]["qualified_name"] == "greet"
        assert "def greet" in json.loads(run(cli, "context", "example.py", *common))["source"]
        assert json.loads(run(cli, "references", "greet", *common))["references"]
        assert json.loads(run(cli, "status", *common))["coverage"]["ready_files"] == 1

        # Port zero avoids races finding a free port; stderr announces readiness.
        with tempfile.TemporaryFile(mode="w+") as errors:
            proc = subprocess.Popen(
                [cli, "orchestrator", "serve", *common, "--http-port", "0"],
                cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                text=True,
            )
            try:
                # Log lines precede the "listening" line; bound the wait so a
                # hung server fails the check instead of blocking forever.
                import selectors, time
                line, deadline = "", time.monotonic() + 60
                with selectors.DefaultSelector() as selector:
                    selector.register(proc.stderr, selectors.EVENT_READ)
                    while not line.startswith("listening http://"):
                        remaining = deadline - time.monotonic()
                        assert remaining > 0 and selector.select(remaining), (
                            f"HTTP startup timed out; last line: {line!r}")
                        line = proc.stderr.readline().strip()
                base = line.split()[1]
                with urllib.request.urlopen(base + "/v1/status", timeout=10) as response:
                    assert json.load(response)["coverage"]["ready_files"] == 1
                with urllib.request.urlopen(base + "/v1/queue", timeout=10) as response:
                    assert json.load(response)["total"] == 0
                with urllib.request.urlopen(base + "/", timeout=10) as response:
                    assert b"IndexedCode" in response.read()
                with urllib.request.urlopen(base + "/llms.txt", timeout=10) as response:
                    assert f"{base}/v1/search".encode() in response.read()
            finally:
                proc.terminate()
                try:
                    _, stderr = proc.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                    raise AssertionError("HTTP shutdown timed out")
                assert proc.returncode == 0, stderr

        async def check_mcp():
            params = StdioServerParameters(command=mcp_cli, args=common, env=env, cwd=str(cwd))
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    result = await client.call_tool("code_search", {"query": "greet"})
                    assert not result.isError
                    assert json.loads(result.content[0].text)["results"]

        asyncio.run(asyncio.wait_for(check_mcp(), timeout=60))
    print("Installed CLI, HTTP, and MCP smoke checks passed")


if __name__ == "__main__":
    main()
