# Service-manager units for the core CLI

These examples run `indexedcode orchestrator serve`: the indexer, watcher, and loopback retrieval API. They do **not** run MLX/ROCm workers or enable inference. The CLI does not load TOML configuration. Semantic search remains unavailable; default hybrid search falls back to symbol/lexical retrieval.

Install the core CLI using the [README](../../README.md), then run `command -v indexedcode` as the service owner. Replace every example executable and data path below with the actual **absolute** path. Service managers do not reliably inherit your interactive shell's `PATH`, and they do not expand `~` or shell variables inside argument lists. Keep the database outside the repository and create its parent directory before starting the service.

Use one writer per database. Stop an existing foreground watcher or service before starting another writer. The examples leave HTTP on loopback without authentication; do not forward the endpoint to a public interface. No GPU dependencies are needed.

## launchd (macOS)

Example LaunchAgent destination: `~/Library/LaunchAgents/dev.indexedcode.orchestrator.plist`. First create `/Users/you/.local/share/indexedcode/myrepo` as the account that will run the service. Replace `/Users/you/.local/bin/indexedcode` with the executable path reported by `command -v indexedcode`.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>dev.indexedcode.orchestrator</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/you/.local/bin/indexedcode</string>
    <string>orchestrator</string>
    <string>serve</string>
    <string>--root</string><string>/Users/you/work/myrepo</string>
    <string>--db</string><string>/Users/you/.local/share/indexedcode/myrepo/index.sqlite</string>
    <string>--http-port</string><string>8765</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>/Users/you/.local/share/indexedcode/myrepo/stdout.log</string>
  <key>StandardErrorPath</key><string>/Users/you/.local/share/indexedcode/myrepo/stderr.log</string>
</dict>
</plist>
```

Load and stop the LaunchAgent in your logged-in GUI session:

```bash
launchctl bootstrap "gui/$(id -u)" "$HOME/Library/LaunchAgents/dev.indexedcode.orchestrator.plist"
launchctl print "gui/$(id -u)/dev.indexedcode.orchestrator"
# Stop and unload before upgrading, uninstalling, or running index once:
launchctl bootout "gui/$(id -u)" "$HOME/Library/LaunchAgents/dev.indexedcode.orchestrator.plist"
```

The log directory must exist and be writable before bootstrap. Configure log rotation if keeping this service enabled long-term.

## systemd user service (Linux)

Example destination: `~/.config/systemd/user/indexedcode.service`. Create that directory and `/home/you/.local/share/indexedcode/myrepo` first. Install the CLI as the same user; replace the executable path with the actual installed path.

```ini
[Unit]
Description=IndexedCode repository indexer and retrieval API

[Service]
Type=simple
ExecStart=/home/you/.local/bin/indexedcode orchestrator serve --root /home/you/work/myrepo --db /home/you/.local/share/indexedcode/myrepo/index.sqlite --http-port 8765
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now indexedcode.service
systemctl --user status indexedcode.service
journalctl --user -u indexedcode.service
# Stop before upgrading or running index once:
systemctl --user stop indexedcode.service
# Disable and stop before uninstalling:
systemctl --user disable --now indexedcode.service
```

A user service normally follows the user session lifecycle; arrange lingering separately with your system administrator if it must run after logout. Paths containing spaces need appropriate systemd quoting.

## Verify and maintain

Both examples serve `http://127.0.0.1:8765`. Check readiness after startup:

```bash
curl http://127.0.0.1:8765/v1/status
```

The process prints its listening URL after the initial scan. `indexedcode status --root /absolute/repo --db /absolute/external/index.sqlite` queries database state; it is not a service-liveness or port check. Use launchd/systemd status and HTTP to verify the process.

SIGTERM requests shutdown of the core service. No inference-worker draining or termination is involved. Stop the service and confirm it has exited before switching package versions or database paths. Recheck the executable path after reinstalling, and restart only after backup/upgrade checks from the [README](../../README.md).

Do not add experimental worker environment variables or model settings to these units expecting them to enable inference. See the [MLX](local-mlx.md) and [ROCm](remote-rocm.md) boundary notes for what remains outside the supported release.
