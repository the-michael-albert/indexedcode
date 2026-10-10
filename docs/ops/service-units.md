# Service-manager units

Examples for running `orchestrator serve` and remote workers as managed
services. Adapt paths, users, and secret delivery to your environment;
never put tokens in unit files that world-readable service directories
could expose — use `LoadCredential`/`EnvironmentFile` with strict modes.

## launchd (macOS, local MLX orchestrator)

`~/Library/LaunchAgents/dev.indexedcode.orchestrator.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>dev.indexedcode.orchestrator</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/env</string>
    <string>indexedcode</string>
    <string>orchestrator</string>
    <string>serve</string>
    <string>--root</string><string>/Users/you/work/myrepo</string>
    <string>--db</string><string>/Users/you/work/myrepo/index.sqlite</string>
    <string>--http-port</string><string>8765</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Background</string>
</dict>
</plist>
```

`launchctl load ~/Library/LaunchAgents/dev.indexedcode.orchestrator.plist`

The orchestrator handles SIGTERM gracefully (stops admission, drains,
asks managed workers to stop, terminates after the grace period), which
is what `launchctl unload` and systemd stop send.

## systemd (Linux, remote ROCm worker host)

`/etc/systemd/system/indexedcode-embed.service`:

```ini
[Unit]
Description=IndexedCode ROCm embedding worker
After=network.target

[Service]
User=indexedcode
WorkingDirectory=/opt/indexedcode
ExecStart=/opt/indexedcode/.venv/bin/python -m indexedcode.workers.serve \
  --role embed --backend rocm \
  --model sentence-transformers/all-MiniLM-L6-v2 --port 8901
Restart=on-failure
RestartSec=5
Environment=INDEXEDCODE_WORKER_SECRET_FILE=/run/secrets/worker-token

[Install]
WantedBy=multi-user.target
```

`systemctl enable --now indexedcode-embed.service`

Worker processes open no repository database and start no watcher; each
host supervises its own.
