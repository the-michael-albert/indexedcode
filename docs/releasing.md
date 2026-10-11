# Releasing IndexedCode

IndexedCode releases are versioned wheel and source-distribution (sdist) assets on [GitHub Releases](https://github.com/the-michael-albert/indexedcode/releases). There is **no PyPI publication**, standalone executable, or package-manager formula in this process. The supported release scope is the core CLI on macOS/Linux with Python 3.11 or newer, not native Windows or integrated inference.

## Release gates

The [CI workflow](../.github/workflows/ci.yml) runs:

- Tests on Linux and macOS with Python 3.11, 3.12, 3.13, and 3.14.
- Wheel and sdist builds with `python -m build`, followed by `python -m twine check`.
- Installation of the wheel into a clean environment, dependency checks, and isolated installed-artifact smoke tests.
- A new wheel built from the sdist, installed into another clean environment and checked with the same smoke tests.
- Isolated uv and pipx tool installations and executable smoke checks.

The [installed smoke script](../scripts/smoke_installed.py) runs with Python's `-I` flag, verifies that imports come from the installed environment rather than the checkout, and exercises version/help commands, `python -m indexedcode`, indexing, search, context, references, status, the loopback HTTP service, and an installed `indexedcode-mcp` tool call. It uses a temporary repository and an external database.

Hardware smoke tests remain opt-in. Passing release CI does not claim MLX/ROCm validation, semantic search, or inference integration. The CLI does not load TOML configuration or connect worker components to indexing/retrieval.

## Prepare and validate locally

1. Start from a clean `main` checkout and use a supported Python interpreter.
2. Set the intended PEP 440 version in [package metadata](../pyproject.toml), the single version source. Use its canonical spelling so the tag and built metadata match exactly.
3. Update pinned release URLs and release-specific notes in the [README](../README.md). Clearly state that the new URLs will not be live until publication.
4. In a fresh checkout, build and validate:

   ```bash
   python -m venv .release-env
   .release-env/bin/python -m pip install -e '.[test]' build twine
   .release-env/bin/python -m pytest
   .release-env/bin/python -m build
   .release-env/bin/python -m twine check dist/*
   ```

   Ensure the output directory contains only the intended version's artifacts. If it contains old outputs, inspect their resolved paths before removing anything, or use a fresh checkout.

5. Validate the wheel in a clean environment:

   ```bash
   .release-env/bin/python -m venv .release-check/wheel
   .release-check/wheel/bin/python -m pip install dist/*.whl
   .release-check/wheel/bin/python -m pip check
   .release-check/wheel/bin/python -I scripts/smoke_installed.py
   ```

6. Validate a wheel rebuilt from the sdist:

   ```bash
   mkdir -p .release-check/from-sdist
   .release-env/bin/python -m pip wheel --no-deps --wheel-dir .release-check/from-sdist dist/*.tar.gz
   .release-env/bin/python -m venv .release-check/sdist
   .release-check/sdist/bin/python -m pip install .release-check/from-sdist/*.whl
   .release-check/sdist/bin/python -m pip check
   .release-check/sdist/bin/python -I scripts/smoke_installed.py
   ```

7. Push the reviewed branch and require CI to pass. Local checks on one interpreter do not replace the platform/version matrix or uv/pipx checks. Merge the reviewed release commit before tagging it.

Use fresh validation environments for each release candidate. Do not use an editable install as evidence that the wheel or sdist is complete.

## Publish

**Automatic:** merging a PR that raises `version` in `pyproject.toml` to `main` triggers [auto-release](../.github/workflows/auto-release.yml). If no `v<version>` release or tag exists, it calls the release workflow below, which creates the tag at the merge commit only after the full CI gate passes. Merges that do not change the version do nothing. If the gate fails, fix forward and re-run the workflow; no tag was created.

**Manual:** publishing can also be done by hand. The [release workflow](../.github/workflows/release.yml) triggers on pushed `v*` tags. After approval, create a tag that exactly matches `v` plus the package metadata version:

```bash
git tag -a v1.0.0a3 -m "IndexedCode 1.0.0a3"
git push origin v1.0.0a3
```

The workflow:

1. Rejects tags that do not exactly match the package version.
2. Runs the reusable CI matrix and all artifact/install gates.
3. Downloads the exact distributions verified by CI; it does not independently rebuild release assets.
4. Generates `SHA256SUMS` for the wheel and sdist.
5. Refuses to overwrite an existing release, including an existing draft.
6. Creates a draft release, uploads the wheel, sdist, and checksums, then publishes it after the upload succeeds. This is an automated draft-to-published transition, not a manual approval pause.

PEP 440 prereleases are marked as prereleases. Only the publication job has `contents: write`; other jobs are read-only. Assets are treated as **immutable**: never replace a published wheel or sdist, reuse a published version, or move its tag. This policy is enforced by refusing an existing release; it is not a claim that repository-level GitHub immutable-release settings are enabled.

No workflow uploads to PyPI; the only tags created automatically are by auto-release, at the merge commit, after CI passes. Versioned download URLs are public only after publication.

## Verify the published release

Check that the expected wheel, sdist, and checksums appear on the release page. Download all three into a fresh directory, then verify the assets from that directory:

```bash
# Linux:
sha256sum -c SHA256SUMS
# macOS:
shasum -a 256 -c SHA256SUMS
```

Test installation in a clean macOS or Linux environment, once per installer:

```bash
uv tool install --python 3.12 https://github.com/the-michael-albert/indexedcode/releases/download/v1.0.0a3/indexedcode-1.0.0a3-py3-none-any.whl
uv tool update-shell
# Or, with Python 3.11+ selected by pipx:
pipx install https://github.com/the-michael-albert/indexedcode/releases/download/v1.0.0a3/indexedcode-1.0.0a3-py3-none-any.whl
pipx ensurepath
```

Open a new shell if needed. Check `indexedcode --version`, `indexedcode --help`, and `indexedcode-mcp --help`. Follow the [README quick start](../README.md) against a disposable source repository with explicit absolute `--root` and external `--db` paths. Verify one-shot indexing and JSON retrieval, then the watcher and loopback status endpoint, and an MCP client tool call. Default hybrid search reporting `semantic_unavailable` is expected, not a failed release check.

## Failure, upgrade, and rollback

- **Verification fails before a release exists:** fix and review the candidate. Do not bypass the gates. Prefer a fresh version/tag; if an unpublished tag must be removed, coordinate that action with maintainers rather than force-moving it silently.
- **A partial draft exists:** inspect its assets and workflow logs before deciding whether to delete the draft and retry. A rerun refuses the existing draft; it does not silently replace assets.
- **A published release is defective:** keep its tag and assets unchanged. Warn users in release notes, fix the issue, increment the version, and publish a new release.
- **User upgrades:** install the new, exact wheel URL with `uv tool install --force --python 3.12 URL` or `pipx install --force URL`. A pinned URL does not automatically track the newest release. Use the owning installer, not both.
- **Local data:** stop the writer before switching versions. With the SQLite command available, `sqlite3 "$DB" ".backup '${DB}.backup'"` creates a consistent backup; use a unique destination for retention. Never copy only a live WAL-mode main database. Sidecars are named with `-wal` and `-shm` suffixes.
- **Rollback:** stop the writer, reinstall the prior wheel, and restore its known-good backup to a separate database path, or rebuild into a fresh database from the repository. Do not assume cross-version schema compatibility. Keep the original data until validation succeeds.

The [README](../README.md) also covers PATH troubleshooting and uninstalling with `uv tool uninstall indexedcode` or `pipx uninstall indexedcode`. Uninstall does not remove indexes or service definitions; stop and disable services separately.
