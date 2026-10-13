# Vendored UI libraries

Minified third-party scripts served at `/web/<name>` for the dashboard graph
view. They are static code (no index data), loaded by `index.html`, and are
the only files the `/web/` route will serve (allowlist in `http_api.py`).

| File | Library | sha256 |
| --- | --- | --- |
| `graphology.min.js` | [graphology](https://github.com/graphology/graphology) (MIT) | `641ea047e2f414dead999769d62567ce3c6f1ddc334f1e728bd5edb19d337977` |
| `sigma.min.js` | [sigma.js](https://github.com/jacomyal/sigma.js) (MIT) | `a86b7eb3578e9028ee84d792bf20f1503f0335ce769d5ad454252dc0a5787618` |
| `fa2.min.js` | [graphology-layout-forceatlas2](https://github.com/graphology/graphology) (MIT) | `717f607ba7d7e002c7610aa7894927e8be847156ec0f0f86e71d519ea7b5e767` |

The upstream version of each build was not recorded when these files were
added. Before updating one, replace it with a build from a pinned upstream
release, record that version here, and refresh the hash
(`sha256sum src/indexedcode/web/*.min.js`).
