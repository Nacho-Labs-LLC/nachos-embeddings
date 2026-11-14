# Changelog

## 0.6.0

**Breaking / migration.** Most changes are backward compatible, but four behaviours differ. See "Migrating from 0.5.x" in the README.

- The default Transformers model cache moved from `.cache/transformers` (relative to the working directory) to a per-user directory. The first run re-downloads the model (~90MB). Set `NACHOS_EMBEDDINGS_CACHE` to the old path, or move the directory, to keep it.
- A corrupt or malformed store file now makes `init()`/`load()` reject with `StoreCorruptError` and is never overwritten. Use `onCorruptStore: "quarantine"` to start fresh.
- The library no longer writes to the console. Pass `logger` or `verbose: true`.
- `BedrockProvider.getConfig()` redacts `secretAccessKey` and `sessionToken`.

### Added

- CLI `serve`: persistent JSON-lines server. Prints `{"ready":true,"model","dimension"}`, handles one `{id, texts}` request per line sequentially, replies `{id, vectors}` or `{id, error}` (`id: null` for malformed lines), and exits 0 on EOF.
- CLI `info [--json]`, and `--cache-dir` and `--dtype` flags. `--provider` is validated, and a value flag with no value is a usage error (exit 1).
- Transformers `dtype` option (`fp32`, `fp16`, `q8`, `int8`, `uint8`, `q4`, `bnb4`, `q4f16`, `auto`); a non-default dtype is appended to the model id. `q8` was measured at about 3x faster batch embedding with cosine similarity of about 0.99 to fp32.
- Optional `dispose()` and `getModelId()` on `EmbeddingProvider`, implemented for Transformers and Bedrock; the CLI disposes the provider on completion.
- Exported defaults and helpers: `TRANSFORMERS_DEFAULT_MODEL`, `TRANSFORMERS_DEFAULT_DIMENSION`, `TRANSFORMERS_DTYPES`, `resolveDefaultCacheDir`, `BEDROCK_DEFAULT_MODEL_ID`, `BEDROCK_DEFAULT_REGION` (the Bedrock ones from `/bedrock`).
- `SemanticSearch` options: `deduplicate` (`"off"`, `"exact"`, `"normalized"`, `{ similarity }`), `autoSave`, `storePath`, `loadOnInit`, `onCorruptStore`, `autoChunk`, `maxChunkTokens`, `chunkOverlap`, `temporalBoost` (also per search), `logger`, `verbose`, `onError`. New methods `getDocument`, `listDocuments`, `load`, `persist`, `getConfig`.
- `StoreCorruptError` (with `path`), and exported types `SemanticSearchOptions`, `SemanticSearchLogger`, `DeduplicateOption`, `ResolvedSemanticSearchConfig`.

### Changed

- `SemanticSearch` and `EnhancedSemanticSearch` are unified: `SemanticSearch` carries every feature, and `EnhancedSemanticSearch` is a thin deprecated alias that maps the old options (`deduplicateExact`, `deduplicateSimilarity`, `forceSave()`, default `.semantic-store.json` with `autoSave`).
- Base `SemanticSearch` now loads `storePath` on `init()` (it previously only wrote).
- Persistence is a single serialized, coalescing write queue with atomic writes (temp file, fsync, rename), async fs only. `autoSave` failures go to `onError`/`logger` and are rethrown once by the next `persist()`.
- Mutations are serialized, so concurrent duplicate adds dedupe correctly. `remove()` is O(1) and removes the chunks of a chunked parent. Fuzzy dedup reuses the new document's embedding instead of embedding twice.
- Transformers cache is per-user: `$NACHOS_EMBEDDINGS_CACHE`, else `$XDG_CACHE_HOME/nachos-embeddings`, else `~/.cache/nachos-embeddings` (Windows: `%LOCALAPPDATA%\nachos-embeddings`). An explicit `cacheDir`/`--cache-dir` still wins.
- Transformers no longer mutates global `env.cacheDir`; cache dir and dtype are passed per pipeline call, and `env.allowLocalModels` is set only during a load (loads are serialized).
- `serve` reports the provider's actual model id via `getModelId()`.
- Refactors merged to master since 0.5.2: consolidated `chunkText` logic (#61), extracted shared duplicate detection (#63), batched `addDocuments` (#65), faster `cosineSimilarity` (#66), and removed a stray `semantic-search.ts.orig` (#62).
- Process: PR CI workflow, a single `npm run check` gate (lint, format, typecheck, build, tests, packed-tarball smoke test), `simple-git-hooks` pre-commit/pre-push, v8 coverage thresholds, stricter ESLint (no `any`, complexity, max-depth, max-lines-per-function, max-params, no-console, type-aware promise rules), Prettier over the whole repo, actions pinned by SHA, and Dependabot grouping (aws-sdk, dev, production, plus github-actions). Dev dependencies updated (vitest 5, eslint 10, typescript-eslint 8.72, and others).

### Fixed

- `init()` is memoized for Transformers and Bedrock (concurrent callers share one load; a failure resets it so a retry works).
- Bedrock `init()` commits state only after the dimension probe succeeds, and destroys the failed client. Bedrock dependency load errors keep `{ cause }`; only a real missing module produces the "install it" message.
- The CLI handles EPIPE on stdout (e.g. `serve | head -1`) by disposing and exiting 0.
- Temporal boost is applied before the result limit, so recent documents are no longer cut off by the semantic top-N.
- A write can no longer clobber a store that was not read (writes load first if `init()` was skipped).

### Security

- `BedrockProvider.getConfig()` no longer exposes `secretAccessKey` or `sessionToken` (replaced with `"[redacted]"`) and returns copies of nested objects.
- Runtime and dev dependency vulnerabilities fixed: `@huggingface/transformers` ^4.3.2 (adm-zip 0.6.1 and sharp 0.35.5 now resolve naturally, so those overrides were removed), `brace-expansion` 5.0.13 and `minimatch` 11.0.8 overrides. `npm audit` reports 0 vulnerabilities.

### Deprecated

- `EnhancedSemanticSearch` and `EnhancedSemanticSearchConfig`: use `SemanticSearch` / `SemanticSearchOptions`.
- Options `deduplication`, `deduplicateExact`, `deduplicateSimilarity`: use `deduplicate`.
- `forceSave()`: use `persist()`.
- `cacheEmbeddings` on `SemanticSearch` and `VectorStoreConfig` (it never had any effect).

### Removed

- The local `release:*` publish/version npm scripts. Releases are published from CI only; see RELEASING.md.
- Source maps and declaration maps from the published package.
- `docs/claude-code-integration.md` (a stub pointing at the README).

## 0.5.2

- Remediated audit findings and simplified modules (#54).
- Prepared the v0.5.2 release (#55).

## 0.5.1

- Fixed the npm publish tag version check and the publish retry path in CI (#49, #52).
- Aligned provenance repository metadata (#53).
- Dependency updates (#50, #51).

## 0.5.0

- Maintenance release: consolidated supported dependency updates and lockfile security fixes.
- Added an enforced format-check script and normalized existing formatting.
- Documented the temporary TypeScript 6 hold pending typescript-eslint support for TypeScript 7.

Earlier versions are available on npm only.
