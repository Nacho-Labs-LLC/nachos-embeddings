# Changelog

## 0.7.0

Adds query/document input types with per-model prefixes and pooling, more model profiles, a local cross-encoder reranker, offline mode, and customizable models. Everything new is additive except the vector changes listed under "Migration from 0.6.x".

### Migration from 0.6.x

Checked against `git diff origin/master..HEAD -- src bin` (master being the published 0.6.0).

- **Vectors change for e5, bge and nomic models. Re-index stores built with them.** 0.6.0 embedded every text the same way (mean pooling, no prefix). Now e5 documents carry `passage: ` (and queries `query: `), nomic documents carry `search_document: ` (queries `search_query: `) and nomic applies the layer norm its card prescribes (cosine about 0.9999 to the previous vectors, plus the prefix change), and bge models switch to cls pooling with the retrieval instruction on queries. MiniLM (the default) and unknown models are unchanged. `SemanticSearch` now embeds documents as `"document"` and `search()` queries as `"query"`, so the query side of a store changes too. Pass `queryPrefix: ""`, `documentPrefix: ""` and `pooling: "mean"` to reproduce 0.6.0 vectors for a model.
- **`serve --max-length` no longer applies to the reranker.** In `serve`, `--max-length` is for the embedding model only; the reranker has its own `--rerank-max-length`. (`rerank` still falls back to `--max-length`.)
- **`getModelId()` can differ.** It now appends overrides that change vectors (`:pooling=...`, a document prefix, `normalize=false`), so a store keyed on the id sees a new id when you set those.
- **Model loading is cache-first.** A fully cached model now loads with no network request; files are downloaded only when missing. Before, Transformers.js asked the hub on every load.
- **`EmbeddingProvider.embed/embedBatch` take an optional second `EmbedOptions` argument.** Existing custom providers keep compiling and ignore it.
- **`serve` ready line and `info` gain optional fields** (`version`, `pooling`, `queryPrefix`, `documentPrefix`, `maxLength`, `normalize`, `rerank`, `offline`). Existing readers that ignore unknown keys are unaffected. Requests may also carry `type` or `op: "rerank"`.
- **Model ids that look like paths** (`./dir`, `/abs/dir`, `~/dir`, `file://...`) are now loaded as local directories instead of being sent to the hub.
- **Version detection.** The 0.6.0 CLI has no `--version` and no `version` in the ready line. Clients can treat a CLI without them as older than 0.7.0.

### Added

- CLI `--version` prints the package version (exit 0), and the `serve` ready line carries `"version"`, so clients can tell a 0.7.0+ CLI from an older one that has neither.
- Query/document input types: `embed(text, { type })` and `embedBatch(texts, { type })` (`"query"` or `"document"`, default `"document"`), CLI `embed --type`, and an optional per-request `"type"` in `serve` (requests without it behave as before). `SemanticSearch` embeds documents as `"document"` and `search()` queries as `"query"`.
- Built-in per-model prefix and pooling table (e5: `query: ` / `passage: `; bge-*-en-v1.5: cls pooling and the retrieval instruction on queries; nomic-embed-text-v1.5: `search_query: ` / `search_document: `; MiniLM and unknown models: none, mean), overridable with `queryPrefix`, `documentPrefix`, `pooling` options and `--query-prefix`, `--document-prefix`, `--pooling`. Exports `getModelProfile`, `POOLING_MODES`, `INPUT_TYPES` and the `EmbedOptions`, `PoolingMode`, `InputType`, `ModelProfile` types. Overrides that change document vectors are appended to `getModelId()`.
- Prefix/pooling profiles for `snowflake-arctic-embed-*` (cls; `Represent this sentence for searching relevant passages: ` on queries, or `query: ` for the v2.0 models) and `gte-modernbert-base` (cls, no prefixes), checked against their model cards. `nomic-embed-text-v1.5` now applies the card's layer norm between mean pooling and L2 normalization (vectors change slightly, cosine about 0.9999 to the previous ones).
- `maxLength` option and `--max-length` to truncate inputs to fewer tokens than the tokenizer limit (which still applies; the default is unchanged).
- `info` and the `serve` ready line report `pooling`, `queryPrefix`, `documentPrefix` (and `maxLength` when set) for the transformers provider; new optional fields only.
- Optional local cross-encoder reranker: `rerank(query, passages, { model, dtype, maxLength, cacheDir })` and the `Reranker` class return one score per passage in input order (`sigmoid(logit)`, in (0, 1)). Default `jinaai/jina-reranker-v1-tiny-en` at q8 (about 35 MB; loads in Transformers.js without remote code), documented fallback `Xenova/ms-marco-MiniLM-L-6-v2` (about 24.5 MB). It is lazy, separate from the embedding model, truncates pairs at `maxLength` tokens (see profiles below) and scores in length-sorted chunks of 8. Exports `Reranker`, `rerank`, `disposeRerankers`, `RERANK_DEFAULT_MODEL`, `RERANK_FALLBACK_MODEL`, `RERANK_DEFAULT_DTYPE`, `RERANK_DEFAULT_MAX_LENGTH`.
- CLI `rerank --query <q> --stdin --json` (one passage per line, prints one JSON array of scores). `serve --rerank-model/--rerank-dtype` accepts `{"id","op":"rerank","query","passages"}` and replies `{"id","scores"}`, and adds `"rerank":{"model"}` to the ready line only when enabled; requests without `op` and a `serve` without the flags are unchanged and load no reranker.
- Offline and privacy: a model whose files are all cached now loads with zero network requests, for embedding and reranker models. Transformers.js 4 otherwise asks the hub which tokenizer files exist on every load, even when cached. Files are downloaded only when missing. New `offline` option (embedder, `TransformersProvider`, `Reranker`, `rerank()`), CLI `--offline` and `NACHOS_EMBEDDINGS_OFFLINE=1` never touch the network and fail with `ModelNotCachedError` (naming the missing files and the `pull` fix) when a model is not cached; `serve` reports `"offline": true` in the ready line only when on, and `--provider bedrock` refuses to run offline. New `nachos-embeddings pull <model> [--dtype] [--cache-dir] [--kind embedding|rerank]` pre-downloads a model for air-gapped setups. Exports `ModelNotCachedError`, `isOfflineEnv`, `OFFLINE_ENV_VAR`.
- Reranker profiles (`getRerankerProfile`): `jinaai/jina-reranker-v1-tiny-en` has ALiBi positions, so its tokenizer limit is raised and `maxLength` defaults to 1024 (up to 8192; scores for short inputs are unchanged). Models without a profile (MiniLM) stay at 512, because their graph fails above that. `--max-length` applies to `rerank` and `pull`.
- Customizable models: any Hugging Face ONNX encoder or single-logit cross-encoder works, with explicit `pooling`, `queryPrefix`, `documentPrefix`, `maxLength`, `dtype` and the new `normalize` option (`--normalize` / `--no-normalize`) always beating the built-in profile. A local model directory (`./dir`, `/abs/dir`, `~/dir`, `file://`; ONNX in `onnx/` or at the top level) loads fully offline.
- New `models [--json]` command (built-in profiles with dimension, licence, download size per dtype and pooling/prefixes, plus cached models) and `info <model>` (profile, cached files, architecture and dimension without downloading; `info` with no id is unchanged). Exports `MODEL_CATALOG`, `findCatalogEntry`, `profileFor`, `listCachedModels`, `inspectModel`.
- `serve --rerank-max-length <n>` gives the reranker its own truncation length. Before, `--max-length` was applied to both the embedding model and the reranker in `serve`; it now applies to the embedding model only (`rerank` still falls back to `--max-length`).
- Clear model errors: `ModelLoadError` with codes `not-found`, `no-onnx-weights`, `unsupported-architecture` and `local-model-incomplete`. Generative and encoder-decoder models are rejected from `config.json` before any weights are downloaded.
- README: recommended retrieval models (bge-small-en-v1.5, e5-small-v2, nomic-embed-text-v1.5) with download sizes.

### Changed

- Transformers `embedBatch` now makes one padded pipeline call per chunk of 8 texts (length-sorted, results in input order) instead of concurrent single-text calls: 1.0x to 1.8x faster in a CPU benchmark, vectors match single calls (cosine > 0.999).
- `info` and the `serve` ready line report the active pooling and prefixes for the transformers provider.
- Tests no longer depend on the real model cache: provider tests use a tiny local model fixture, and a network blocker (`scripts/block-network.mjs`) backs the offline tests. The packed-tarball smoke test (`SMOKE_MODEL=1`) also covers `rerank`, `serve` ops and offline loading.

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
