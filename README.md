# @nacho-labs/nachos-embeddings

Vector embeddings and semantic search for Node.js. Embeds locally with [Transformers.js](https://huggingface.co/docs/transformers.js) by default (no API key, no network after the first model download), or through AWS Bedrock or your own provider. Includes an in-memory vector store, safe JSON persistence, and a small CLI for non-JS callers.

## Install

```bash
npm install @nacho-labs/nachos-embeddings
```

- Node.js >= 20.9, **ESM only** (`import`, not `require`).
- The default model (`Xenova/all-MiniLM-L6-v2`, 384 dimensions, ~90MB on disk) is downloaded on first use, then cached.
- Bedrock is optional: install `@aws-sdk/client-bedrock-runtime` (plus `@aws-sdk/credential-provider-ini` for the `profile` strategy and `@aws-sdk/credential-providers` for `role`) only if you use it.

## Quick start

```typescript
import { SemanticSearch } from "@nacho-labs/nachos-embeddings";

const search = new SemanticSearch();
await search.init(); // loads the model (downloads it on first run)

await search.addDocument({
  id: "pref-1",
  text: "User loves breakfast tacos",
  metadata: { kind: "preference" },
});

const results = await search.search("What does the user eat in the morning?", {
  limit: 5,
  filter: (meta) => meta?.kind === "preference",
});
// [{ id: "pref-1", similarity: <score>, text: "User loves breakfast tacos", metadata: {...} }]
```

`init()` must be called before any other method. Search is silent by default; see `logger` and `verbose` below.

## SemanticSearch

`new SemanticSearch<Metadata>(options?)`. All options are optional.

| Option            | Default                   | Description                                                                                                                                                                                                                    |
| ----------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `provider`        | local Transformers        | Any `EmbeddingProvider`. Overrides `model`, `cacheDir`, `dtype`, `pooling`, `queryPrefix`, `documentPrefix`, `maxLength`, `progressLogging`.                                                                                   |
| `model`           | `Xenova/all-MiniLM-L6-v2` | Transformers.js model id (default provider only).                                                                                                                                                                              |
| `cacheDir`        | per-user dir              | Model cache directory (see [Transformers](#transformers-default)).                                                                                                                                                             |
| `progressLogging` | `false`                   | Log model download progress.                                                                                                                                                                                                   |
| `minSimilarity`   | `0.4`                     | Minimum cosine similarity (0-1) for a result. Overridable per search.                                                                                                                                                          |
| `deduplicate`     | `"exact"`                 | `"off"`, `"exact"` (identical text), `"normalized"` (case and whitespace-insensitive), or `{ similarity: 0.95 }` (also skips near-duplicates by embedding cosine similarity). Not applied to pieces of auto-chunked documents. |
| `autoChunk`       | `false`                   | Split long documents into overlapping chunks with ids `<id>#chunk<n>` and metadata `chunkIndex` / `parentId`. Removing the parent removes its chunks.                                                                          |
| `maxChunkTokens`  | `500`                     | Chunk size in estimated tokens (about 4 characters each).                                                                                                                                                                      |
| `chunkOverlap`    | `50`                      | Overlap between chunks, in estimated tokens.                                                                                                                                                                                   |
| `temporalBoost`   | `false`                   | Stamp new documents with `metadata.timestamp` and re-rank results toward recent ones (70% similarity, 30% recency). Per-search override. The boost is applied before `limit`.                                                  |
| `storePath`       | none                      | JSON store file used by `persist()`, `autoSave` and `loadOnInit`.                                                                                                                                                              |
| `autoSave`        | `false`                   | Write the store after every mutation. Requires `storePath` (the constructor throws otherwise).                                                                                                                                 |
| `loadOnInit`      | `true`                    | Load `storePath` during `init()`, and before the first write if `init()` was skipped.                                                                                                                                          |
| `onCorruptStore`  | `"throw"`                 | `"throw"` rejects with `StoreCorruptError`; `"quarantine"` renames the file to `<file>.corrupt-<timestamp>` and starts empty.                                                                                                  |
| `logger`          | silent                    | `{ debug, info, warn, error }` (each optional).                                                                                                                                                                                |
| `verbose`         | `false`                   | When no `logger` is given, log to the console.                                                                                                                                                                                 |
| `onError`         | none                      | Called when an `autoSave` write fails.                                                                                                                                                                                         |

Deprecated option aliases still work: `deduplication` (boolean), `deduplicateExact` (`true` = `"normalized"`), `deduplicateSimilarity` (n = `{ similarity: n }`), and `cacheEmbeddings` (no effect). `deduplicate` wins when both are set.

`SearchOptions`: `limit` (default 10), `minSimilarity`, `filter(metadata)`, `temporalBoost`.

| Method                      | Description                                                        |
| --------------------------- | ------------------------------------------------------------------ |
| `init()`                    | Initialize the provider and load the store if `storePath` is set.  |
| `addDocument(doc)`          | Add `{ id, text, metadata? }`. An existing id is replaced.         |
| `addDocuments(docs)`        | Batch add; one embedding call for the batch.                       |
| `search(query, opts?)`      | Results `{ id, similarity, text, metadata }`, best first.          |
| `remove(id)`                | Remove a document (and its chunks). Resolves `true` if it existed. |
| `clear()`                   | Remove everything.                                                 |
| `size()`                    | Number of indexed entries (chunks count individually).             |
| `getDocument(id)`           | `{ id, text, metadata }` without the vector, or `undefined`.       |
| `listDocuments()`           | `{ id, metadata }` for every entry; no text, no vectors.           |
| `export()` / `import(data)` | Documents with vectors as plain data.                              |
| `load()`                    | Read `storePath` into memory (existing in-memory documents win).   |
| `persist()`                 | Write `storePath` now. Rejects if the write fails.                 |
| `getConfig()`               | Resolved options (defaults applied).                               |
| `isInitialized()`           | Whether the provider is ready.                                     |

Mutations are serialized, so concurrent `addDocument` calls dedupe correctly.

### Persistence

- Writes are atomic: a temp file in the same directory is fsynced and renamed over the store, so a crash or failed write leaves the previous file intact. Writes go through one coalescing queue; the file always reflects the latest state.
- A missing store file is an empty store.
- A store that exists but is not valid (bad JSON, not an array, malformed entries) makes `init()`/`load()` reject with `StoreCorruptError` (`error.path` is the file), and it is **never overwritten**. Set `onCorruptStore: "quarantine"` to move it aside and start fresh.
- `autoSave` failures go to `onError` and `logger.error`, and are rethrown once by the next `persist()`.
- Vectors are tied to the model that produced them. After changing model, dtype or provider, re-index.

```typescript
const search = new SemanticSearch({
  storePath: "./index.json",
  autoSave: true,
  onCorruptStore: "quarantine",
  logger: { warn: (m) => console.warn(m) },
});
await search.init();
```

## Providers

### Transformers (default)

`Embedder` (a thin wrapper) and `TransformersProvider` run models locally.

```typescript
import { Embedder } from "@nacho-labs/nachos-embeddings";

const embedder = new Embedder({ dtype: "q8" });
await embedder.init();
const vec = await embedder.embed("hello"); // number[384]
```

Options: `model`, `cacheDir`, `dtype`, `pooling`, `queryPrefix`, `documentPrefix`, `maxLength`, `progressLogging`. Methods: `init()`, `embed()`, `embedBatch()`, `getDimension()`, `getModelId()`, `getConfig()`, `isInitialized()`, `dispose()`.

**Cache directory**, first match wins: `cacheDir` option (CLI `--cache-dir`), `$NACHOS_EMBEDDINGS_CACHE`, `$XDG_CACHE_HOME/nachos-embeddings`, `~/.cache/nachos-embeddings`. On Windows: `%LOCALAPPDATA%\nachos-embeddings`. `resolveDefaultCacheDir()` returns the result.

**dtype**: `auto`, `fp32`, `fp16`, `q8`, `int8`, `uint8`, `q4`, `bnb4`, `q4f16` (`TRANSFORMERS_DTYPES`). Unset uses the library default (fp32). In this release's benchmark, `q8` embedded batches about 3x faster than fp32, with cosine similarity of about 0.99 to the fp32 vectors. A non-default dtype is appended to the model id (`Xenova/all-MiniLM-L6-v2:q8`) and vectors differ from fp32, so do not mix them in one index.

### AWS Bedrock

```typescript
import { SemanticSearch } from "@nacho-labs/nachos-embeddings";
import { BedrockProvider } from "@nacho-labs/nachos-embeddings/bedrock";

const search = new SemanticSearch({
  provider: new BedrockProvider({
    region: "us-east-1", // default
    modelId: "amazon.titan-embed-text-v2:0", // default
    modelOptions: { dimensions: 512 },
  }),
});
await search.init();
```

| Model ID                       | Dimensions               | `modelOptions`                                                      |
| ------------------------------ | ------------------------ | ------------------------------------------------------------------- |
| `amazon.titan-embed-text-v2:0` | 256, 512, 1024 (default) | `dimensions`, `normalize` (default `true`)                          |
| `cohere.embed-english-v3`      | 1024                     | `inputType` (default `search_document`), `truncate` (default `END`) |
| `cohere.embed-multilingual-v3` | 1024                     | same as above                                                       |

Other `BedrockProvider` options: `endpoint` (e.g. a VPC endpoint), `batchSize` (25), `maxConcurrency` (5), `timeout` in ms (30000), `retry: { maxAttempts: 3, backoffMs: 200 }` (backoff doubles per retry), `progressLogging`.

**Credentials** (`credentials.strategy`): `default` (AWS SDK chain: env, shared config, instance profile), `profile` (`profile`), `explicit` (`accessKeyId`, `secretAccessKey`, optional `sessionToken`), `role` (`roleArn`, optional `roleSessionName`, `externalId`).

`getConfig()` redacts `secretAccessKey` and `sessionToken` as `"[redacted]"` and returns copies, so it is safe to log.

**Custom model adapter** for other Bedrock models:

```typescript
import {
  BedrockProvider,
  type BedrockModelAdapter,
} from "@nacho-labs/nachos-embeddings/bedrock";

const adapter: BedrockModelAdapter = {
  modelName: "My Model",
  defaultDimension: 768,
  formatRequest: (text, options) => JSON.stringify({ input: text, ...options }),
  parseResponse: (body) => JSON.parse(body).vector,
  // validateOptions?(options) { ... }  // optional, throw on invalid
};
new BedrockProvider({ modelId: "acme.custom-embed-v1", modelAdapter: adapter });
```

### Custom provider

Implement `EmbeddingProvider` (exported from the package root and `@nacho-labs/nachos-embeddings/providers`):

```typescript
interface EmbeddingProvider {
  readonly name: string;
  init(): Promise<void>;
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
  getDimension(): Promise<number | null>;
  isInitialized(): boolean;
  getModelId?(): string; // optional: actual model id, e.g. to detect re-embed needs
  dispose?(): Promise<void>; // optional: release sessions / clients
}
```

`init()` may be called more than once and concurrently; the built-in providers memoize it and allow retry after a failure.

## VectorStore and utilities

```typescript
import { VectorStore } from "@nacho-labs/nachos-embeddings";

const store = new VectorStore({ minSimilarity: 0.4, defaultLimit: 10 });
store.add("a", vecA, { any: "metadata" });
store.search(queryVec, { limit: 5, filter: (m) => true });
```

Methods: `add`, `get`, `has`, `remove`, `clear`, `size`, `keys`, `export`, `import`, `search`. Utilities: `cosineSimilarity(a, b)`, `normalizeVector(v)`, `chunkText(text, { maxTokens, overlapTokens })`, `estimateTokens(text)`, `normalizeText(text)`. Defaults are exported from the root too: `TRANSFORMERS_DEFAULT_MODEL`, `TRANSFORMERS_DEFAULT_DIMENSION`, `TRANSFORMERS_DTYPES`, `resolveDefaultCacheDir`; and from `/bedrock`: `BEDROCK_DEFAULT_MODEL_ID`, `BEDROCK_DEFAULT_REGION`.

The in-memory store suits tens of thousands of vectors at most; beyond that use a dedicated vector database.

**Query vs document, prefixes and pooling.** Retrieval models are trained with asymmetric inputs: a search query and the passages it should find get different prefixes. `embed()` and `embedBatch()` take `{ type: "query" | "document" }` (default `"document"`, which is what 0.5.x did). `SemanticSearch` embeds documents as `"document"` and `search()` queries as `"query"` for you. Prefixes and pooling default per model from a small built-in table (matched on the model id, case-insensitive):

| Model id contains          | Pooling | Query prefix                                                | Document prefix     |
| -------------------------- | ------- | ----------------------------------------------------------- | ------------------- |
| `e5-` (e5-small-v2, ...)   | `mean`  | `query: `                                                   | `passage: `         |
| `bge-*-en-v1.5`            | `cls`   | `Represent this sentence for searching relevant passages: ` | none                |
| `nomic-embed-text-v1.5`    | `mean`  | `search_query: `                                            | `search_document: ` |
| MiniLM and any other model | `mean`  | none                                                        | none                |

Override with the `pooling` (`mean` or `cls`), `queryPrefix` and `documentPrefix` options (CLI: `--pooling`, `--query-prefix`, `--document-prefix`; pass `""` to disable a prefix). Vectors are always L2-normalized. `getConfig()` reports the active values. An overridden `pooling` or `documentPrefix` that differs from the model's default is appended to `getModelId()` (e.g. `...:pooling=cls`), because it changes stored vectors. Switching a prefixed model from 0.5.x to this release changes its document vectors (0.5.x embedded without prefixes), so re-index.

**Truncation.** Inputs are truncated to the tokenizer's own limit (`model_max_length`, typically 512 tokens, fewer for some models); longer text is cut silently, never an error. `maxLength` (CLI `--max-length <tokens>`) lowers that limit, counting special tokens; it cannot raise it. For long documents use `autoChunk` instead of relying on truncation.

**Batching.** `embedBatch` runs one padded pipeline call per chunk of 8 texts, with texts ordered by length first so chunks pad to similar lengths. Results come back in input order and match single-text vectors (cosine > 0.999, checked against a real model). On this release's benchmark (64 texts, CPU) it was 1.0x to 1.8x faster than the previous concurrent single-text calls; chunks of 32 were up to 1.5x slower on mixed-length input because of padding, hence 8.

#### Recommended retrieval models

The default `Xenova/all-MiniLM-L6-v2` is a fine general starter. For retrieval, models trained with query/document prefixes usually rank better. Sizes are the fp32 `model.onnx` download (`dtype: "q8"` is smaller); all four rows were run end to end in this release (info, embed, serve, ranking a query against four documents).

| Model                            | Dim | Download (fp32) | Notes                                                                                                   |
| -------------------------------- | --- | --------------- | ------------------------------------------------------------------------------------------------------- |
| `Xenova/all-MiniLM-L6-v2`        | 384 | 90 MB           | Default. No prefixes.                                                                                   |
| `BAAI/bge-small-en-v1.5`         | 384 | 133 MB          | `cls` pooling, query instruction. Smoke-tested.                                                         |
| `Xenova/e5-small-v2`             | 384 | 133 MB          | `query: ` / `passage: `. Smoke-tested. Use the `Xenova/` repo: `intfloat/e5-small-v2` has no ONNX file. |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 547 MB          | `search_query: ` / `search_document: `. Smoke-tested.                                                   |

```bash
npx nachos-embeddings serve --model BAAI/bge-small-en-v1.5
```

## CLI

For callers that are not JavaScript. All diagnostics go to stderr; stdout carries only data.

```bash
npx nachos-embeddings embed --stdin --json < texts.txt   # one text per line in, one JSON vector per line out
npx nachos-embeddings serve                              # persistent JSON-lines server
npx nachos-embeddings info [--json]                      # provider, model, dimension, cache dir
```

| Flag                                          | Applies to   | Description                                                                          |
| --------------------------------------------- | ------------ | ------------------------------------------------------------------------------------ |
| `--stdin`, `--json`                           | `embed`      | Both required. Output order matches input order. Empty input exits 0 with no output. |
| `--provider <name>`                           | all          | `transformers` (default) or `bedrock`. Anything else is a usage error.               |
| `--model <id>`                                | all          | Transformers model name, or Bedrock model id.                                        |
| `--region <region>`                           | bedrock      | AWS region (default `us-east-1`).                                                    |
| `--cache-dir <dir>`                           | transformers | Model cache directory (default: per-user, see above).                                |
| `--dtype <type>`                              | transformers | Precision, e.g. `q8`. Appended to the reported model id.                             |
| `--type <t>`                                  | `embed`      | `query` or `document` (default). Selects the model's prefix. In `serve`, see below.  |
| `--pooling <mode>`                            | transformers | `mean` or `cls`. Default per model.                                                  |
| `--query-prefix <s>`, `--document-prefix <s>` | transformers | Override the model's prefixes (`""` disables).                                       |
| `--max-length <n>`                            | transformers | Truncate to `n` tokens (can only lower the tokenizer limit).                         |

**`serve`** loads the provider once, then handles one request per stdin line, sequentially and in order. First stdout line:

```json
{
  "ready": true,
  "model": "Xenova/all-MiniLM-L6-v2",
  "dimension": 384,
  "pooling": "mean",
  "queryPrefix": "",
  "documentPrefix": ""
}
```

`pooling`, `queryPrefix`, `documentPrefix` (and `maxLength`, when set) are optional extra fields, sent for the transformers provider only; existing readers of `model` and `dimension` are unaffected. A request may carry an optional `"type": "query"` or `"document"`; omitting it behaves exactly as before.

```text
> {"id": 1, "texts": ["hello", "world"]}
< {"id": 1, "vectors": [[0.01, ...], [0.02, ...]]}
> {"id": 2, "texts": ["what do I eat?"], "type": "query"}
< {"id": 2, "vectors": [[0.03, ...]]}
> not json
< {"id": null, "error": "invalid request: expected {id, texts: [...]}"}
```

A malformed line, a request without a `texts` array, or an unknown `type` gets an error reply (with `id: null` when the id cannot be read) and the server keeps running. An embedding failure replies `{"id": n, "error": "<message>"}`. Blank lines are ignored. On stdin EOF the provider is disposed and the process exits 0. If the reader closes stdout early (EPIPE, e.g. `serve | head -1`), the process disposes and exits 0.

**`info`** prints `provider`, `model`, `dimension`, `cacheDir` and (transformers) `pooling`, `queryPrefix`, `documentPrefix` as `key: value` lines, or one JSON object with `--json` (`cacheDir` is `null` for Bedrock).

**Exit codes:** `0` success, `1` usage error (unknown subcommand/provider, missing flag value, missing `--stdin --json`), `2` provider init or embedding failure.

## Using with Claude Code and MCP

[`@nacho-labs/mcp-semantic-search`](https://github.com/Nacho-Labs-LLC/mcp-semantic-search) is a ready-made MCP server built on this package. It gives Claude Code, Cursor and other MCP clients persistent semantic memory. To build your own, import `McpServer` from `@modelcontextprotocol/sdk/server/mcp.js` and `StdioServerTransport` from `@modelcontextprotocol/sdk/server/stdio.js`, and use `SemanticSearch` with `storePath` and `autoSave`.

## Migrating from 0.5.x

- **`EnhancedSemanticSearch` is deprecated.** `SemanticSearch` now has every feature, and the alias keeps working. Option mapping:

  | 0.5.x (`EnhancedSemanticSearch`) | 0.6.0 (`SemanticSearch`)                                                        |
  | -------------------------------- | ------------------------------------------------------------------------------- |
  | `deduplication: false`           | `deduplicate: "off"`                                                            |
  | `deduplicateExact: true`         | `deduplicate: "normalized"`                                                     |
  | `deduplicateSimilarity: 0.9`     | `deduplicate: { similarity: 0.9 }`                                              |
  | `forceSave()`                    | `persist()`                                                                     |
  | `autoSave` without `storePath`   | set `storePath` explicitly (the alias still defaults to `.semantic-store.json`) |

  Unchanged names: `autoSave`, `storePath`, `autoChunk`, `maxChunkTokens`, `chunkOverlap`, `temporalBoost`, `verbose`. Note `SemanticSearch` defaults to `deduplicate: "exact"`; the alias defaults to `"normalized"`.

- **Model cache moved.** It was `.cache/transformers` relative to the working directory; it is now per-user (see [Transformers](#transformers-default)). The first run re-downloads the ~90MB model. To keep the old copy, set `NACHOS_EMBEDDINGS_CACHE` to the old `.cache/transformers` path, or move that directory to the new location.
- **Corrupt stores now throw.** `init()`/`load()` reject with `StoreCorruptError` instead of being treated as empty and overwritten. Use `onCorruptStore: "quarantine"` for the old forgiving behaviour. A base `SemanticSearch` with `storePath` now also loads it on `init()`.
- **Silent by default.** The library no longer writes to the console. Pass `logger` or `verbose: true`.
- **`BedrockProvider.getConfig()`** returns `"[redacted]"` for `secretAccessKey` and `sessionToken`, and copies rather than internal objects.
- Temporal boost is now applied before `limit`, so results can change order for `temporalBoost` users.

See [CHANGELOG.md](CHANGELOG.md) for the full list.

## Development

```bash
npm install
npm run check   # lint, prettier, typecheck, build, tests with coverage, packed-tarball smoke test
```

`npm install` installs git hooks (pre-commit: lint, format, typecheck; pre-push: `npm run check`). Releasing is documented in [RELEASING.md](RELEASING.md).

## License

MIT
