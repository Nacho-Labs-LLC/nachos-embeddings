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

Options: `model`, `cacheDir`, `dtype`, `pooling`, `queryPrefix`, `documentPrefix`, `maxLength`, `offline`, `progressLogging`. Methods: `init()`, `embed()`, `embedBatch()`, `getDimension()`, `getModelId()`, `getConfig()`, `isInitialized()`, `dispose()`.

**Cache directory**, first match wins: `cacheDir` option (CLI `--cache-dir`), `$NACHOS_EMBEDDINGS_CACHE`, `$XDG_CACHE_HOME/nachos-embeddings`, `~/.cache/nachos-embeddings`. On Windows: `%LOCALAPPDATA%\nachos-embeddings`. `resolveDefaultCacheDir()` returns the result.

**dtype**: `auto`, `fp32`, `fp16`, `q8`, `int8`, `uint8`, `q4`, `bnb4`, `q4f16` (`TRANSFORMERS_DTYPES`). Unset uses the library default (fp32). In this release's benchmark, `q8` embedded batches about 3x faster than fp32, with cosine similarity of about 0.99 to the fp32 vectors. A non-default dtype is appended to the model id (`Xenova/all-MiniLM-L6-v2:q8`) and vectors differ from fp32, so do not mix them in one index.

### Privacy and offline use

Transformers.js 4 contacts the Hugging Face hub on **every** model load, even when every file is already cached: to decide which tokenizer files exist it sends a request for `tokenizer_config.json` that ignores the cache directory. Measured on this release, a fully cached MiniLM load resolved `huggingface.co` each time, which leaks that you are running the model and makes loads flaky or slow without a connection.

This package avoids that for both the embedding and the reranker models:

- **Cached means no network.** Before loading, it checks the cache for the model's `config.json`, `tokenizer.json`, `tokenizer_config.json` and the ONNX weights for the requested `dtype`. If they are all there, it loads with `local_files_only`, which makes no request. Files are downloaded only when some are missing (first use, or a new `dtype`).
- **Offline mode never touches the network.** Pass `offline: true` (CLI `--offline`) or set `NACHOS_EMBEDDINGS_OFFLINE=1` (also `true`, `yes`, `on`). A model that is not fully cached then fails with a `ModelNotCachedError` that names the missing files and the fix, instead of downloading. The explicit option wins over the environment variable (`offline: false` overrides it). `BedrockProvider` is a network service, so the CLI refuses `--provider bedrock` in offline mode.
- **`pull` pre-downloads.** `nachos-embeddings pull <model> [--dtype q8] [--cache-dir <dir>]` fetches an embedding or reranker model into the cache and prints `{"pulled":true,"kind","model","cacheDir"}`. The kind is guessed from the id (names containing `rerank`, `cross-encoder` or `ms-marco` are rerankers); `--kind embedding|rerank` overrides it. `pull` is the one command that needs the network, so it refuses to run with `--offline` or `NACHOS_EMBEDDINGS_OFFLINE=1`.

Air-gapped setup: run `pull` on a connected machine for each model and dtype you will use, copy the cache directory (`$NACHOS_EMBEDDINGS_CACHE`, default `~/.cache/nachos-embeddings`) to the target, and run with `NACHOS_EMBEDDINGS_OFFLINE=1`. `serve` announces `"offline": true` in its ready line when offline mode is on.

How this was verified: the vitest suite runs the real Transformers.js against a tiny cached tokenizer with a fake `env.fetch` (cached load: zero requests; the same load through plain Transformers.js: requests; offline plus a missing model: a clear error and zero requests). The packed-tarball smoke test (`SMOKE_MODEL=1 npm run smoke:pack`) then runs the real models a second time under `scripts/block-network.mjs`, a preload that patches the socket layer (`net.Socket#connect` and `dns`, which fetch, undici, http and https all use) to throw and log any attempt, and checks that embed and rerank succeed with an empty attempt log, with and without `--offline`. Not covered: this proves the package's own load path; it cannot vouch for other code in your process.

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

| Model id contains                                        | Pooling | Query prefix                                                | Document prefix     |
| -------------------------------------------------------- | ------- | ----------------------------------------------------------- | ------------------- |
| `e5-` (e5-small-v2, ...)                                 | `mean`  | `query: `                                                   | `passage: `         |
| `bge-*-en-v1.5`                                          | `cls`   | `Represent this sentence for searching relevant passages: ` | none                |
| `nomic-embed-text-v1.5`                                  | `mean`  | `search_query: `                                            | `search_document: ` |
| `snowflake-arctic-embed-*` (xs, s, m, l, m-long, m-v1.5) | `cls`   | `Represent this sentence for searching relevant passages: ` | none                |
| `snowflake-arctic-embed-*-v2.0` (m, l)                   | `cls`   | `query: `                                                   | none                |
| `gte-modernbert-base`                                    | `cls`   | none                                                        | none                |
| MiniLM and any other model                               | `mean`  | none                                                        | none                |

Override with the `pooling` (`mean` or `cls`), `queryPrefix` and `documentPrefix` options (CLI: `--pooling`, `--query-prefix`, `--document-prefix`; pass `""` to disable a prefix). Vectors are always L2-normalized. `nomic-embed-text-v1.5` additionally gets the parameter-free layer norm between pooling and normalization that its model card prescribes (built in, not configurable). `getConfig()` reports the active values. An overridden `pooling` or `documentPrefix` that differs from the model's default is appended to `getModelId()` (e.g. `...:pooling=cls`), because it changes stored vectors. Switching a prefixed model from 0.5.x to this release changes its document vectors (0.5.x embedded without prefixes), so re-index.

**Truncation.** Inputs are truncated to the tokenizer's own limit (`model_max_length`, typically 512 tokens, fewer for some models); longer text is cut silently, never an error. `maxLength` (CLI `--max-length <tokens>`) lowers that limit, counting special tokens; it cannot raise it. For long documents use `autoChunk` instead of relying on truncation.

**Batching.** `embedBatch` runs one padded pipeline call per chunk of 8 texts, with texts ordered by length first so chunks pad to similar lengths. Results come back in input order and match single-text vectors (cosine > 0.999, checked against a real model). On this release's benchmark (64 texts, CPU) it was 1.0x to 1.8x faster than the previous concurrent single-text calls; chunks of 32 were up to 1.5x slower on mixed-length input because of padding, hence 8.

#### Recommended retrieval models

The default `Xenova/all-MiniLM-L6-v2` is a fine general starter. For retrieval, models trained with query/document prefixes usually rank better. Sizes are the fp32 `model.onnx` download (`dtype: "q8"` is smaller); rows marked smoke-tested were run end to end in this release (info, embed, serve, ranking a query against four documents); the q8 size is the `model_quantized.onnx` download, and latency is the median of 20 single-text `embed()` calls on a 16-core CPU.

| Model                                | Dim | Download (fp32 / q8) | Notes                                                                                                   |
| ------------------------------------ | --- | -------------------- | ------------------------------------------------------------------------------------------------------- |
| `Xenova/all-MiniLM-L6-v2`            | 384 | 90 MB / -            | Default. No prefixes.                                                                                   |
| `BAAI/bge-small-en-v1.5`             | 384 | 133 MB / -           | `cls` pooling, query instruction. Smoke-tested.                                                         |
| `Xenova/e5-small-v2`                 | 384 | 133 MB / -           | `query: ` / `passage: `. Smoke-tested. Use the `Xenova/` repo: `intfloat/e5-small-v2` has no ONNX file. |
| `nomic-ai/nomic-embed-text-v1.5`     | 768 | 547 MB / -           | `search_query: ` / `search_document: `, plus layer norm. Smoke-tested.                                  |
| `Snowflake/snowflake-arctic-embed-s` | 384 | 133 MB / 34 MB       | `cls` pooling, query instruction. Smoke-tested: 16.6 ms fp32, 4.8 ms q8 per text.                       |
| `Alibaba-NLP/gte-modernbert-base`    | 768 | 596 MB / 150 MB      | `cls` pooling, no prefixes, long context. Smoke-tested: 36 ms fp32, 17 ms q8 per text.                  |

```bash
npx nachos-embeddings serve --model BAAI/bge-small-en-v1.5
```

## Using any model

The tested models above have built-in profiles, but any Hugging Face encoder with an ONNX export works for embeddings (a BERT, RoBERTa, XLM-R, DistilBERT, MPNet or ModernBERT-style model with an `onnx/` folder, which `Xenova/...` and `onnx-community/...` repositories provide), and any single-logit cross-encoder works for reranking. Pass its id as `model` / `--model`.

**Explicit options always beat the built-in profile.** The profile only supplies defaults for the model ids it recognises.

| Option (library / CLI)                                                  | What it overrides                                                                                                                                      |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `pooling` / `--pooling mean\|cls`                                       | The model's pooling (bge, arctic and gte-modernbert use `cls`; everything else `mean`).                                                                |
| `queryPrefix`, `documentPrefix` / `--query-prefix`, `--document-prefix` | The trained prefixes. `""` switches a prefix off.                                                                                                      |
| `maxLength` / `--max-length`                                            | Input length in tokens. Embedding models: only lowers the tokenizer limit. Rerankers: can raise it for models whose profile allows (jina, up to 8192). |
| `dtype` / `--dtype`                                                     | Weights precision (`fp32`, `fp16`, `q8`, `int8`, `uint8`, `q4`, `q4f16`, `bnb4`); the hub repository must ship that file.                              |
| `normalize` / `--normalize`, `--no-normalize`                           | L2 normalization of the output vector (default on). Off returns raw pooled vectors.                                                                    |

A non-default `pooling`, `documentPrefix` or `normalize` is appended to `getModelId()`, because it changes stored vectors.

**Local models.** A directory path works as the model: `--model ./my-model`, `/abs/dir`, `~/dir` or `file:///...` (a bare `org/name` is always a hub id). It needs `config.json`, `tokenizer.json`, `tokenizer_config.json` and the ONNX weights for the dtype, in `onnx/model[_quantized|_fp16|...].onnx` or at the top level (the layout `optimum-cli export onnx` writes). It loads with no network at all, whether or not `--offline` is set, and the reported model id is the absolute path. Built-in profiles match on the path text, so a directory named `my-bge-small-en-v1.5` picks up the bge profile; override as above for anything else.

**Commands.**

```bash
npx nachos-embeddings models [--json]              # built-in profiles (dims, licence, download size per dtype, pooling/prefixes) and cached models
npx nachos-embeddings info <model> [--dtype q8]    # what this model is: profile, cached files, architecture, dimension; never downloads
npx nachos-embeddings pull <model> [--dtype q8]    # download it (see Privacy and offline use)
```

**Clear errors.** Failures name the cause and the fix instead of a raw Transformers.js message, and carry a stable `code` on `ModelLoadError` (`not-found`, `no-onnx-weights`, `unsupported-architecture`, `local-model-incomplete`):

- no ONNX weights (for example `intfloat/e5-small-v2`, which has no `onnx/` folder): says which file is missing for the dtype, and suggests another `--dtype`, an ONNX repository, or exporting with `optimum-cli export onnx`;
- unknown or private repository (the hub answers 401): says to check the id or set `HF_TOKEN`;
- generative or encoder-decoder models (GPT-2, T5, Llama, ...): rejected from `config.json` before any weights are downloaded;
- unsupported architecture in Transformers.js, or a config without `model_type`;
- incomplete local directory: lists the missing files.

A reranker must emit a single logit (`num_labels` 1); any other output shape is rejected with the shape in the message.

## Reranking

A cross-encoder reads the query and one passage together and scores how well the passage answers it. It is more accurate than comparing embeddings but too slow to run over a whole corpus, so use it to re-score the top ~25 candidates of a first-stage search. The reranker is a separate model from the embedding model; a process can hold both, and nothing is downloaded or loaded until you ask for it.

```typescript
import { rerank } from "@nacho-labs/nachos-embeddings";

const scores = await rerank("how many people live in Berlin?", [
  "New York City is famous for the Metropolitan Museum of Art.",
  "Berlin has a population of 3,520,031 registered inhabitants.",
]);
// scores[i] belongs to passages[i]; higher is more relevant
```

`rerank(query, passages, { model, dtype, maxLength, cacheDir })` returns one score per passage in input order. It loads the model lazily on first use and caches it per option set; `disposeRerankers()` releases them. For explicit control use `new Reranker(options)` with `init()`, `rerank()`, `getModelId()` and `dispose()`.

- **Scores** are `sigmoid(logit)`, so they lie in (0, 1) and higher is more relevant. Both supported models are single-logit cross-encoders trained with a sigmoid head, and their reference implementations apply the same sigmoid. Treat scores as a ranking: do not compare them across models, and do not expect a fixed relevance threshold to carry between models (jina-tiny scored the relevant Berlin passage 0.93 and an unrelated one 0.28; MiniLM-MS-MARCO is far more extreme, near 1 and near 0).
- **Models.** Default `jinaai/jina-reranker-v1-tiny-en` (Apache-2.0, 4 layers, about 35 MB at q8: 33.4 MB model plus 2.0 MB tokenizer). It loads in Transformers.js with the standard `AutoModelForSequenceClassification`; it does not need remote code, because Transformers.js reads the plain BERT config and ignores the `auto_map` entries. The fallback is `Xenova/ms-marco-MiniLM-L-6-v2` (Apache-2.0, about 24.5 MB at q8: 23.1 MB plus 0.7 MB tokenizer). Both are English only.
- **`dtype`** defaults to `q8` (the small download); other dtypes fetch other, larger files. A non-default dtype is appended to `getModelId()`.
- **Length.** Each query + passage pair is truncated to `maxLength` tokens; longer text is cut silently, never an error. The limit is per model (`getRerankerProfile`, like the embedding profile table). **jina-reranker-v1-tiny-en** uses ALiBi positions, so its real limit is 8192 even though its tokenizer says 512: this package raises the tokenizer limit for it, defaults `maxLength` to **1024** (a few paragraphs, sensible for memory bodies) and lets `maxLength` / `--max-length` go up to 8192. Verified: scores for inputs under 512 tokens are unchanged, and a relevant sentence at token ~900 is found at 1024 (top rank) but invisible at 512. **ms-marco-MiniLM** has learned absolute positions: its ONNX graph fails above 512 tokens, so it stays at 512 and a larger `maxLength` is capped by the tokenizer. Cost grows with length: 25 passages of about 900 tokens took a median of 2.4 s at 1024 against 1.5 s at 512 (the 512 run cuts the passages), while 25 passages of 100 to 300 tokens cost the same either way (about 0.4 s, within noise: 459 / 395 / 397 ms at 1024 and 358 / 459 / 427 ms at 512 over three runs).
- **Batching.** Pairs are scored in padded chunks of 8, sorted by length like `embedBatch`; results come back in input order. An empty list returns `[]` without loading the model; an empty passage is scored like any other text.
- **Latency** (16-core CPU, q8, 25 passages of roughly 100 to 300 tokens, median of 5 runs after loading; four separate smoke runs): jina-tiny 412, 306, 280 and 311 ms, MiniLM 298, 370, 320 and 310 ms, so about 0.3 to 0.4 s either way on this machine (the first call of a run is slower, e.g. 482 to 533 ms). Model load from a warm cache took about 0.2 to 0.4 s. Reproduce with `SMOKE_MODEL=1 npx vitest run tests/reranker-smoke.test.ts --disableConsoleIntercept`, which also checks that an obviously relevant passage outranks distractors for both models.

From the CLI (output is one JSON array of scores on one line):

```bash
printf 'Berlin has 3.5 million inhabitants.\nNew York has museums.\n' \
  | npx nachos-embeddings rerank --query "How many people live in Berlin?" --stdin --json
# [0.931...,0.28...]
```

`serve` can answer rerank requests on the same connection as embeddings. Pass `--rerank-model <id>` and/or `--rerank-dtype <type>` (either enables it; the reranker loads before the ready line). The ready line then carries `"rerank": {"model": "<id>"}`, which clients can use to detect support; without the flag the key is absent and no reranker is downloaded or loaded.

```text
> {"id": 3, "op": "rerank", "query": "capital of France", "passages": ["Paris is in France.", "Cats purr."]}
< {"id": 3, "scores": [0.97, 0.02]}
```

Requests without `"op"` behave exactly as before. A rerank request when the reranker is not enabled, an unknown `op`, or a missing/non-string `query` or `passages` gets `{"id", "error"}`.

## CLI

For callers that are not JavaScript. All diagnostics go to stderr; stdout carries only data.

```bash
npx nachos-embeddings embed --stdin --json < texts.txt   # one text per line in, one JSON vector per line out
npx nachos-embeddings serve                              # persistent JSON-lines server
npx nachos-embeddings info [--json]                      # provider, model, dimension, cache dir
npx nachos-embeddings rerank --query <q> --stdin --json  # one passage per line in, one JSON array of scores out
npx nachos-embeddings pull <model> [--dtype q8]          # pre-download a model for offline use
npx nachos-embeddings models [--json]                    # built-in profiles and cached models
npx nachos-embeddings info <model> [--dtype q8]          # describe a model id or local directory (no download)
```

| Flag                                          | Applies to   | Description                                                                                                                   |
| --------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `--version`                                   | alone        | Print the package version and exit 0 (clients use it to detect the feature set; releases before 0.7.0 have no such flag).     |
| `--stdin`, `--json`                           | `embed`      | Both required. Output order matches input order. Empty input exits 0 with no output.                                          |
| `--provider <name>`                           | all          | `transformers` (default) or `bedrock`. Anything else is a usage error.                                                        |
| `--model <id>`                                | all          | Transformers model name, or Bedrock model id.                                                                                 |
| `--region <region>`                           | bedrock      | AWS region (default `us-east-1`).                                                                                             |
| `--cache-dir <dir>`                           | transformers | Model cache directory (default: per-user, see above).                                                                         |
| `--dtype <type>`                              | transformers | Precision, e.g. `q8`. Appended to the reported model id.                                                                      |
| `--type <t>`                                  | `embed`      | `query` or `document` (default). Selects the model's prefix. In `serve`, see below.                                           |
| `--pooling <mode>`                            | transformers | `mean` or `cls`. Default per model.                                                                                           |
| `--query-prefix <s>`, `--document-prefix <s>` | transformers | Override the model's prefixes (`""` disables).                                                                                |
| `--max-length <n>`                            | transformers | Truncate to `n` tokens (can only lower the tokenizer limit).                                                                  |
| `--query <q>`                                 | `rerank`     | The query; also needs `--stdin --json`. Uses `--model`, `--dtype`, `--cache-dir`.                                             |
| `--rerank-model <id>`, `--rerank-dtype <t>`   | `serve`      | Enable `op: "rerank"` (see Reranking). Also accepted by `rerank`.                                                             |
| `--rerank-max-length <n>`                     | `serve`      | The reranker's truncation length. `--max-length` stays with the embedding model; in `rerank` this defaults to `--max-length`. |
| `--normalize`, `--no-normalize`               | transformers | L2-normalize vectors (default) or return raw pooled vectors.                                                                  |
| `--offline`                                   | transformers | Never use the network; fail if a model is not cached. Env: `NACHOS_EMBEDDINGS_OFFLINE=1`.                                     |
| `<model>`, `--kind embedding\|rerank`         | `pull`       | Download a model into the cache (see Privacy and offline use).                                                                |

**`serve`** loads the provider once, then handles one request per stdin line, sequentially and in order. First stdout line:

```json
{
  "ready": true,
  "version": "0.7.0",
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

## Migrating from 0.6.x

- **Vectors changed for e5, bge and nomic models. Re-index stores built with them.** Documents and queries now carry each model's trained prefix, bge uses cls pooling, and nomic gets its layer norm (see [Providers](#providers)). `SemanticSearch` embeds documents as `"document"` and queries as `"query"`. MiniLM (the default) and unknown models are unchanged. Pass `queryPrefix: ""`, `documentPrefix: ""` and `pooling: "mean"` to reproduce 0.6.0 vectors.
- **`serve --max-length` applies to the embedding model only.** The reranker has its own `--rerank-max-length`.
- **`getModelId()` includes vector-changing overrides** (pooling, document prefix, `normalize=false`).
- Loading a cached model makes no network request; ids that look like paths (`./dir`, `/abs/dir`, `~/dir`, `file://`) load as local directories.
- The 0.6.0 CLI has no `--version`; clients can treat that as older than 0.7.0.

See [CHANGELOG.md](CHANGELOG.md) for the full list.

## Development

```bash
npm install
npm run check   # lint, prettier, typecheck, build, tests with coverage, packed-tarball smoke test
```

`npm install` installs git hooks (pre-commit: lint, format, typecheck; pre-push: `npm run check`). Releasing is documented in [RELEASING.md](RELEASING.md).

## License

MIT
