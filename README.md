# @nacho-labs/nachos-embeddings

Local, privacy-first vector embeddings and semantic search. Runs entirely locally using [Transformers.js](https://huggingface.co/docs/transformers.js) — no API keys, no cloud services, no costs.

## Prerequisites

- **Node.js 18+** (uses ESM and top-level await)
- **Internet connection on first run** to download the embedding model (~25MB, cached permanently after that)

## Features

- **Local embeddings** — No API calls, no costs, no rate limits
- **Semantic search** — Find similar text even with different wording
- **Privacy-first** — Data never leaves your machine
- **Lightweight** — ~25MB model download, then runs offline
- **Fast** — In-memory vector index with cosine similarity
- **Standalone** — Zero framework dependencies, use anywhere

## Installation

```bash
npm install @nacho-labs/nachos-embeddings
```

## Quick start

### Semantic search

```typescript
import { SemanticSearch } from '@nacho-labs/nachos-embeddings';

const search = new SemanticSearch();
await search.init(); // Downloads model on first run (~25MB)

// Add documents
await search.addDocument({
  id: 'pref-1',
  text: 'User loves breakfast tacos',
  metadata: { kind: 'preference' },
});

await search.addDocument({
  id: 'pref-2',
  text: 'User dislikes mushrooms',
  metadata: { kind: 'preference' },
});

// Semantic search — finds results even with different wording
const results = await search.search('What does user like for morning meals?');
// [{ id: 'pref-1', similarity: 0.87, text: 'User loves breakfast tacos', metadata: { kind: 'preference' } }]
```

### Low-level API

For more control, use `Embedder` and `VectorStore` directly:

```typescript
import { Embedder, VectorStore } from '@nacho-labs/nachos-embeddings';

const embedder = new Embedder();
await embedder.init();

// Generate embeddings
const vec1 = await embedder.embed('Hello world');
const vec2 = await embedder.embed('Hi there');
console.log(vec1.length); // 384 (vector dimensions)

// Store and search vectors
const store = new VectorStore();
store.add('doc1', vec1, { content: 'Hello world' });
store.add('doc2', vec2, { content: 'Hi there' });

const queryVec = await embedder.embed('Greetings');
const results = store.search(queryVec, { limit: 5 });
// [{ id: 'doc2', similarity: 0.92, metadata: { content: 'Hi there' } }, ...]
```

## Configuration

### Model selection

```typescript
const search = new SemanticSearch({
  model: 'Xenova/all-MiniLM-L6-v2',     // Default — fast, 384 dimensions
  // model: 'Xenova/all-mpnet-base-v2',  // Higher quality, slower
  cacheDir: '.cache/transformers',
  progressLogging: true,
});
```

### Similarity threshold

```typescript
const search = new SemanticSearch({
  minSimilarity: 0.7, // Default (range: 0-1)
});

// Or override per search
const results = await search.search('query', {
  minSimilarity: 0.8,
  limit: 10,
});
```

### Filtering

```typescript
const results = await search.search('query', {
  filter: (metadata) => metadata?.kind === 'preference',
});
```

## Persistence

Export and import for saving to disk:

```typescript
import { readFile, writeFile } from 'node:fs/promises';

// Export
const data = search.export();
await writeFile('embeddings.json', JSON.stringify(data));

// Import
const saved = JSON.parse(await readFile('embeddings.json', 'utf-8'));
search.import(saved);
```

## Using with Claude Code

Give Claude Code semantic memory over your project — decisions, patterns, and
context that persists across sessions and gets recalled by meaning, not keywords.

### Why this matters

Claude Code can read files, but it doesn't *remember* what you discussed
yesterday or *know* which files are most relevant to your current question.
nachos-embeddings adds a local semantic search layer through
[MCP](https://modelcontextprotocol.io) (Model Context Protocol):

- **Semantic recall** — "How do we handle auth?" finds your auth docs even if
  no file is literally named "auth"
- **Project memory** — Index decisions, patterns, and context that survive
  across sessions
- **Zero token spend** — Search happens locally, not in the LLM context window

### 1. Create the MCP server

```bash
mkdir my-semantic-search && cd my-semantic-search
npm init -y
npm install @nacho-labs/nachos-embeddings @modelcontextprotocol/sdk zod tsx
```

Create `server.ts`:

```typescript
import { McpServer, StdioServerTransport } from '@modelcontextprotocol/sdk/server/index.js';
import { z } from 'zod';
import { SemanticSearch } from '@nacho-labs/nachos-embeddings';
import { readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';

const STORE_PATH = '.semantic-store.json';

// Initialize search engine
const search = new SemanticSearch({ minSimilarity: 0.6 });

try {
  await search.init();
} catch (err) {
  console.error('Failed to load embedding model. Is this the first run? An internet connection is required to download the model (~25MB).', err);
  process.exit(1);
}

// Load persisted index if it exists
if (existsSync(STORE_PATH)) {
  const data = JSON.parse(await readFile(STORE_PATH, 'utf-8'));
  search.import(data);
}

async function persist() {
  await writeFile(STORE_PATH, JSON.stringify(search.export()));
}

// Create MCP server
const server = new McpServer(
  { name: 'semantic-search', version: '1.0.0' },
  { capabilities: { logging: {} } }
);

// Tool: Search for semantically similar content
server.registerTool(
  'semantic_search',
  {
    title: 'Semantic Search',
    description: 'Search indexed documents by meaning. Use this to find relevant context, past decisions, code patterns, or any previously indexed content.',
    inputSchema: z.object({
      query: z.string().describe('Natural language search query'),
      limit: z.number().optional().default(5).describe('Max results to return'),
    }),
  },
  async ({ query, limit }) => {
    const results = await search.search(query, { limit });
    if (results.length === 0) {
      return { content: [{ type: 'text', text: 'No relevant results found.' }] };
    }
    const formatted = results.map((r, i) =>
      `${i + 1}. [${(r.similarity * 100).toFixed(0)}%] ${r.text}${r.metadata ? `\n   metadata: ${JSON.stringify(r.metadata)}` : ''}`
    ).join('\n\n');
    return { content: [{ type: 'text', text: formatted }] };
  }
);

// Tool: Add a document to the index
server.registerTool(
  'semantic_index',
  {
    title: 'Index Document',
    description: 'Add a document to the semantic search index. Use this to remember decisions, patterns, file summaries, or any context worth recalling later.',
    inputSchema: z.object({
      id: z.string().describe('Unique document ID'),
      text: z.string().describe('The text content to index'),
      metadata: z.record(z.string()).optional().describe('Optional key-value metadata'),
    }),
  },
  async ({ id, text, metadata }) => {
    await search.addDocument({ id, text, metadata });
    await persist();
    return { content: [{ type: 'text', text: `Indexed "${id}" (${search.size()} total documents)` }] };
  }
);

// Tool: Remove a document
server.registerTool(
  'semantic_remove',
  {
    title: 'Remove Document',
    description: 'Remove a document from the semantic search index by ID.',
    inputSchema: z.object({
      id: z.string().describe('Document ID to remove'),
    }),
  },
  async ({ id }) => {
    const removed = search.remove(id);
    if (removed) await persist();
    return {
      content: [{ type: 'text', text: removed ? `Removed "${id}"` : `"${id}" not found` }],
    };
  }
);

// Tool: Get index stats
server.registerTool(
  'semantic_stats',
  {
    title: 'Index Stats',
    description: 'Get the number of documents currently in the semantic search index.',
    inputSchema: z.object({}),
  },
  async () => ({
    content: [{ type: 'text', text: `${search.size()} documents indexed` }],
  })
);

// Start
const transport = new StdioServerTransport();
await server.connect(transport);
```

### 2. Register with Claude Code

```bash
claude mcp add --transport stdio semantic-search -- npx tsx /absolute/path/to/server.ts
```

Or add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "semantic-search": {
      "type": "stdio",
      "command": "npx",
      "args": ["tsx", "/absolute/path/to/server.ts"]
    }
  }
}
```

### 3. Use it

Once registered, Claude Code gains four new tools:

| Tool | What it does |
| ------ | ------------- |
| `semantic_search` | Find relevant content by meaning |
| `semantic_index` | Add content to the index |
| `semantic_remove` | Remove content by ID |
| `semantic_stats` | Check index size |

Claude Code will use these automatically when relevant. You can also prompt it:

```text
> Search my indexed context for how we handle authentication
> Index this decision: we chose JWT over sessions because...
> What do we know about the database schema?
```

### What to index

The power comes from what you put in. High-value patterns:

**Architecture decisions** — "ADR-012: We separated embeddings into a standalone
repo to prove they're a market differentiator."

**File summaries** — "gateway.ts: Main entry point. Initializes NATS, loads the
policy engine, sets up routing, manages context via ContextManager."

**Conventions** — "All containers use node:22-alpine, non-root user, read-only
filesystem, dropped capabilities."

**Debugging insights** — "NATS timeouts are usually the bus container not being
ready. Check docker compose health checks first."

### How it works

```text
You ask: "How do we handle rate limiting?"
         |
Claude Code calls: semantic_search("rate limiting")
         |
nachos-embeddings converts query to a 384-dimension vector
         |
Cosine similarity search against all indexed vectors
         |
Returns: "We throttle API requests using sliding windows..."
         (matched by meaning, not keywords)
```

The model understands meaning, not just keywords:

| Query | Finds |
| ------- | ------- |
| "rate limiting" | "We throttle API requests using sliding windows" |
| "how to deploy" | "Production runs via docker compose up with..." |
| "error handling pattern" | "We use Result types instead of try/catch for..." |

## Use cases

- **Give your chatbot memory across sessions** — Index facts, preferences, and decisions for semantic recall
- **Search documents by meaning** — Find relevant content even when the wording is completely different
- **Match user questions to known answers** — Build FAQ systems without keyword engineering
- **Detect near-duplicate content** — Find semantically similar text for deduplication
- **Build "more like this" features** — Recommend similar items based on text similarity

## Performance

| Operation | Time (approx) |
|-----------|---------------|
| Model init (first time) | ~2-5 seconds |
| Model init (cached) | ~500ms |
| Embed single text | ~10-50ms |
| Embed batch (100 texts) | ~500ms-2s |
| Search 1000 vectors | ~5-10ms |

**Memory:**
- Model: ~100MB (loaded once, reused)
- Each vector: ~1.5KB (384 floats)
- 1000 documents: ~1.5MB vectors + original text

The in-memory store works well up to ~10K documents. Beyond that, consider a
dedicated vector database like [Qdrant](https://qdrant.tech) or
[Milvus](https://milvus.io).

## Comparison

| Feature | nachos-embeddings | OpenAI | Pinecone |
|---------|-------------------|--------|----------|
| Cost | Free | ~$0.0001/1k chars | ~$70/month |
| Setup | `npm install` | API key | Account + API |
| Privacy | 100% local | Cloud | Cloud |
| Offline | Yes | No | No |
| Quality | Good (85-90%) | Excellent (95%) | N/A (database) |

## API reference

### SemanticSearch

High-level API combining embedder and vector store.

| Method | Description |
| -------- | ------------- |
| `new SemanticSearch(config?)` | Create instance. Config: `model`, `minSimilarity`, `cacheDir`, `progressLogging` |
| `init()` | Load the embedding model. **Must be called before any other method.** |
| `addDocument(doc)` | Add `{ id, text, metadata? }` to the index |
| `addDocuments(docs)` | Batch add (more efficient for multiple documents) |
| `search(query, opts?)` | Search by meaning. Options: `limit`, `minSimilarity`, `filter` |
| `remove(id)` | Remove a document by ID |
| `clear()` | Remove all documents |
| `size()` | Get document count |
| `export()` | Export all documents and vectors for persistence |
| `import(data)` | Import previously exported data |
| `isInitialized()` | Check if the model is loaded |

### Embedder

Low-level text-to-vector conversion.

| Method | Description |
| -------- | ------------- |
| `new Embedder(config?)` | Create instance. Config: `model`, `cacheDir`, `progressLogging` |
| `init()` | Load the model |
| `embed(text)` | Convert text to a 384-dimension vector |
| `embedBatch(texts)` | Convert multiple texts (batched for efficiency) |
| `getDimension()` | Get vector dimension (384 for default model) |
| `isInitialized()` | Check if ready |
| `getConfig()` | Get current configuration |

### VectorStore

In-memory vector storage and similarity search.

| Method | Description |
| -------- | ------------- |
| `new VectorStore(config?)` | Create instance. Config: `minSimilarity`, `defaultLimit` |
| `add(id, vector, metadata?)` | Store a vector |
| `addBatch(entries)` | Store multiple vectors |
| `search(queryVector, opts?)` | Find similar vectors. Options: `limit`, `minSimilarity`, `filter` |
| `get(id)` | Retrieve a vector by ID |
| `remove(id)` | Remove by ID |
| `clear()` | Remove all |
| `size()` | Get count |
| `keys()` | Get all IDs |
| `export()` / `import(entries)` | Persistence |

### Utilities

| Function | Description |
| ---------- | ------------- |
| `cosineSimilarity(a, b)` | Cosine similarity between two vectors (-1 to 1) |
| `normalizeVector(v)` | Normalize to unit length |
| `getGlobalEmbedder(config?)` | Shared singleton instance (avoids loading model twice) |
| `resetGlobalEmbedder()` | Reset the singleton (useful for testing) |

## Advanced

### Batch processing

```typescript
const texts = ['Document 1', 'Document 2', /* ... */];
const embeddings = await embedder.embedBatch(texts);
store.addBatch(texts.map((text, i) => ({
  id: `doc-${i}`,
  vector: embeddings[i],
  metadata: { text },
})));
```

### Global singleton

```typescript
import { getGlobalEmbedder } from '@nacho-labs/nachos-embeddings';

const embedder = getGlobalEmbedder();
await embedder.init(); // Only loads model once
```

### Vector utilities

```typescript
import { cosineSimilarity, normalizeVector } from '@nacho-labs/nachos-embeddings';

const similarity = cosineSimilarity([0.5, 0.3, 0.8], [0.6, 0.4, 0.7]);
const normalized = normalizeVector([3, 4]); // Unit vector
```

## Troubleshooting

### Model download fails

The embedding model (~25MB) is downloaded on first run and cached at
`.cache/transformers/`. If the download fails:

- Check your internet connection
- Check disk space
- Try setting a custom cache directory: `new SemanticSearch({ cacheDir: '/tmp/models' })`

### "Embedder not initialized" error

You must call `init()` before `embed()`, `embedBatch()`, `search()`, or
`addDocument()`. This is an async operation that loads the model:

```typescript
const search = new SemanticSearch();
await search.init(); // Don't forget this
```

### Out of memory with large batches

Reduce batch size by processing in chunks:

```typescript
const CHUNK = 100;
for (let i = 0; i < texts.length; i += CHUNK) {
  const batch = texts.slice(i, i + CHUNK);
  const vecs = await embedder.embedBatch(batch);
  store.addBatch(batch.map((text, j) => ({
    id: `doc-${i + j}`,
    vector: vecs[j],
    metadata: { text },
  })));
}
```

## Development

```bash
npm install
npm run build
npm test
npm run typecheck
```

## License

MIT
