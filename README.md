# @nacho-labs/nachos-embeddings

Local, privacy-first vector embeddings and semantic search. Runs entirely locally using [Transformers.js](https://huggingface.co/docs/transformers.js) — no API keys, no cloud services, no costs.

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

## Quick Start

### Semantic Search

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

### Low-Level API

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

### Model Selection

```typescript
const search = new SemanticSearch({
  model: 'Xenova/all-MiniLM-L6-v2',     // Default — fast, 384 dimensions
  // model: 'Xenova/all-mpnet-base-v2',  // Higher quality, slower
  cacheDir: '.cache/transformers',
  progressLogging: true,
});
```

### Similarity Threshold

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
// Export
const data = search.export();
await fs.writeFile('embeddings.json', JSON.stringify(data));

// Import
const saved = JSON.parse(await fs.readFile('embeddings.json', 'utf-8'));
search.import(saved);
```

## Use Cases

- **AI memory** — Semantic recall for chatbot/assistant context
- **Document search** — Find similar documents by meaning, not keywords
- **FAQ matching** — Match user questions to known answers
- **Deduplication** — Find near-duplicate content
- **Recommendation** — "More like this" based on text similarity

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

## Comparison

| Feature | nachos-embeddings | OpenAI | Pinecone |
|---------|-------------------|--------|----------|
| Cost | Free | ~$0.0001/1k chars | ~$70/month |
| Setup | `npm install` | API key | Account + API |
| Privacy | 100% local | Cloud | Cloud |
| Offline | Yes | No | No |
| Quality | Good (85-90%) | Excellent (95%) | N/A (database) |

## Advanced

### Vector Utilities

```typescript
import { cosineSimilarity, normalizeVector } from '@nacho-labs/nachos-embeddings';

const similarity = cosineSimilarity([0.5, 0.3, 0.8], [0.6, 0.4, 0.7]);
const normalized = normalizeVector([3, 4]); // Unit vector
```

### Global Singleton

```typescript
import { getGlobalEmbedder } from '@nacho-labs/nachos-embeddings';

const embedder = getGlobalEmbedder();
await embedder.init(); // Only loads model once
```

### Batch Processing

```typescript
const texts = ['Document 1', 'Document 2', /* ... */];
const embeddings = await embedder.embedBatch(texts);
store.addBatch(texts.map((text, i) => ({
  id: `doc-${i}`,
  vector: embeddings[i],
  metadata: { text },
})));
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
