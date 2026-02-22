# Using nachos-embeddings with Claude Code

Give Claude Code semantic memory over your project — files, decisions, patterns,
and context that persists across sessions and gets recalled by meaning, not
keywords.

## Why this matters

Claude Code is powerful but stateless between sessions. It can read files, but
it doesn't *remember* what you discussed yesterday or *know* which files are
most relevant to your current question. nachos-embeddings fixes this by adding
a local semantic search layer:

- **Semantic recall** — "How do we handle auth?" finds your auth implementation
  even if no file is literally named "auth"
- **Project memory** — Index decisions, patterns, and context that survive
  across sessions
- **Zero cost, fully local** — No API keys, no cloud, no token spend. Runs on
  your machine with Transformers.js
- **Fast** — Sub-50ms per query after model loads. Searching 1000 documents
  takes ~5ms

## Setup: MCP Server for Claude Code

The integration works through MCP (Model Context Protocol). You create a small
server that wraps nachos-embeddings, and Claude Code calls it as a tool.

### 1. Create the MCP server

```bash
mkdir my-semantic-search && cd my-semantic-search
npm init -y
npm install @nacho-labs/nachos-embeddings @modelcontextprotocol/sdk zod
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
await search.init();

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
claude mcp add --transport stdio semantic-search -- node /absolute/path/to/server.ts
```

Or add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "semantic-search": {
      "type": "stdio",
      "command": "node",
      "args": ["/absolute/path/to/server.ts"]
    }
  }
}
```

### 3. Use it

Once registered, Claude Code gains four new tools it can call:

| Tool | What it does |
|------|-------------|
| `semantic_search` | Find relevant content by meaning |
| `semantic_index` | Add content to the index |
| `semantic_remove` | Remove content by ID |
| `semantic_stats` | Check index size |

Claude Code will automatically use these when relevant. You can also prompt it
directly:

```
> Search my indexed context for how we handle authentication
> Index this decision: we chose JWT over sessions because...
> What do we know about the database schema?
```

## What to index

The power comes from what you put in. Here are high-value patterns:

### Architecture decisions

```
Index this: "ADR-012: We separated embeddings and context-manager into
standalone repos to prove they're market differentiators. Embeddings first
(zero deps), context-manager after decoupling from nachos types/config."
```

### File summaries

```
Index a summary of gateway.ts: "Main entry point for the gateway service.
Initializes NATS connection, loads cheese policy engine, sets up message
routing, and manages context via ContextManager. Key config: nachos.toml."
```

### Patterns and conventions

```
Index this project convention: "All containers use node:22-alpine base,
non-root user, read-only filesystem, dropped capabilities. Network access
is deny-by-default per manifest.json declarations."
```

### Debugging insights

```
Remember this: "NATS connection timeouts are usually caused by the bus
container not being ready. The gateway retries with exponential backoff
but check docker compose health checks first."
```

## How it works under the hood

```
You ask: "How do we handle rate limiting?"
         ↓
Claude Code calls: semantic_search("rate limiting")
         ↓
nachos-embeddings converts query to a 384-dimension vector
         ↓
Cosine similarity search against all indexed vectors
         ↓
Returns: "API rate limiting uses a sliding window counter in Redis..."
         (even though the indexed text never said "rate limiting")
```

The model (`all-MiniLM-L6-v2`) understands meaning, not just keywords:

| Query | Finds |
|-------|-------|
| "rate limiting" | "We throttle API requests using sliding windows" |
| "how to deploy" | "Production runs via docker compose up with..." |
| "error handling pattern" | "We use Result types instead of try/catch for..." |

## Performance

- **Model load**: ~2-5s first time, ~500ms cached
- **Index a document**: ~10-50ms
- **Search**: ~5-10ms for 1000 documents
- **Memory**: ~100MB for model + ~1.5KB per document
- **Storage**: Index persists to `.semantic-store.json`

Everything runs locally. No network calls after the initial model download
(~25MB, cached permanently).
