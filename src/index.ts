/**
 * @nacho-labs/nachos-embeddings - Local, privacy-first vector embeddings and semantic search
 *
 * Standalone package for text embeddings and vector similarity search.
 * Uses Transformers.js for local embedding generation — no API keys, no cloud, no costs.
 *
 * @example
 * ```typescript
 * import { SemanticSearch } from '@nacho-labs/nachos-embeddings';
 *
 * const search = new SemanticSearch();
 * await search.init();
 *
 * await search.addDocument({
 *   id: 'doc1',
 *   text: 'User loves breakfast tacos',
 * });
 *
 * const results = await search.search('morning food preferences');
 * console.log(results); // Finds "breakfast tacos"
 * ```
 */

export { SemanticSearch, type SemanticSearchConfig, type Document } from './semantic-search.js';

export {
  EnhancedSemanticSearch,
  type EnhancedSemanticSearchConfig,
} from './semantic-search-enhanced.js';

export { Embedder, getGlobalEmbedder, resetGlobalEmbedder, type EmbedderConfig } from './embedder.js';

export {
  VectorStore,
  cosineSimilarity,
  normalizeVector,
  type VectorStoreConfig,
  type VectorEntry,
  type SearchResult,
} from './vector-store.js';

export { chunkText, estimateTokens, textSimilarity, normalizeText } from './utils.js';
