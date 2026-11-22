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

export {
  SemanticSearch,
  type SemanticSearchConfig,
  type SemanticSearchOptions,
  type SemanticSearchLogger,
  type SemanticSearchResult,
  type ResolvedSemanticSearchConfig,
  type DeduplicateOption,
  type SearchOptions,
  StoreCorruptError,
  type Document,
  type DocumentSummary,
} from "./semantic-search.js";

export {
  EnhancedSemanticSearch,
  type EnhancedSemanticSearchConfig,
} from "./semantic-search-enhanced.js";

export { Embedder, type EmbedderConfig } from "./embedder.js";

export {
  VectorStore,
  cosineSimilarity,
  normalizeVector,
  type VectorStoreConfig,
  type VectorEntry,
  type SearchResult,
} from "./vector-store.js";

export { chunkText, estimateTokens, normalizeText } from "./utils.js";

// Provider system
export type {
  EmbeddingProvider,
  BaseProviderConfig,
  EmbedOptions,
} from "./providers/types.js";
export {
  getModelProfile,
  POOLING_MODES,
  INPUT_TYPES,
  type PoolingMode,
  type InputType,
  type ModelProfile,
} from "./providers/model-profiles.js";
export {
  TransformersProvider,
  TRANSFORMERS_DEFAULT_MODEL,
  TRANSFORMERS_DEFAULT_DIMENSION,
  TRANSFORMERS_DTYPES,
  resolveDefaultCacheDir,
  type TransformersProviderConfig,
  type TransformersDtype,
  type ResolvedTransformersConfig,
  type CacheDirEnvironment,
} from "./providers/transformers-provider.js";

// Cross-encoder reranker
export {
  Reranker,
  rerank,
  disposeRerankers,
  RERANK_DEFAULT_MODEL,
  RERANK_FALLBACK_MODEL,
  RERANK_DEFAULT_DTYPE,
  RERANK_DEFAULT_MAX_LENGTH,
  type RerankOptions,
  type ResolvedRerankOptions,
} from "./reranker.js";

// Offline mode and reranker profiles
export { OFFLINE_ENV_VAR, isOfflineEnv } from "./offline.js";
export { ModelNotCachedError } from "./providers/model-loading.js";
export {
  getRerankerProfile,
  type RerankerProfile,
} from "./providers/reranker-profiles.js";

// Model catalogue and inspection
export {
  MODEL_CATALOG,
  findCatalogEntry,
  profileFor,
  type CatalogEntry,
  type EffectiveProfile,
  type ModelKind,
} from "./providers/model-catalog.js";
export {
  listCachedModels,
  inspectModel,
  type CachedModel,
  type ModelInfo,
} from "./model-cache.js";
export {
  ModelLoadError,
  isLocalModelPath,
  resolveModelRef,
  type ModelErrorCode,
} from "./providers/model-loading.js";
