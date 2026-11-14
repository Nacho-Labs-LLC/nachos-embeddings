/**
 * Single source of truth for search-layer defaults.
 *
 * Model name and cache directory defaults belong to the embedding provider
 * (see Embedder / TransformersProvider) and are intentionally not duplicated
 * here: the search layer only forwards them when the caller sets them.
 */

/** Minimum cosine similarity for a result to be returned. */
export const DEFAULT_MIN_SIMILARITY = 0.4;

/** Default maximum number of search results. */
export const DEFAULT_LIMIT = 10;

/** Chunk size (estimated tokens) used by autoChunk. */
export const DEFAULT_MAX_CHUNK_TOKENS = 500;

/** Overlap (estimated tokens) between chunks used by autoChunk. */
export const DEFAULT_CHUNK_OVERLAP = 50;

/** Store path used only by the deprecated EnhancedSemanticSearch alias. */
export const LEGACY_DEFAULT_STORE_PATH = ".semantic-store.json";
