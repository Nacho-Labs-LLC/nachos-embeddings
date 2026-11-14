import {
  SemanticSearch,
  type SemanticSearchOptions,
} from "./semantic-search.js";
import { LEGACY_DEFAULT_STORE_PATH } from "./defaults.js";

/**
 * @deprecated Use SemanticSearchOptions. Every enhanced option (autoSave,
 * storePath, autoChunk, maxChunkTokens, chunkOverlap, temporalBoost, verbose,
 * deduplicateExact, deduplicateSimilarity) is now accepted by SemanticSearch.
 */
export type EnhancedSemanticSearchConfig = SemanticSearchOptions;

interface DocumentMetadata {
  timestamp?: number;
  chunkIndex?: number;
  parentId?: string;
  [key: string]: unknown;
}

/**
 * @deprecated Use SemanticSearch, which now includes all enhanced features.
 *
 * Compatibility alias: it differs from SemanticSearch only in the legacy
 * defaults it applies, namely `deduplicateExact: true` (normalized-text
 * dedup) and the `.semantic-store.json` storePath when autoSave is on
 * without a path.
 */
export class EnhancedSemanticSearch<
  T extends DocumentMetadata = DocumentMetadata,
> extends SemanticSearch<T> {
  constructor(config: EnhancedSemanticSearchConfig = {}) {
    super({
      ...config,
      deduplicateExact: config.deduplicateExact ?? true,
      ...(config.autoSave &&
        !config.storePath && { storePath: LEGACY_DEFAULT_STORE_PATH }),
    });
  }

  /** @deprecated Use persist(). A no-op when no storePath is configured. */
  async forceSave(): Promise<void> {
    if (this.getConfig().storePath) {
      await this.persist();
    }
  }
}
