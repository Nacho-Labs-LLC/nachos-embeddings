/**
 * High-level semantic search API combining embedder + vector store
 */

import { Embedder } from './embedder.js';
import { VectorStore, type SearchResult } from './vector-store.js';
import { writeFile, mkdir } from 'fs/promises';
import { dirname } from 'path';
import { existsSync } from 'fs';

export interface SemanticSearchConfig {
  /**
   * Embedding model to use
   * @default 'Xenova/all-MiniLM-L6-v2'
   */
  model?: string;

  /**
   * Minimum similarity score (0-1)
   * @default 0.7
   */
  minSimilarity?: number;

  /**
   * Cache directory for models
   * @default '.cache/transformers'
   */
  cacheDir?: string;

  /**
   * Enable progress logging
   * @default false
   */
  progressLogging?: boolean;

  /**
   * Enable automatic persistence after every write operation
   * @default false
   */
  autoSave?: boolean;

  /**
   * Path to save/load the search index
   * Required if autoSave is true
   */
  storePath?: string;

  /**
   * Enable exact text deduplication (prevents duplicate content with different IDs)
   * @default true
   */
  deduplication?: boolean;
}

export interface Document<T = unknown> {
  id: string;
  text: string;
  metadata?: T | undefined;
}

/**
 * Semantic search engine with text-to-vector embedding
 *
 * @example
 * ```typescript
 * const search = new SemanticSearch();
 * await search.init();
 *
 * // Add documents
 * await search.addDocument({
 *   id: 'doc1',
 *   text: 'User loves breakfast tacos',
 *   metadata: { kind: 'preference' }
 * });
 *
 * // Search semantically
 * const results = await search.search('What does user like for morning meals?');
 * // Finds "breakfast tacos" even though query doesn't say "breakfast"
 * ```
 */
export class SemanticSearch<T = unknown> {
  private embedder: Embedder;
  private vectorStore: VectorStore<T>;
  private documents = new Map<string, string>(); // id -> original text
  private config: SemanticSearchConfig;

  constructor(config: SemanticSearchConfig = {}) {
    this.config = {
      deduplication: config.deduplication ?? true,
      ...config,
    };

    if (this.config.autoSave && !this.config.storePath) {
      throw new Error('storePath is required when autoSave is enabled');
    }

    this.embedder = new Embedder({
      ...(config.model !== undefined && { model: config.model }),
      ...(config.cacheDir !== undefined && { cacheDir: config.cacheDir }),
      ...(config.progressLogging !== undefined && { progressLogging: config.progressLogging }),
    });
    this.vectorStore = new VectorStore<T>({
      ...(config.minSimilarity !== undefined && { minSimilarity: config.minSimilarity }),
    });
  }

  /**
   * Initialize the search engine (downloads model on first run)
   */
  async init(): Promise<void> {
    await this.embedder.init();
  }

  /**
   * Auto-persist the index if autoSave is enabled
   */
  private async autoPersist(): Promise<void> {
    if (this.config.autoSave && this.config.storePath) {
      const dir = dirname(this.config.storePath);
      if (!existsSync(dir)) {
        await mkdir(dir, { recursive: true });
      }
      await writeFile(this.config.storePath, JSON.stringify(this.export(), null, 2), 'utf-8');
    }
  }

  /**
   * Add a document to the search index
   */
  async addDocument(doc: Document<T>): Promise<void> {
    // Deduplication check
    if (this.config.deduplication) {
      for (const [existingId, existingText] of this.documents.entries()) {
        if (existingText === doc.text) {
          console.warn(
            `[SemanticSearch] Duplicate content detected: "${doc.id}" matches "${existingId}". Skipping.`
          );
          return;
        }
      }
    }

    const vector = await this.embedder.embed(doc.text);
    this.vectorStore.add(doc.id, vector, doc.metadata);
    this.documents.set(doc.id, doc.text);
    await this.autoPersist();
  }

  /**
   * Add multiple documents in batch (more efficient)
   */
  async addDocuments(docs: Document<T>[]): Promise<void> {
    // Apply deduplication if enabled
    let filteredDocs = docs;
    if (this.config.deduplication) {
      const seenTexts = new Set<string>(this.documents.values());
      filteredDocs = docs.filter((doc) => {
        if (seenTexts.has(doc.text)) {
          console.warn(
            `[SemanticSearch] Duplicate content detected in batch: "${doc.id}". Skipping.`
          );
          return false;
        }
        seenTexts.add(doc.text);
        return true;
      });
    }

    const texts = filteredDocs.map((d) => d.text);
    const vectors = await this.embedder.embedBatch(texts);

    for (let i = 0; i < filteredDocs.length; i++) {
      const doc = filteredDocs[i];
      const vector = vectors[i];
      if (!doc || !vector) continue;
      this.vectorStore.add(doc.id, vector, doc.metadata);
      this.documents.set(doc.id, doc.text);
    }

    await this.autoPersist();
  }

  /**
   * Semantic search for similar documents
   */
  async search(
    query: string,
    options?: {
      limit?: number;
      minSimilarity?: number;
      filter?: (metadata?: T) => boolean;
    }
  ): Promise<Array<SearchResult<T> & { text: string }>> {
    const queryVector = await this.embedder.embed(query);
    const results = this.vectorStore.search(queryVector, options);

    // Add original text to results
    return results.map((result) => ({
      ...result,
      text: this.documents.get(result.id) ?? '',
    }));
  }

  /**
   * Remove a document by ID
   */
  async remove(id: string): Promise<boolean> {
    this.documents.delete(id);
    const removed = this.vectorStore.remove(id);
    if (removed) {
      await this.autoPersist();
    }
    return removed;
  }

  /**
   * Clear all documents
   */
  async clear(): Promise<void> {
    this.vectorStore.clear();
    this.documents.clear();
    await this.autoPersist();
  }

  /**
   * Get number of documents indexed
   */
  size(): number {
    return this.vectorStore.size();
  }

  /**
   * Check if initialized
   */
  isInitialized(): boolean {
    return this.embedder.isInitialized();
  }

  /**
   * Export all documents and vectors (for persistence)
   */
  export(): Array<{
    id: string;
    text: string;
    vector: number[];
    metadata?: T | undefined;
  }> {
    const vectors = this.vectorStore.export();
    return vectors.map((v) => ({
      id: v.id,
      text: this.documents.get(v.id) ?? '',
      vector: v.vector,
      metadata: v.metadata,
    }));
  }

  /**
   * Import documents and vectors (from persistence)
   */
  import(
    data: Array<{
      id: string;
      text: string;
      vector: number[];
      metadata?: T | undefined;
    }>
  ): void {
    for (const item of data) {
      this.vectorStore.add(item.id, item.vector, item.metadata);
      this.documents.set(item.id, item.text);
    }
  }

  /**
   * Manually persist the index to disk
   * Useful when autoSave is disabled but you want to save at specific points
   */
  async persist(): Promise<void> {
    if (!this.config.storePath) {
      throw new Error('storePath must be configured to use persist()');
    }
    const dir = dirname(this.config.storePath);
    if (!existsSync(dir)) {
      await mkdir(dir, { recursive: true });
    }
    await writeFile(this.config.storePath, JSON.stringify(this.export(), null, 2), 'utf-8');
  }
}
