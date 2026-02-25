import { SemanticSearch, type SemanticSearchConfig, type Document } from './semantic-search.js';
import { type SearchResult } from './vector-store.js';
import { chunkText, normalizeText, estimateTokens } from './utils.js';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname } from 'node:path';

export interface EnhancedSemanticSearchConfig extends SemanticSearchConfig {
  autoSave?: boolean;
  storePath?: string;
  autoChunk?: boolean;
  maxChunkTokens?: number;
  chunkOverlap?: number;
  deduplicateExact?: boolean;
  deduplicateSimilarity?: number;
  temporalBoost?: boolean;
  verbose?: boolean;
}

interface DocumentMetadata {
  timestamp?: number;
  chunkIndex?: number;
  parentId?: string;
  [key: string]: any;
}

export class EnhancedSemanticSearch<T extends DocumentMetadata = DocumentMetadata> extends SemanticSearch<T> {
  private config: Required<EnhancedSemanticSearchConfig>;
  private saveQueue = Promise.resolve();
  private textHashes = new Map<string, string>();

  constructor(config: EnhancedSemanticSearchConfig = {}) {
    super(config);
    
    this.config = {
      ...config,
      model: config.model ?? 'Xenova/all-MiniLM-L6-v2',
      minSimilarity: config.minSimilarity ?? 0.7,
      cacheDir: config.cacheDir ?? '.cache/transformers',
      progressLogging: config.progressLogging ?? false,
      autoSave: config.autoSave ?? false,
      storePath: config.storePath ?? '.semantic-store.json',
      autoChunk: config.autoChunk ?? false,
      maxChunkTokens: config.maxChunkTokens ?? 500,
      chunkOverlap: config.chunkOverlap ?? 50,
      deduplicateExact: config.deduplicateExact ?? true,
      deduplicateSimilarity: config.deduplicateSimilarity ?? 0,
      temporalBoost: config.temporalBoost ?? false,
      verbose: config.verbose ?? false,
    };
  }

  override async init(): Promise<void> {
    await super.init();
    await this.load();
  }

  async load(): Promise<void> {
    if (!this.config.storePath || !existsSync(this.config.storePath)) {
      return;
    }

    try {
      const raw = await readFile(this.config.storePath, 'utf-8');
      const data = JSON.parse(raw);

      if (Array.isArray(data)) {
        super.import(data);

        for (const item of data) {
          const normalized = normalizeText(item.text);
          this.textHashes.set(normalized, item.id);
        }

        if (this.config.verbose) {
          console.log(`[EnhancedSemanticSearch] Loaded ${data.length} documents from ${this.config.storePath}`);
        }
      }
    } catch (err) {
      if (this.config.verbose) {
        console.warn(`[EnhancedSemanticSearch] Failed to load from ${this.config.storePath}:`, err);
      }
    }
  }

  private async save(): Promise<void> {
    if (!this.config.autoSave || !this.config.storePath) {
      return;
    }

    this.saveQueue = this.saveQueue.then(async () => {
      try {
        const dir = dirname(this.config.storePath!);
        if (!existsSync(dir)) {
          await mkdir(dir, { recursive: true });
        }

        const data = super.export();
        await writeFile(this.config.storePath!, JSON.stringify(data, null, 2));

        if (this.config.verbose) {
          console.log(`[EnhancedSemanticSearch] Saved ${data.length} documents to ${this.config.storePath}`);
        }
      } catch (err) {
        console.error('[EnhancedSemanticSearch] Save failed:', err);
      }
    });

    return this.saveQueue;
  }

  private async checkExactDuplicate(text: string): Promise<string | null> {
    if (!this.config.deduplicateExact) {
      return null;
    }

    const normalized = normalizeText(text);
    return this.textHashes.get(normalized) ?? null;
  }

  private async checkFuzzyDuplicate(text: string): Promise<{ id: string; similarity: number } | null> {
    if (this.config.deduplicateSimilarity === 0) {
      return null;
    }

    const results = await super.search(text, {
      limit: 1,
      minSimilarity: this.config.deduplicateSimilarity,
    });

    return results.length > 0 ? { id: results[0]!.id, similarity: results[0]!.similarity } : null;
  }

  override async addDocument(doc: Document<T>): Promise<void> {
    const exactDup = await this.checkExactDuplicate(doc.text);
    if (exactDup) {
      if (this.config.verbose) {
        console.log(`[EnhancedSemanticSearch] Skipped exact duplicate: "${doc.id}" matches "${exactDup}"`);
      }
      return;
    }

    const fuzzyDup = await this.checkFuzzyDuplicate(doc.text);
    if (fuzzyDup) {
      if (this.config.verbose) {
        console.log(
          `[EnhancedSemanticSearch] Skipped fuzzy duplicate: "${doc.id}" ~= "${fuzzyDup.id}" (${(fuzzyDup.similarity * 100).toFixed(1)}%)`
        );
      }
      return;
    }

    const metadata = { ...doc.metadata } as T;
    if (this.config.temporalBoost && !metadata.timestamp) {
      metadata.timestamp = Date.now();
    }

    const tokenCount = estimateTokens(doc.text);
    if (this.config.autoChunk && tokenCount > this.config.maxChunkTokens) {
      await this.addLongDocument({ ...doc, metadata });
    } else {
      await super.addDocument({ ...doc, metadata });
      
      const normalized = normalizeText(doc.text);
      this.textHashes.set(normalized, doc.id);
    }

    await this.save();
  }

  private async addLongDocument(doc: Document<T>): Promise<void> {
    const chunks = chunkText(doc.text, {
      maxTokens: this.config.maxChunkTokens,
      overlapTokens: this.config.chunkOverlap,
    });

    if (this.config.verbose) {
      console.log(`[EnhancedSemanticSearch] Chunking "${doc.id}" into ${chunks.length} parts`);
    }

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i]!;
      const chunkId = chunks.length === 1 ? doc.id : `${doc.id}#chunk${i}`;
      
      const chunkMetadata = {
        ...doc.metadata,
        chunkIndex: i,
        parentId: doc.id,
      } as T;

      await super.addDocument({
        id: chunkId,
        text: chunk,
        metadata: chunkMetadata,
      });

      const normalized = normalizeText(chunk);
      this.textHashes.set(normalized, chunkId);
    }
  }

  override async addDocuments(docs: Document<T>[]): Promise<void> {
    for (const doc of docs) {
      await this.addDocument(doc);
    }
  }

  override async search(
    query: string,
    options?: {
      limit?: number;
      minSimilarity?: number;
      filter?: (metadata?: T) => boolean;
      temporalBoost?: boolean;
    }
  ): Promise<Array<SearchResult<T> & { text: string }>> {
    let results = await super.search(query, options);

    const shouldBoost = options?.temporalBoost ?? this.config.temporalBoost;
    if (shouldBoost) {
      const now = Date.now();
      
      results = results.map((r) => {
        const timestamp = r.metadata?.timestamp ?? now;
        const ageMs = now - timestamp;
        const ageDays = ageMs / (1000 * 60 * 60 * 24);

        // Exponential decay: after 30 days boost = 0.5, after 365 days = ~0.2
        const recencyBoost = 1 / (1 + Math.log(1 + ageDays));

        // 70% semantic, 30% recency
        const boostedSimilarity = r.similarity * (0.7 + 0.3 * recencyBoost);

        return {
          ...r,
          similarity: boostedSimilarity,
        };
      });

      results.sort((a, b) => b.similarity - a.similarity);

      if (options?.limit) {
        results = results.slice(0, options.limit);
      }
    }

    return results;
  }

  override remove(id: string): boolean {
    const exportedDocs = super.export();
    const doc = exportedDocs.find((d) => d.id === id);
    if (doc) {
      const normalized = normalizeText(doc.text);
      this.textHashes.delete(normalized);
    }

    const removed = super.remove(id);
    
    if (removed) {
      this.save().catch((err) => {
        console.error('[EnhancedSemanticSearch] Save after remove failed:', err);
      });
    }

    return removed;
  }

  override clear(): void {
    super.clear();
    this.textHashes.clear();
    this.save().catch((err) => {
      console.error('[EnhancedSemanticSearch] Save after clear failed:', err);
    });
  }

  async forceSave(): Promise<void> {
    await this.save();
  }

  getConfig(): Readonly<Required<EnhancedSemanticSearchConfig>> {
    return { ...this.config };
  }
}
