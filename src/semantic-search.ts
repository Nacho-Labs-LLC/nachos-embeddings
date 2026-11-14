/**
 * High-level semantic search API combining embedder + vector store.
 *
 * This is the single search class: chunking, deduplication, temporal boost
 * and safe persistence are all options on SemanticSearch.
 */

import type { EmbeddingProvider } from "./providers/types.js";
import { Embedder } from "./embedder.js";
import { VectorStore, type SearchResult } from "./vector-store.js";
import {
  DEFAULT_CHUNK_OVERLAP,
  DEFAULT_LIMIT,
  DEFAULT_MAX_CHUNK_TOKENS,
} from "./defaults.js";
import { chunkText, estimateTokens, normalizeText } from "./utils.js";
import {
  StoreCorruptError,
  quarantineStoreFile,
  readStoreFile,
  writeStoreFileAtomic,
  type StoredDocument,
} from "./store-file.js";

export { StoreCorruptError } from "./store-file.js";

/** Optional logger. Library output is silent unless one is provided. */
export interface SemanticSearchLogger {
  debug(message: string): void;
  info(message: string): void;
  warn(message: string): void;
  error(message: string): void;
}

/**
 * Duplicate handling:
 * - `'off'`: never skip anything.
 * - `'exact'`: skip when the text is byte-for-byte identical.
 * - `'normalized'`: skip when text matches after lower-casing and collapsing whitespace.
 * - `{ similarity }`: normalized match, plus skip when an existing document's
 *   embedding has cosine similarity >= `similarity` (0-1] with the new one.
 *   (Not applied to the pieces of auto-chunked documents.)
 */
export type DeduplicateOption =
  "off" | "exact" | "normalized" | { similarity: number };

export interface SemanticSearchConfig {
  /** Embedding model (forwarded to the default Embedder) */
  model?: string | undefined;

  /**
   * Minimum similarity score (0-1)
   * @default 0.4
   */
  minSimilarity?: number | undefined;

  /** Cache directory for models (forwarded to the default Embedder) */
  cacheDir?: string | undefined;

  /** Enable model-download progress logging (forwarded to the default Embedder) */
  progressLogging?: boolean | undefined;

  /**
   * Persist after every write operation. Requires `storePath`.
   * Writes go through one serialized queue and are atomic.
   * @default false
   */
  autoSave?: boolean | undefined;

  /** Path of the JSON store used by autoSave, persist() and load-on-init */
  storePath?: string | undefined;

  /**
   * Load `storePath` during init() (and before the first write if init was
   * skipped), so a write can never clobber an unread store.
   * @default true
   */
  loadOnInit?: boolean | undefined;

  /**
   * What to do when the store file exists but is corrupt.
   * `'throw'` rejects with StoreCorruptError and leaves the file alone.
   * `'quarantine'` renames it to `<file>.corrupt-<timestamp>`, logs a warning
   * and starts empty.
   * @default 'throw'
   */
  onCorruptStore?: "throw" | "quarantine" | undefined;

  /**
   * Duplicate handling. Takes precedence over the deprecated keys below.
   * @default 'exact'
   */
  deduplicate?: DeduplicateOption | undefined;

  /** @deprecated Use `deduplicate`. `false` maps to `'off'`, `true` to `'exact'`. */
  deduplication?: boolean | undefined;

  /** @deprecated Use `deduplicate: 'normalized'`. */
  deduplicateExact?: boolean | undefined;

  /** @deprecated Use `deduplicate: { similarity }`. 0 disables. */
  deduplicateSimilarity?: number | undefined;

  /**
   * Split documents longer than `maxChunkTokens` into overlapping chunks with
   * ids `<id>#chunk<n>` and metadata `chunkIndex` / `parentId`.
   * @default false
   */
  autoChunk?: boolean | undefined;

  /** @default 500 (estimated tokens) */
  maxChunkTokens?: number | undefined;

  /** @default 50 (estimated tokens) */
  chunkOverlap?: number | undefined;

  /**
   * Stamp new documents with `metadata.timestamp` and re-rank searches to
   * favour recent ones. Can be overridden per search.
   * @default false
   */
  temporalBoost?: boolean | undefined;

  /** Logger for debug/info/warn/error output. Silent by default. */
  logger?: Partial<SemanticSearchLogger> | undefined;

  /** Shorthand: when no logger is given, log to the console. */
  verbose?: boolean | undefined;

  /**
   * Called when an autoSave write fails. The error is also logged and is
   * rethrown by the next explicit persist()/forceSave().
   */
  onError?: ((error: Error) => void) | undefined;

  /**
   * @deprecated Never had any effect on the search layer.
   */
  cacheEmbeddings?: boolean | undefined;
}

export interface SemanticSearchOptions extends SemanticSearchConfig {
  /**
   * Custom embedding provider (overrides model/cacheDir/progressLogging)
   */
  provider?: EmbeddingProvider | undefined;
}

export interface Document<T = unknown> {
  id: string;
  text: string;
  metadata?: T | undefined;
}

/** A document reference that never includes document text or an embedding vector. */
export interface DocumentSummary<T = unknown> {
  id: string;
  metadata?: T | undefined;
}

export interface SearchOptions<T = unknown> {
  limit?: number;
  minSimilarity?: number;
  filter?: (metadata?: T) => boolean;
  /** Override the instance-level temporalBoost for this search */
  temporalBoost?: boolean;
}

export type SemanticSearchResult<T = unknown> = SearchResult<T> & {
  text: string;
};

/** Resolved (defaults applied) configuration, as returned by getConfig(). */
export interface ResolvedSemanticSearchConfig {
  minSimilarity: number | undefined;
  autoSave: boolean;
  storePath: string | undefined;
  loadOnInit: boolean;
  onCorruptStore: "throw" | "quarantine";
  deduplicate: DeduplicateOption;
  autoChunk: boolean;
  maxChunkTokens: number;
  chunkOverlap: number;
  temporalBoost: boolean;
}

/**
 * Maps new and legacy dedup options to the single `deduplicate` shape.
 * Precedence: `deduplicate` > deduplicateSimilarity > deduplicateExact >
 * deduplication > default ('exact').
 */
export function resolveDeduplicate(
  config: Pick<
    SemanticSearchConfig,
    | "deduplicate"
    | "deduplication"
    | "deduplicateExact"
    | "deduplicateSimilarity"
  >,
): DeduplicateOption {
  if (config.deduplicate !== undefined) return config.deduplicate;
  if (
    config.deduplicateSimilarity !== undefined &&
    config.deduplicateSimilarity > 0
  ) {
    return { similarity: config.deduplicateSimilarity };
  }
  if (config.deduplicateExact === true) return "normalized";
  return config.deduplication === false ? "off" : "exact";
}

function consoleLogger(): SemanticSearchLogger {
  return {
    debug: (m) => console.log(m),
    info: (m) => console.log(m),
    warn: (m) => console.warn(m),
    error: (m) => console.error(m),
  };
}

function buildLogger(config: SemanticSearchConfig): SemanticSearchLogger {
  const base: Partial<SemanticSearchLogger> =
    config.logger ?? (config.verbose ? consoleLogger() : {});
  const noop = (): void => undefined;
  return {
    debug: base.debug?.bind(base) ?? noop,
    info: base.info?.bind(base) ?? noop,
    warn: base.warn?.bind(base) ?? noop,
    error: base.error?.bind(base) ?? noop,
  };
}

function toError(err: unknown): Error {
  return err instanceof Error ? err : new Error(String(err));
}

/** A unit that will be embedded and stored (a document, or one chunk of one). */
interface Unit<T> {
  id: string;
  text: string;
  metadata: T | undefined;
  /** Set on the pieces of a multi-chunk document */
  parentId?: string;
  /** Set on the first piece: whole-document dedup key owned by the parent */
  wholeKey?: string | undefined;
}

/**
 * Semantic search engine with text-to-vector embedding
 *
 * @example
 * ```typescript
 * const search = new SemanticSearch({ storePath: 'index.json', autoSave: true });
 * await search.init();
 *
 * await search.addDocument({
 *   id: 'doc1',
 *   text: 'User loves breakfast tacos',
 *   metadata: { kind: 'preference' }
 * });
 *
 * const results = await search.search('What does user like for morning meals?');
 * ```
 */
export class SemanticSearch<T = unknown> {
  private embedder: EmbeddingProvider;
  private vectorStore: VectorStore<T>;
  private documents = new Map<string, string>(); // id -> original text
  /** dedup key -> id that owns it */
  private dedupIndex = new Map<string, string>();
  /** id -> dedup keys to release when the id is removed or replaced */
  private keysById = new Map<string, string[]>();
  /** parent id -> chunk ids */
  private children = new Map<string, string[]>();

  private readonly resolved: ResolvedSemanticSearchConfig;
  private readonly logger: SemanticSearchLogger;
  private readonly onError: ((error: Error) => void) | undefined;

  private mutationTail: Promise<void> = Promise.resolve();
  private saveTail: Promise<void> = Promise.resolve();
  private scheduledSave: Promise<void> | undefined;
  private deferredSaveError: Error | undefined;
  private loaded = false;

  constructor(config: SemanticSearchOptions = {}) {
    if (config.autoSave && !config.storePath) {
      throw new Error("storePath is required when autoSave is enabled");
    }

    this.resolved = {
      minSimilarity: config.minSimilarity,
      autoSave: config.autoSave ?? false,
      storePath: config.storePath,
      loadOnInit: config.loadOnInit ?? true,
      onCorruptStore: config.onCorruptStore ?? "throw",
      deduplicate: resolveDeduplicate(config),
      autoChunk: config.autoChunk ?? false,
      maxChunkTokens: config.maxChunkTokens ?? DEFAULT_MAX_CHUNK_TOKENS,
      chunkOverlap: config.chunkOverlap ?? DEFAULT_CHUNK_OVERLAP,
      temporalBoost: config.temporalBoost ?? false,
    };
    this.logger = buildLogger(config);
    this.onError = config.onError;

    this.embedder =
      config.provider ??
      new Embedder({
        ...(config.model !== undefined && { model: config.model }),
        ...(config.cacheDir !== undefined && { cacheDir: config.cacheDir }),
        ...(config.progressLogging !== undefined && {
          progressLogging: config.progressLogging,
        }),
      });
    this.vectorStore = new VectorStore<T>({
      ...(config.minSimilarity !== undefined && {
        minSimilarity: config.minSimilarity,
      }),
    });
  }

  /** Initializes the embedder and loads the store (when storePath is set). */
  async init(): Promise<void> {
    await this.embedder.init();
    if (this.shouldLoad() && !this.loaded) {
      await this.load();
    }
  }

  getConfig(): Readonly<ResolvedSemanticSearchConfig> {
    return { ...this.resolved };
  }

  // ---------------------------------------------------------------- mutation

  /** Runs fn after all earlier mutations finish, so check-embed-add is atomic. */
  private runExclusive<R>(fn: () => Promise<R>): Promise<R> {
    const run = this.mutationTail.then(fn);
    this.mutationTail = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  }

  async addDocument(doc: Document<T>): Promise<void> {
    await this.addDocuments([doc]);
  }

  /** Batch add (more efficient than individual calls) */
  async addDocuments(docs: Document<T>[]): Promise<void> {
    const changed = await this.runExclusive(() => this.addLocked(docs));
    if (changed) {
      await this.autoPersist();
    }
  }

  private dedupKey(text: string): string | undefined {
    const mode = this.resolved.deduplicate;
    if (mode === "off") return undefined;
    return mode === "exact" ? text : normalizeText(text);
  }

  private fuzzyThreshold(): number | undefined {
    const mode = this.resolved.deduplicate;
    return typeof mode === "object" && mode.similarity > 0
      ? mode.similarity
      : undefined;
  }

  /** Builds embeddable units for the documents that survive exact dedup. */
  private planUnits(docs: Document<T>[]): Unit<T>[] {
    const units: Unit<T>[] = [];
    const seen = new Set<string>();
    const now = Date.now();
    const { autoChunk, maxChunkTokens, chunkOverlap, temporalBoost } =
      this.resolved;

    for (const doc of docs) {
      const key = this.dedupKey(doc.text);
      if (key !== undefined) {
        const existing = this.dedupIndex.get(key);
        if (existing !== undefined && existing !== doc.id) {
          this.logger.debug(
            `Skipped duplicate: "${doc.id}" matches "${existing}"`,
          );
          continue;
        }
        if (existing === doc.id || seen.has(key)) {
          this.logger.debug(`Skipped duplicate: "${doc.id}"`);
          continue;
        }
        seen.add(key);
      }

      let metadata = doc.metadata;
      if (temporalBoost) {
        const meta = { ...(metadata as object | undefined) } as {
          timestamp?: unknown;
        };
        if (!meta.timestamp) meta.timestamp = now;
        metadata = meta as T;
      }

      if (autoChunk && estimateTokens(doc.text) > maxChunkTokens) {
        const chunks = chunkText(doc.text, {
          maxTokens: maxChunkTokens,
          overlapTokens: chunkOverlap,
        });
        this.logger.debug(`Chunking "${doc.id}" into ${chunks.length} parts`);
        const multi = chunks.length > 1;
        chunks.forEach((chunk, i) => {
          units.push({
            id: multi ? `${doc.id}#chunk${i}` : doc.id,
            text: chunk,
            metadata: {
              ...(metadata as object | undefined),
              chunkIndex: i,
              parentId: doc.id,
            } as T,
            ...(multi && { parentId: doc.id }),
            ...(multi && i === 0 && { wholeKey: key }),
          });
        });
      } else {
        units.push({ id: doc.id, text: doc.text, metadata });
      }
    }
    return units;
  }

  private registerKey(id: string, key: string): void {
    this.dedupIndex.set(key, id);
    const keys = this.keysById.get(id) ?? [];
    keys.push(key);
    this.keysById.set(id, keys);
  }

  private async embedTexts(texts: string[]): Promise<number[][]> {
    if (texts.length === 1) {
      return [await this.embedder.embed(texts[0] as string)];
    }
    return this.embedder.embedBatch(texts);
  }

  private async addLocked(docs: Document<T>[]): Promise<boolean> {
    const units = this.planUnits(docs);
    if (units.length === 0) return false;

    const vectors = await this.embedTexts(units.map((u) => u.text));

    const threshold = this.fuzzyThreshold();
    let added = 0;
    units.forEach((unit, i) => {
      const vector = vectors[i];
      if (!vector) return;
      if (threshold !== undefined && unit.parentId === undefined) {
        const near = this.vectorStore.search(vector, {
          limit: 1,
          minSimilarity: threshold,
        })[0];
        if (near && near.id !== unit.id) {
          this.logger.debug(
            `Skipped fuzzy duplicate: "${unit.id}" ~= "${near.id}" (${(near.similarity * 100).toFixed(1)}%)`,
          );
          return;
        }
      }
      this.store(unit, vector);
      added++;
    });
    return added > 0;
  }

  /** Stores one unit, replacing any previous document with the same id. */
  private store(unit: Unit<T>, vector: number[]): void {
    if (unit.parentId !== undefined && unit.wholeKey !== undefined) {
      this.removeOne(unit.parentId); // replacing a chunked document
    }
    if (this.documents.has(unit.id)) {
      this.removeOne(unit.id);
    }
    this.vectorStore.add(unit.id, vector, unit.metadata);
    this.documents.set(unit.id, unit.text);
    const key = this.dedupKey(unit.text);
    if (key !== undefined) this.registerKey(unit.id, key);
    if (unit.parentId !== undefined) {
      if (unit.wholeKey !== undefined) {
        this.registerKey(unit.parentId, unit.wholeKey);
      }
      const list = this.children.get(unit.parentId) ?? [];
      if (!list.includes(unit.id)) list.push(unit.id);
      this.children.set(unit.parentId, list);
    }
  }

  private releaseKeys(id: string): void {
    for (const key of this.keysById.get(id) ?? []) {
      if (this.dedupIndex.get(key) === id) this.dedupIndex.delete(key);
    }
    this.keysById.delete(id);
  }

  /** O(1) for plain documents; chunked parents also remove their chunks. */
  private removeOne(id: string): boolean {
    const parent = this.parentOf(this.vectorStore.get(id)?.metadata, id);
    let removed = this.vectorStore.remove(id);
    this.documents.delete(id);
    this.releaseKeys(id);
    if (parent !== undefined) {
      const siblings = this.children.get(parent);
      const at = siblings?.indexOf(id) ?? -1;
      if (siblings && at !== -1) siblings.splice(at, 1);
    }
    const kids = this.children.get(id);
    if (kids) {
      this.children.delete(id);
      for (const kid of [...kids]) {
        this.removeOne(kid);
      }
      return true;
    }
    return removed;
  }

  async remove(id: string): Promise<boolean> {
    const removed = await this.runExclusive(async () => this.removeOne(id));
    if (removed) {
      await this.autoPersist();
    }
    return removed;
  }

  async clear(): Promise<void> {
    await this.runExclusive(async () => {
      this.vectorStore.clear();
      this.documents.clear();
      this.dedupIndex.clear();
      this.keysById.clear();
      this.children.clear();
    });
    await this.autoPersist();
  }

  // ------------------------------------------------------------------ search

  async search(
    query: string,
    options?: SearchOptions<T>,
  ): Promise<Array<SemanticSearchResult<T>>> {
    const queryVector = await this.embedder.embed(query);
    const boost = options?.temporalBoost ?? this.resolved.temporalBoost;
    const limit = options?.limit ?? DEFAULT_LIMIT;

    // When re-ranking, fetch every candidate so the limit applies after boosting.
    let results = this.vectorStore.search(queryVector, {
      ...options,
      limit: boost ? Infinity : limit,
    });

    if (boost) {
      const now = Date.now();
      results = results
        .map((r) => ({
          ...r,
          similarity: r.similarity * recencyWeight(r.metadata, now),
        }))
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, limit);
    }

    return results.map((result) => ({
      ...result,
      text: this.documents.get(result.id) ?? "",
    }));
  }

  size(): number {
    return this.vectorStore.size();
  }

  isInitialized(): boolean {
    return this.embedder.isInitialized();
  }

  /**
   * Retrieves one stored document without exposing its embedding vector.
   * Returns undefined when the document ID is not present.
   */
  getDocument(id: string): Document<T> | undefined {
    const text = this.documents.get(id);
    if (text === undefined) {
      return undefined;
    }

    return {
      id,
      text,
      metadata: this.vectorStore.get(id)?.metadata,
    };
  }

  /**
   * Lists document IDs and metadata without exposing document text or vectors.
   */
  listDocuments(): DocumentSummary<T>[] {
    return this.vectorStore.keys().map((id) => ({
      id,
      metadata: this.vectorStore.get(id)?.metadata,
    }));
  }

  // ------------------------------------------------------------- import/export

  /** For persistence */
  export(): Array<StoredDocument<T>> {
    return this.vectorStore.export().map((v) => ({
      id: v.id,
      text: this.documents.get(v.id) ?? "",
      vector: v.vector,
      metadata: v.metadata,
    }));
  }

  /** From persistence (does not trigger autoSave) */
  import(data: Array<StoredDocument<T>>): void {
    for (const item of data) {
      const parentId = this.parentOf(item.metadata, item.id);
      this.store(
        {
          id: item.id,
          text: item.text,
          metadata: item.metadata,
          ...(parentId !== undefined && { parentId }),
        },
        item.vector,
      );
    }
  }

  private parentOf(
    metadata: T | undefined,
    selfId: string,
  ): string | undefined {
    const parent = (metadata as { parentId?: unknown } | undefined)?.parentId;
    return typeof parent === "string" && parent !== "" && parent !== selfId
      ? parent
      : undefined;
  }

  // -------------------------------------------------------------- persistence

  private shouldLoad(): boolean {
    return this.resolved.loadOnInit && this.resolved.storePath !== undefined;
  }

  /**
   * Loads the store file into memory (documents already in memory win).
   * A missing file is an empty store. A corrupt file rejects with
   * StoreCorruptError unless `onCorruptStore: 'quarantine'` is set.
   */
  async load(): Promise<void> {
    const path = this.resolved.storePath;
    if (!path) return;

    let data: Array<StoredDocument<T>> | undefined;
    try {
      data = await readStoreFile<T>(path);
    } catch (err) {
      if (
        err instanceof StoreCorruptError &&
        this.resolved.onCorruptStore === "quarantine"
      ) {
        const moved = await quarantineStoreFile(path);
        this.logger.warn(`${err.message}. Moved aside to ${moved}`);
        this.loaded = true;
        return;
      }
      throw err;
    }

    await this.runExclusive(async () => {
      const fresh = (data ?? []).filter((d) => !this.documents.has(d.id));
      this.import(fresh);
    });
    this.loaded = true;
    this.logger.info(`Loaded ${data?.length ?? 0} documents from ${path}`);
  }

  private async autoPersist(): Promise<void> {
    if (!this.resolved.autoSave) return;
    try {
      await this.requestSave();
    } catch (err) {
      const error = toError(err);
      this.deferredSaveError = error;
      this.logger.error(`Auto-save failed: ${error.message}`);
      this.onError?.(error);
    }
  }

  /**
   * Queues a write. Writes are strictly serialized; requests made while one
   * is still waiting to start share it, and each write snapshots the state
   * when it begins, so the final file always reflects the latest mutation.
   */
  private requestSave(): Promise<void> {
    if (this.scheduledSave) return this.scheduledSave;
    const job = this.saveTail
      .catch(() => undefined)
      .then(async () => {
        this.scheduledSave = undefined;
        await this.writeSnapshot();
      });
    this.scheduledSave = job;
    this.saveTail = job;
    return job;
  }

  private async writeSnapshot(): Promise<void> {
    const path = this.resolved.storePath as string;
    if (this.shouldLoad() && !this.loaded) {
      await this.load(); // never overwrite a store we have not read
    }
    const data = this.export();
    await writeStoreFileAtomic(path, data);
    this.logger.debug(`Saved ${data.length} documents to ${path}`);
  }

  /**
   * Writes the store now. Rejects if the write fails, or if an earlier
   * autoSave failed (that error is rethrown once, after this write).
   */
  async persist(): Promise<void> {
    if (!this.resolved.storePath) {
      throw new Error("storePath must be configured to use persist()");
    }
    await this.runExclusive(async () => undefined); // let queued mutations land
    try {
      await this.requestSave();
    } catch (err) {
      this.deferredSaveError = undefined;
      throw err;
    }
    if (this.deferredSaveError) {
      const error = this.deferredSaveError;
      this.deferredSaveError = undefined;
      throw error;
    }
  }
}

/**
 * 70% semantic, 30% recency. Recency decays logarithmically with age:
 * weight 1 for now, ~0.59 at 30 days, ~0.14 at one year.
 */
function recencyWeight(metadata: unknown, now: number): number {
  const ts = (metadata as { timestamp?: unknown } | undefined)?.timestamp;
  const timestamp = typeof ts === "number" ? ts : now;
  const ageDays = Math.max(0, now - timestamp) / (1000 * 60 * 60 * 24);
  const recency = 1 / (1 + Math.log(1 + ageDays));
  return 0.7 + 0.3 * recency;
}
