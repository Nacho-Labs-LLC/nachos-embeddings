/**
 * Text embedding using Transformers.js (local, no API needed)
 *
 * This is a backward-compatible wrapper around TransformersProvider.
 */

import type { EmbeddingProvider } from './providers/types.js';
import { TransformersProvider, type TransformersProviderConfig } from './providers/transformers-provider.js';

export type EmbedderConfig = TransformersProviderConfig;

/**
 * Text embedder using local transformer models
 *
 * @example
 * ```typescript
 * const embedder = new Embedder();
 * await embedder.init();
 *
 * const vector = await embedder.embed('Hello world');
 * console.log(vector); // [0.23, -0.15, 0.87, ...] (384 dimensions)
 *
 * const batch = await embedder.embedBatch(['Text 1', 'Text 2', 'Text 3']);
 * console.log(batch.length); // 3
 * ```
 */
export class Embedder implements EmbeddingProvider {
  readonly name = 'transformers';

  private provider: TransformersProvider;

  constructor(config: EmbedderConfig = {}) {
    this.provider = new TransformersProvider(config);
  }

  /** Call before using embed() or embedBatch() */
  async init(): Promise<void> {
    return this.provider.init();
  }

  /** @throws Error if not initialized */
  async embed(text: string): Promise<number[]> {
    return this.provider.embed(text);
  }

  /** More efficient than calling embed() multiple times */
  async embedBatch(texts: string[]): Promise<number[][]> {
    return this.provider.embedBatch(texts);
  }

  /** Returns null if not initialized */
  async getDimension(): Promise<number | null> {
    return this.provider.getDimension();
  }

  isInitialized(): boolean {
    return this.provider.isInitialized();
  }

  getConfig(): Readonly<Required<EmbedderConfig>> {
    return this.provider.getConfig();
  }
}

/** Singleton instance - avoids loading model multiple times */
let globalEmbedder: Embedder | null = null;

export function getGlobalEmbedder(config?: EmbedderConfig): Embedder {
  if (!globalEmbedder) {
    globalEmbedder = new Embedder(config);
  }
  return globalEmbedder;
}

/** For testing */
export function resetGlobalEmbedder(): void {
  globalEmbedder = null;
}
