/**
 * Embedding provider using Transformers.js (local, no API needed)
 */

import { pipeline, env } from '@huggingface/transformers';
import type { EmbeddingProvider, BaseProviderConfig } from './types.js';

export interface TransformersProviderConfig extends BaseProviderConfig {
  /**
   * Model to use for embeddings
   * @default 'Xenova/all-MiniLM-L6-v2'
   */
  model?: string;

  /**
   * Cache directory for downloaded models
   * @default '.cache/transformers'
   */
  cacheDir?: string;
}

export class TransformersProvider implements EmbeddingProvider {
  readonly name = 'transformers';

  private pipeline: any = null;
  private config: Required<TransformersProviderConfig>;
  private initialized = false;

  constructor(config: TransformersProviderConfig = {}) {
    this.config = {
      model: config.model ?? 'Xenova/all-MiniLM-L6-v2',
      cacheDir: config.cacheDir ?? '.cache/transformers',
      progressLogging: config.progressLogging ?? false,
    };

    env.cacheDir = this.config.cacheDir;
    env.allowLocalModels = false;
  }

  async init(): Promise<void> {
    if (this.initialized) {
      return;
    }

    if (this.config.progressLogging) {
      console.log(`[TransformersProvider] Loading model: ${this.config.model}`);
      console.log(`[TransformersProvider] Cache dir: ${this.config.cacheDir}`);
    }

    this.pipeline = await pipeline('feature-extraction', this.config.model);
    this.initialized = true;

    if (this.config.progressLogging) {
      console.log('[TransformersProvider] Model loaded successfully');
    }
  }

  async embed(text: string): Promise<number[]> {
    if (!this.initialized || !this.pipeline) {
      throw new Error('Embedder not initialized. Call init() first.');
    }

    const output = await this.pipeline(text, {
      pooling: 'mean',
      normalize: true,
    });

    return Array.from(output.data);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (!this.initialized || !this.pipeline) {
      throw new Error('Embedder not initialized. Call init() first.');
    }

    const embeddings: number[][] = [];

    const BATCH_SIZE = 32;
    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
      const batch = texts.slice(i, i + BATCH_SIZE);

      for (const text of batch) {
        const output = await this.pipeline(text, {
          pooling: 'mean',
          normalize: true,
        });
        embeddings.push(Array.from(output.data));
      }
    }

    return embeddings;
  }

  async getDimension(): Promise<number | null> {
    if (!this.initialized) {
      return null;
    }

    const testVector = await this.embed('test');
    return testVector.length;
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  getConfig(): Readonly<Required<TransformersProviderConfig>> {
    return { ...this.config };
  }
}
