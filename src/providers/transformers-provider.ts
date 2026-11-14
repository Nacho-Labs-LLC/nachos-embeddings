/**
 * Embedding provider using Transformers.js (local, no API needed)
 */

import { homedir } from "node:os";
import { posix, win32 } from "node:path";
import {
  pipeline,
  env,
  type FeatureExtractionPipeline,
} from "@huggingface/transformers";
import type { EmbeddingProvider, BaseProviderConfig } from "./types.js";

/** Default Transformers.js model. */
export const TRANSFORMERS_DEFAULT_MODEL = "Xenova/all-MiniLM-L6-v2";

/** Vector dimension of {@link TRANSFORMERS_DEFAULT_MODEL} (other models differ). */
export const TRANSFORMERS_DEFAULT_DIMENSION = 384;

/** Quantization / precision variants accepted by Transformers.js v4. */
export const TRANSFORMERS_DTYPES = [
  "auto",
  "fp32",
  "fp16",
  "q8",
  "int8",
  "uint8",
  "q4",
  "bnb4",
  "q4f16",
] as const;

export type TransformersDtype = (typeof TRANSFORMERS_DTYPES)[number];

/** Environment inputs for {@link resolveDefaultCacheDir}; injectable for tests. */
export interface CacheDirEnvironment {
  env?: Record<string, string | undefined>;
  platform?: string;
  homeDir?: string;
}

/**
 * Per-user model cache location:
 * `$NACHOS_EMBEDDINGS_CACHE`, else `$XDG_CACHE_HOME/nachos-embeddings`,
 * else `~/.cache/nachos-embeddings` (Windows: `%LOCALAPPDATA%\nachos-embeddings`).
 */
export function resolveDefaultCacheDir(
  options: CacheDirEnvironment = {},
): string {
  const env = options.env ?? process.env;
  const platform = options.platform ?? process.platform;
  const home = options.homeDir ?? homedir();
  const override = env["NACHOS_EMBEDDINGS_CACHE"];
  if (override) return override;

  if (platform === "win32") {
    const base = env["LOCALAPPDATA"] || win32.join(home, "AppData", "Local");
    return win32.join(base, "nachos-embeddings");
  }
  const xdg = env["XDG_CACHE_HOME"];
  if (xdg) return posix.join(xdg, "nachos-embeddings");
  return posix.join(home, ".cache", "nachos-embeddings");
}

export interface TransformersProviderConfig extends BaseProviderConfig {
  /**
   * Model to use for embeddings
   * @default TRANSFORMERS_DEFAULT_MODEL ('Xenova/all-MiniLM-L6-v2')
   */
  model?: string;

  /**
   * Cache directory for downloaded models
   * @default resolveDefaultCacheDir() (per-user; see its docs)
   */
  cacheDir?: string;

  /**
   * Precision / quantization of the model weights (e.g. 'q8' is smaller and
   * faster, with slightly different vectors). Unset uses the library default
   * (fp32 on Node). A non-default dtype is appended to the model id.
   */
  dtype?: TransformersDtype;
}

export type ResolvedTransformersConfig = Required<
  Omit<TransformersProviderConfig, "dtype">
> & { dtype?: TransformersDtype };

/**
 * Transformers.js reads `env.allowLocalModels` from process-global state, so
 * loads are serialized and the flag is set only for the duration of a load
 * (then restored). Cache directory and dtype are passed per pipeline call and
 * do not depend on globals. Limitation: loads from different provider
 * instances run one at a time.
 */
let loadLock: Promise<void> = Promise.resolve();

async function withRemoteModelsOnly<T>(fn: () => Promise<T>): Promise<T> {
  const previous = loadLock;
  let release!: () => void;
  loadLock = new Promise<void>((resolve) => {
    release = resolve;
  });
  await previous;
  const saved = env.allowLocalModels;
  env.allowLocalModels = false;
  try {
    return await fn();
  } finally {
    env.allowLocalModels = saved;
    release();
  }
}

export class TransformersProvider implements EmbeddingProvider {
  readonly name = "transformers";

  private pipeline: FeatureExtractionPipeline | null = null;
  private config: ResolvedTransformersConfig;
  private initialized = false;
  private initPromise: Promise<void> | null = null;

  constructor(config: TransformersProviderConfig = {}) {
    if (
      config.dtype !== undefined &&
      !(TRANSFORMERS_DTYPES as readonly string[]).includes(config.dtype)
    ) {
      throw new RangeError(
        `Invalid dtype '${String(config.dtype)}'. Expected one of: ${TRANSFORMERS_DTYPES.join(", ")}`,
      );
    }
    this.config = {
      model: config.model ?? TRANSFORMERS_DEFAULT_MODEL,
      cacheDir: config.cacheDir ?? resolveDefaultCacheDir(),
      progressLogging: config.progressLogging ?? false,
      ...(config.dtype !== undefined && { dtype: config.dtype }),
    };
  }

  init(): Promise<void> {
    if (this.initialized) return Promise.resolve();
    // Memoize the in-flight load; clear it afterwards so a failure can be retried.
    this.initPromise ??= this.load().finally(() => {
      this.initPromise = null;
    });
    return this.initPromise;
  }

  private async load(): Promise<void> {
    const { model, cacheDir, dtype, progressLogging } = this.config;
    if (progressLogging) {
      console.log(`[TransformersProvider] Loading model: ${model}`);
      console.log(`[TransformersProvider] Cache dir: ${cacheDir}`);
    }

    const loaded = await withRemoteModelsOnly(() =>
      pipeline("feature-extraction", model, {
        cache_dir: cacheDir,
        ...(dtype !== undefined && { dtype }),
      }),
    );
    this.pipeline = loaded;
    this.initialized = true;

    if (progressLogging) {
      console.log("[TransformersProvider] Model loaded successfully");
    }
  }

  async embed(text: string): Promise<number[]> {
    if (!this.initialized || !this.pipeline) {
      throw new Error("Embedder not initialized. Call init() first.");
    }

    const output = await this.pipeline(text, {
      pooling: "mean",
      normalize: true,
    });

    return Array.from(output.data);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const extractor = this.pipeline;
    if (!this.initialized || !extractor) {
      throw new Error("Embedder not initialized. Call init() first.");
    }

    const embeddings: number[][] = [];

    const BATCH_SIZE = 32;
    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
      const batch = texts.slice(i, i + BATCH_SIZE);

      const batchResults = await Promise.all(
        batch.map((text) =>
          extractor(text, {
            pooling: "mean",
            normalize: true,
          }),
        ),
      );

      for (const output of batchResults) {
        embeddings.push(Array.from(output.data));
      }
    }

    return embeddings;
  }

  async getDimension(): Promise<number | null> {
    if (!this.initialized) {
      return null;
    }

    const testVector = await this.embed("test");
    return testVector.length;
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Model identifier for persistence. A non-default dtype is appended
   * (e.g. `Xenova/all-MiniLM-L6-v2:q8`) so consumers re-embed on change.
   */
  getModelId(): string {
    const { model, dtype } = this.config;
    return dtype && dtype !== "fp32" && dtype !== "auto"
      ? `${model}:${dtype}`
      : model;
  }

  /** Release the model session. The provider can be init()ed again afterwards. */
  async dispose(): Promise<void> {
    if (this.initPromise) {
      await this.initPromise.catch(() => undefined);
    }
    const loaded = this.pipeline;
    this.pipeline = null;
    this.initialized = false;
    await loaded?.dispose();
  }

  getConfig(): Readonly<ResolvedTransformersConfig> {
    return { ...this.config };
  }
}
