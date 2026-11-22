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
import type {
  EmbeddingProvider,
  BaseProviderConfig,
  EmbedOptions,
} from "./types.js";
import {
  getModelProfile,
  assertInputType,
  assertPooling,
  type PoolingMode,
  type ModelProfile,
} from "./model-profiles.js";

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

  /**
   * Pooling of token embeddings. Defaults per model from the built-in table
   * (bge-*-en-v1.5: `cls`; everything else: `mean`). Vectors are always
   * L2-normalized.
   */
  pooling?: PoolingMode;

  /** Prefix prepended to query texts. Default: built-in per-model table. */
  queryPrefix?: string;

  /** Prefix prepended to document texts. Default: built-in per-model table. */
  documentPrefix?: string;

  /**
   * Truncate inputs to this many tokens (including special tokens). It can
   * only lower the limit: the tokenizer's own `model_max_length` (usually 512,
   * 256 for some models) always applies. Unset: truncate at the tokenizer
   * limit; longer text is silently cut, never an error.
   */
  maxLength?: number;
}

export type ResolvedTransformersConfig = Required<
  Omit<TransformersProviderConfig, "dtype" | "maxLength">
> & { dtype?: TransformersDtype; maxLength?: number };

/** Texts per padded pipeline call in `embedBatch`. */
const BATCH_SIZE = 8;

/**
 * Transformers.js reads `env.allowLocalModels` from process-global state, so
 * loads are serialized and the flag is set only for the duration of a load
 * (then restored). Cache directory and dtype are passed per pipeline call and
 * do not depend on globals. Limitation: loads from different provider
 * instances run one at a time.
 */
let loadLock: Promise<void> = Promise.resolve();

export async function withRemoteModelsOnly<T>(
  fn: () => Promise<T>,
): Promise<T> {
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

/**
 * transformers.js feature-extraction calls its tokenizer with
 * `{ padding, truncation }` and no max length option, so wrap the tokenizer
 * to add `max_length` (the tokenizer still caps it at its own limit).
 */
function withMaxLength(
  extractor: FeatureExtractionPipeline,
  maxLength: number,
): FeatureExtractionPipeline {
  extractor.tokenizer = new Proxy(extractor.tokenizer, {
    apply: (target, thisArg, args: unknown[]) =>
      Reflect.apply(target, thisArg, [
        args[0],
        { ...(args[1] as object), max_length: maxLength },
      ]),
  });
  return extractor;
}

/**
 * Parameter-free layer norm (torch default eps 1e-5) followed by L2
 * normalization, as nomic-embed-text-v1.5's card prescribes after pooling.
 */
export function layerNormThenNormalize(row: ArrayLike<number>): number[] {
  const n = row.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += row[i] as number;
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += ((row[i] as number) - mean) ** 2;
  const scale = 1 / Math.sqrt(variance / n + 1e-5);
  const out = Array.from(row, (x) => (x - mean) * scale);
  const norm = Math.hypot(...out) || 1;
  return out.map((x) => x / norm);
}

function validateConfig(config: TransformersProviderConfig): void {
  if (
    config.dtype !== undefined &&
    !(TRANSFORMERS_DTYPES as readonly string[]).includes(config.dtype)
  ) {
    throw new RangeError(
      `Invalid dtype '${String(config.dtype)}'. Expected one of: ${TRANSFORMERS_DTYPES.join(", ")}`,
    );
  }
  if (config.pooling !== undefined) assertPooling(config.pooling);
  const { maxLength } = config;
  if (
    maxLength !== undefined &&
    !(Number.isInteger(maxLength) && maxLength > 0)
  ) {
    throw new RangeError(
      `Invalid maxLength '${String(maxLength)}'. Expected a positive integer`,
    );
  }
}

export class TransformersProvider implements EmbeddingProvider {
  readonly name = "transformers";

  private pipeline: FeatureExtractionPipeline | null = null;
  private config: ResolvedTransformersConfig;
  private initialized = false;
  private initPromise: Promise<void> | null = null;
  private defaults: ModelProfile;

  constructor(config: TransformersProviderConfig = {}) {
    validateConfig(config);
    const model = config.model ?? TRANSFORMERS_DEFAULT_MODEL;
    const profile = getModelProfile(model);
    this.config = {
      model,
      cacheDir: config.cacheDir ?? resolveDefaultCacheDir(),
      progressLogging: config.progressLogging ?? false,
      pooling: config.pooling ?? profile.pooling,
      queryPrefix: config.queryPrefix ?? profile.queryPrefix,
      documentPrefix: config.documentPrefix ?? profile.documentPrefix,
      ...(config.dtype !== undefined && { dtype: config.dtype }),
      ...(config.maxLength !== undefined && { maxLength: config.maxLength }),
    };
    this.defaults = profile;
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
      // eslint-disable-next-line no-console -- opt-in `progressLogging` is documented to log to the console
      console.log(`[TransformersProvider] Loading model: ${model}`);
      // eslint-disable-next-line no-console -- opt-in `progressLogging` is documented to log to the console
      console.log(`[TransformersProvider] Cache dir: ${cacheDir}`);
    }

    const loaded = await withRemoteModelsOnly(() =>
      pipeline("feature-extraction", model, {
        cache_dir: cacheDir,
        ...(dtype !== undefined && { dtype }),
      }),
    );
    this.pipeline = this.config.maxLength
      ? withMaxLength(loaded, this.config.maxLength)
      : loaded;
    this.initialized = true;

    if (progressLogging) {
      // eslint-disable-next-line no-console -- opt-in `progressLogging` is documented to log to the console
      console.log("[TransformersProvider] Model loaded successfully");
    }
  }

  private prepare(texts: string[], options?: EmbedOptions): string[] {
    const type = options?.type ?? "document";
    assertInputType(type);
    const prefix =
      type === "query" ? this.config.queryPrefix : this.config.documentPrefix;
    return prefix ? texts.map((text) => prefix + text) : texts;
  }

  /** Layer-norm models are normalized by `finish`, after the layer norm. */
  private callOptions() {
    return {
      pooling: this.config.pooling,
      normalize: !this.defaults.layerNorm,
    };
  }

  private finish(row: Float32Array): number[] {
    return this.defaults.layerNorm
      ? layerNormThenNormalize(row)
      : Array.from(row);
  }

  async embed(text: string, options?: EmbedOptions): Promise<number[]> {
    if (!this.initialized || !this.pipeline) {
      throw new Error("Embedder not initialized. Call init() first.");
    }
    const [input] = this.prepare([text], options);
    const output = await this.pipeline(input as string, this.callOptions());

    return this.finish(output.data as Float32Array);
  }

  /**
   * One padded pipeline call per chunk of 8 texts. Texts are ordered by
   * length first so each chunk pads to similar lengths (mixed-length input
   * is otherwise slower than single calls); results come back in input order.
   */
  async embedBatch(
    texts: string[],
    options?: EmbedOptions,
  ): Promise<number[][]> {
    const extractor = this.pipeline;
    if (!this.initialized || !extractor) {
      throw new Error("Embedder not initialized. Call init() first.");
    }
    const inputs = this.prepare(texts, options);
    const order = inputs
      .map((_, index) => index)
      .sort(
        (a, b) => (inputs[a] as string).length - (inputs[b] as string).length,
      );
    const embeddings = new Array<number[]>(inputs.length);
    for (let i = 0; i < order.length; i += BATCH_SIZE) {
      const indices = order.slice(i, i + BATCH_SIZE);
      const output = await extractor(
        indices.map((index) => inputs[index] as string),
        this.callOptions(),
      );
      const dim = output.dims[1] ?? 0;
      const data = output.data as Float32Array;
      indices.forEach((index, row) => {
        embeddings[index] = this.finish(
          data.subarray(row * dim, (row + 1) * dim),
        );
      });
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
    const { model, dtype, pooling, documentPrefix } = this.config;
    let id =
      dtype && dtype !== "fp32" && dtype !== "auto"
        ? `${model}:${dtype}`
        : model;
    // Overrides that change stored (document) vectors also change the id.
    if (pooling !== this.defaults.pooling) id += `:pooling=${pooling}`;
    if (documentPrefix !== this.defaults.documentPrefix) {
      id += `:docprefix=${JSON.stringify(documentPrefix)}`;
    }
    return id;
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
