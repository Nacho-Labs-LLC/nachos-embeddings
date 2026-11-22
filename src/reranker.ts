/**
 * Local cross-encoder reranker (Transformers.js, no API needed).
 *
 * A cross-encoder reads the query and one passage together and emits one
 * relevance logit. This is slower than comparing embeddings, so use it to
 * re-score the top few dozen candidates from a first-stage search.
 *
 * The reranker owns its own model and is independent of the embedding
 * model, so one process can hold both.
 */

import {
  AutoModelForSequenceClassification,
  AutoTokenizer,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from "@huggingface/transformers";
import {
  TRANSFORMERS_DTYPES,
  resolveDefaultCacheDir,
  type TransformersDtype,
} from "./providers/transformers-provider.js";
import { loadModel, resolveModelRef } from "./providers/model-loading.js";
import { getRerankerProfile } from "./providers/reranker-profiles.js";
import { isOfflineEnv } from "./offline.js";

/** Default reranker: 4-layer Jina cross-encoder, ~35 MB download at q8. */
export const RERANK_DEFAULT_MODEL = "jinaai/jina-reranker-v1-tiny-en";

/** Documented fallback: MS MARCO MiniLM cross-encoder, ~24 MB at q8. */
export const RERANK_FALLBACK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2";

/** Default precision. The quantized weights are the small download. */
export const RERANK_DEFAULT_DTYPE: TransformersDtype = "q8";

/**
 * Default truncation length in tokens (query + passage together) for models
 * without a profile. The default model uses 1024; see `reranker-profiles.ts`.
 */
export const RERANK_DEFAULT_MAX_LENGTH = 512;

/** Pairs per padded model call. */
const BATCH_SIZE = 8;

export interface RerankOptions {
  /** @default RERANK_DEFAULT_MODEL ('jinaai/jina-reranker-v1-tiny-en') */
  model?: string;
  /** @default RERANK_DEFAULT_DTYPE ('q8') */
  dtype?: TransformersDtype;
  /**
   * Truncate each query + passage pair to this many tokens. Longer passages
   * are cut, never an error. Default per model: 1024 for the jina default
   * (ALiBi, so the tokenizer's 512 limit is raised, up to 8192), 512 for
   * models without a profile such as MiniLM. A larger value is silently
   * capped: at 8192 for jina, at the tokenizer's limit (512 for MiniLM, whose
   * graph fails above that) otherwise.
   */
  maxLength?: number;
  /** @default resolveDefaultCacheDir() (see TransformersProviderConfig) */
  cacheDir?: string;
  /**
   * Never touch the network: load only from the cache and fail with
   * `ModelNotCachedError` if the model is not there. Default: true when
   * `NACHOS_EMBEDDINGS_OFFLINE=1`, else false (a fully cached model still
   * loads without network access; it is downloaded only when files are
   * missing).
   */
  offline?: boolean;
}

export type ResolvedRerankOptions = Required<RerankOptions>;

function validate(options: RerankOptions): void {
  const { dtype, maxLength } = options;
  if (
    dtype !== undefined &&
    !(TRANSFORMERS_DTYPES as readonly string[]).includes(dtype)
  ) {
    throw new RangeError(
      `Invalid dtype '${String(dtype)}'. Expected one of: ${TRANSFORMERS_DTYPES.join(", ")}`,
    );
  }
  if (
    maxLength !== undefined &&
    !(Number.isInteger(maxLength) && maxLength > 0)
  ) {
    throw new RangeError(
      `Invalid maxLength '${String(maxLength)}'. Expected a positive integer`,
    );
  }
}

/**
 * Transformers.js clamps `max_length` to the tokenizer's `model_max_length`
 * (512 for the jina reranker, whose ALiBi positions support 8192). There is
 * no public option to lift it, so raise it on the tokenizer's config, only
 * for models whose profile says the model supports longer inputs.
 */
function raiseTokenizerLimit(
  tokenizer: PreTrainedTokenizer,
  maxLength: number,
): void {
  const config = (
    tokenizer as unknown as { _tokenizerConfig?: Record<string, unknown> }
  )._tokenizerConfig;
  if (config === undefined) {
    throw new Error(
      "Cannot raise the tokenizer length limit: this @huggingface/transformers version changed its tokenizer internals. Pass maxLength <= 512.",
    );
  }
  config["model_max_length"] = maxLength;
}

/** Logistic function; stable for large magnitudes (exp overflow gives 0 or 1). */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

interface Loaded {
  tokenizer: PreTrainedTokenizer;
  model: PreTrainedModel;
}

/**
 * Cross-encoder reranker. Scores are `sigmoid(logit)`, in (0, 1): higher is
 * more relevant. Both supported models are single-logit cross-encoders trained
 * with a sigmoid/BCE head, and their reference implementations apply the same
 * sigmoid. Only the order is meaningful across queries; do not compare scores
 * between models.
 *
 * @example
 * ```typescript
 * const reranker = new Reranker();
 * const scores = await reranker.rerank('capital of France', [
 *   'Paris is the capital of France.',
 *   'Bananas are rich in potassium.',
 * ]);
 * ```
 */
export class Reranker {
  private readonly config: ResolvedRerankOptions;
  private loaded: Loaded | null = null;
  private initPromise: Promise<void> | null = null;
  private readonly raisesTokenizerLimit: boolean;

  constructor(options: RerankOptions = {}) {
    validate(options);
    const model = resolveModelRef(options.model ?? RERANK_DEFAULT_MODEL);
    const profile = getRerankerProfile(model);
    this.config = {
      model,
      dtype: options.dtype ?? RERANK_DEFAULT_DTYPE,
      // Capped at what the model supports; without a profile the tokenizer's
      // own limit applies when the tokenizer is called.
      maxLength: Math.min(
        options.maxLength ?? profile.maxLength,
        profile.maxPositions ?? Infinity,
      ),
      cacheDir: options.cacheDir ?? resolveDefaultCacheDir(),
      offline: options.offline ?? isOfflineEnv(),
    };
    this.raisesTokenizerLimit = profile.maxPositions !== undefined;
  }

  /** Loads the model (downloading it on first use). Safe to call repeatedly. */
  init(): Promise<void> {
    if (this.loaded) return Promise.resolve();
    this.initPromise ??= this.load().finally(() => {
      this.initPromise = null;
    });
    return this.initPromise;
  }

  private async load(): Promise<void> {
    const { model, dtype, cacheDir, offline, maxLength } = this.config;
    const loaded = await loadModel(
      { model, cacheDir, dtype, offline },
      async (extra) => ({
        tokenizer: await AutoTokenizer.from_pretrained(model, {
          cache_dir: cacheDir,
          ...extra,
        }),
        model: await AutoModelForSequenceClassification.from_pretrained(model, {
          cache_dir: cacheDir,
          dtype,
          ...extra,
        }),
      }),
    );
    if (this.raisesTokenizerLimit)
      raiseTokenizerLimit(loaded.tokenizer, maxLength);
    this.loaded = loaded;
  }

  isInitialized(): boolean {
    return this.loaded !== null;
  }

  /**
   * Score each passage against the query. Returns one score per passage in
   * input order. Pairs are scored in padded chunks of 8, sorted by length so
   * each chunk pads to similar lengths. An empty list returns `[]`; an empty
   * passage is scored like any other text; over-long pairs are truncated to
   * `maxLength` tokens.
   */
  async rerank(query: string, passages: string[]): Promise<number[]> {
    if (typeof query !== "string") {
      throw new TypeError("query must be a string");
    }
    if (
      !Array.isArray(passages) ||
      passages.some((passage) => typeof passage !== "string")
    ) {
      throw new TypeError("passages must be an array of strings");
    }
    if (passages.length === 0) return [];
    await this.init();
    const { tokenizer, model } = this.loaded as Loaded;

    const order = passages
      .map((_, index) => index)
      .sort(
        (a, b) =>
          (passages[a] as string).length - (passages[b] as string).length,
      );
    const scores = new Array<number>(passages.length);
    for (let i = 0; i < order.length; i += BATCH_SIZE) {
      const indices = order.slice(i, i + BATCH_SIZE);
      const inputs = tokenizer(Array<string>(indices.length).fill(query), {
        text_pair: indices.map((index) => passages[index] as string),
        padding: true,
        truncation: true,
        max_length: this.config.maxLength,
      });
      const { logits } = (await model(inputs)) as {
        logits: { data: ArrayLike<number>; dims: number[] };
      };
      if ((logits.dims[1] ?? 1) !== 1) {
        throw new Error(
          `Model ${this.config.model} is not a single-logit reranker (logits shape ${logits.dims.join("x")})`,
        );
      }
      indices.forEach((index, row) => {
        scores[index] = sigmoid(logits.data[row] as number);
      });
    }
    return scores;
  }

  /** Model id for reporting; a non-default dtype is appended (`model:fp32`). */
  getModelId(): string {
    const { model, dtype } = this.config;
    return dtype === RERANK_DEFAULT_DTYPE ? model : `${model}:${dtype}`;
  }

  getConfig(): Readonly<ResolvedRerankOptions> {
    return { ...this.config };
  }

  /** Release the model session. The reranker can be init()ed again afterwards. */
  async dispose(): Promise<void> {
    if (this.initPromise) await this.initPromise.catch(() => undefined);
    const loaded = this.loaded;
    this.loaded = null;
    await loaded?.model.dispose();
  }
}

const shared = new Map<string, Reranker>();

/**
 * Score passages against a query, returning one score per passage in input
 * order. The model loads lazily on first use and is cached per
 * (model, dtype, maxLength, cacheDir) for the life of the process; it is
 * independent of any embedding model. Use {@link Reranker} directly to
 * control its lifetime.
 */
export function rerank(
  query: string,
  passages: string[],
  options: RerankOptions = {},
): Promise<number[]> {
  validate(options);
  // Resolve through a Reranker so the cache key holds the effective config.
  const probe = new Reranker(options).getConfig();
  const key = JSON.stringify(probe);
  let reranker = shared.get(key);
  if (!reranker) {
    reranker = new Reranker(probe);
    shared.set(key, reranker);
  }
  return reranker.rerank(query, passages);
}

/** Dispose every reranker cached by {@link rerank}. */
export async function disposeRerankers(): Promise<void> {
  const all = [...shared.values()];
  shared.clear();
  await Promise.all(all.map((reranker) => reranker.dispose()));
}
