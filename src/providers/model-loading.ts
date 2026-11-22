/**
 * Cache-first model loading for Transformers.js.
 *
 * Transformers.js 4 decides which tokenizer files exist by sending a request
 * to the hub on every load (`get_tokenizer_files` checks
 * `tokenizer_config.json` with default options, ignoring `cache_dir`), so even
 * a fully cached model contacted the network each time. When all of a model's
 * files are in the cache we load with `local_files_only`, which makes no
 * request. Only when files are missing do we fall back to the normal download
 * path (or, in offline mode, fail with a clear error).
 *
 * Whether a model is cached is decided up front from the files on disk, not
 * by trying a local load first: a failed local load would be memoized inside
 * Transformers.js (as "tokenizer_config.json does not exist") and break the
 * download that follows in the same process.
 */

import { access } from "node:fs/promises";
import { join } from "node:path";
import { env } from "@huggingface/transformers";
import { OFFLINE_ENV_VAR } from "../offline.js";

/** Thrown in offline mode when a model's files are not in the cache. */
export class ModelNotCachedError extends Error {
  override readonly name = "ModelNotCachedError";

  constructor(
    readonly model: string,
    readonly cacheDir: string,
    cause: unknown,
  ) {
    const detail = cause instanceof Error ? cause.message : String(cause);
    super(
      `Model '${model}' is not in the cache (${cacheDir}) and offline mode is on ` +
        `(--offline or ${OFFLINE_ENV_VAR}=1), so nothing was downloaded. ` +
        `Fetch it once with network access: nachos-embeddings pull ${model} ` +
        `[--dtype <type>] [--cache-dir <dir>]. Cause: ${detail}`,
      { cause },
    );
  }
}

export interface ModelLoadTarget {
  model: string;
  cacheDir: string;
  /** Weights precision; selects which ONNX file must be cached. */
  dtype?: string | undefined;
  /** Never touch the network; fail with {@link ModelNotCachedError} instead. */
  offline: boolean;
}

/** ONNX file name suffix per dtype (Transformers.js `DEFAULT_DTYPE_SUFFIX_MAPPING`). */
const DTYPE_SUFFIX: Record<string, string> = {
  auto: "",
  fp32: "",
  fp16: "_fp16",
  int8: "_int8",
  uint8: "_uint8",
  q8: "_quantized",
  q4: "_q4",
  q4f16: "_q4f16",
  bnb4: "_bnb4",
};

/**
 * Files a model needs to load: config, tokenizer and the ONNX weights for the
 * dtype. Returns the ones missing from `<cacheDir>/<model>/`. Models split
 * into external-data files are caught by the load itself (which then falls
 * back to a download, or fails in offline mode).
 */
export async function missingCacheFiles(
  target: Pick<ModelLoadTarget, "model" | "cacheDir" | "dtype">,
): Promise<string[]> {
  const suffix = DTYPE_SUFFIX[target.dtype ?? "fp32"];
  const files = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    `onnx/model${suffix ?? ""}.onnx`,
  ];
  const missing: string[] = [];
  for (const file of files) {
    try {
      await access(join(target.cacheDir, target.model, file));
    } catch {
      missing.push(file);
    }
  }
  return suffix === undefined
    ? [...missing, `onnx/model.onnx for dtype '${target.dtype}'`]
    : missing;
}

/** Extra options to merge into the Transformers.js `from_pretrained` call. */
export interface ModelLoadOptions {
  local_files_only?: boolean;
}

/**
 * Transformers.js reads `env.allowLocalModels`, `env.allowRemoteModels` and
 * `env.localModelPath` from process-global state, so loads are serialized and
 * the flags are set only for the duration of a load (then restored).
 * Limitation: loads from different instances run one at a time.
 */
let loadLock: Promise<void> = Promise.resolve();

async function acquireLoadLock(): Promise<() => void> {
  const previous = loadLock;
  let release!: () => void;
  loadLock = new Promise<void>((resolve) => {
    release = resolve;
  });
  await previous;
  return release;
}

/**
 * Run `load` against the cache only when the model is fully cached (no network
 * request). Otherwise, unless offline, run it with downloads allowed. During
 * the cache-only load local lookups are pointed at the cache directory (so a
 * stray `./models/<id>` folder is never picked up) and remote access is off;
 * during the download load local lookups are disabled for the same reason.
 */
export async function loadModel<T>(
  target: ModelLoadTarget,
  load: (options: ModelLoadOptions) => Promise<T>,
): Promise<T> {
  const missing = await missingCacheFiles(target);
  if (missing.length > 0 && target.offline) {
    throw new ModelNotCachedError(
      target.model,
      target.cacheDir,
      new Error(`missing: ${missing.join(", ")}`),
    );
  }
  const release = await acquireLoadLock();
  const saved = {
    allowLocalModels: env.allowLocalModels,
    allowRemoteModels: env.allowRemoteModels,
    localModelPath: env.localModelPath,
  };
  try {
    if (missing.length === 0) {
      try {
        env.allowLocalModels = true;
        env.allowRemoteModels = false;
        env.localModelPath = target.cacheDir;
        return await load({ local_files_only: true });
      } catch (cause) {
        if (target.offline) {
          throw new ModelNotCachedError(target.model, target.cacheDir, cause);
        }
      }
    }
    env.allowLocalModels = false;
    env.allowRemoteModels = saved.allowRemoteModels;
    env.localModelPath = saved.localModelPath;
    return await load({});
  } finally {
    env.allowLocalModels = saved.allowLocalModels;
    env.allowRemoteModels = saved.allowRemoteModels;
    env.localModelPath = saved.localModelPath;
    release();
  }
}
