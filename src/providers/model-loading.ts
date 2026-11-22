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

import { access, readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { AutoConfig, env } from "@huggingface/transformers";
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
export const DTYPE_SUFFIX: Readonly<Record<string, string>> = {
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

/** ONNX weights file (relative to a model directory) for a dtype. */
export function weightsFile(
  dtype: string | undefined,
  folder = "onnx",
): string {
  const suffix = DTYPE_SUFFIX[dtype ?? "fp32"] ?? "";
  return `${folder ? `${folder}/` : ""}model${suffix}.onnx`;
}

/** Why a model cannot be used. `code` is stable for programmatic handling. */
export type ModelErrorCode =
  | "not-found"
  | "no-onnx-weights"
  | "unsupported-architecture"
  | "local-model-incomplete";

export class ModelLoadError extends Error {
  override readonly name = "ModelLoadError";

  constructor(
    readonly code: ModelErrorCode,
    readonly model: string,
    message: string,
    options?: { cause?: unknown },
  ) {
    super(message, options);
  }
}

/**
 * True when `model` names a directory on disk rather than a hub id. Only
 * explicit paths count (`/abs`, `./rel`, `../rel`, `~/dir`, `C:\dir`,
 * `file://`), because `org/name` is a valid hub id.
 */
export function isLocalModelPath(model: string): boolean {
  return /^(\.{1,2}[\\/]|[\\/]|~[\\/]|[a-zA-Z]:[\\/]|file:)/.test(model);
}

/** Absolute path for a local model reference; hub ids are returned unchanged. */
export function resolveModelRef(model: string): string {
  if (!isLocalModelPath(model)) return model;
  if (model.startsWith("file:")) return resolve(fileURLToPath(model));
  return resolve(
    model.startsWith("~") ? join(homedir(), model.slice(1)) : model,
  );
}

async function exists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

/**
 * Files a model directory needs to load: config, tokenizer and the ONNX
 * weights for the dtype. Returns the ones missing from `directory`. Models
 * split into external-data files are caught by the load itself.
 */
export async function missingModelFiles(
  directory: string,
  dtype: string | undefined,
  weightsFolder = "onnx",
): Promise<string[]> {
  const files = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    weightsFile(dtype, weightsFolder),
  ];
  const missing: string[] = [];
  for (const file of files) {
    if (!(await exists(join(directory, file)))) missing.push(file);
  }
  const known = dtype === undefined || dtype in DTYPE_SUFFIX;
  return known ? missing : [...missing, `(unknown dtype '${dtype}')`];
}

/** Files missing from the cache for a hub model (see {@link missingModelFiles}). */
export function missingCacheFiles(
  target: Pick<ModelLoadTarget, "model" | "cacheDir" | "dtype">,
): Promise<string[]> {
  return missingModelFiles(join(target.cacheDir, target.model), target.dtype);
}

interface ConfigLike {
  model_type?: unknown;
  architectures?: unknown;
  is_encoder_decoder?: unknown;
}

const GENERATIVE_ARCHITECTURE =
  /CausalLM|LMHeadModel|ForConditionalGeneration|Seq2Seq|ForVision2Seq|Whisper|CLIP/i;

/**
 * Embedding and cross-encoder models are encoder-only transformers (BERT,
 * RoBERTa, XLM-R, DistilBERT, MPNet, ModernBERT, ...). Reject the families
 * that cannot work, up front, with a message that says what to pick instead.
 */
export function assertEncoderArchitecture(
  model: string,
  config: ConfigLike,
): void {
  const architectures = Array.isArray(config.architectures)
    ? config.architectures.map(String)
    : [];
  const generative = architectures.find((a) => GENERATIVE_ARCHITECTURE.test(a));
  if (config.is_encoder_decoder === true || generative !== undefined) {
    const what =
      generative ??
      `an encoder-decoder (model_type '${String(config.model_type)}')`;
    throw new ModelLoadError(
      "unsupported-architecture",
      model,
      `Model '${model}' is ${what}, which cannot produce sentence embeddings or relevance scores. ` +
        `Use an encoder-only model (BERT, RoBERTa, XLM-R, DistilBERT, MPNet, ModernBERT or similar) that has an ONNX export.`,
    );
  }
  if (typeof config.model_type !== "string" || config.model_type === "") {
    throw new ModelLoadError(
      "unsupported-architecture",
      model,
      `Model '${model}' has no model_type in its config.json, so Transformers.js cannot tell which architecture to load.`,
    );
  }
}

/** Turn a Transformers.js failure into a {@link ModelLoadError} when we recognise it. */
export function explainLoadError(
  model: string,
  dtype: string | undefined,
  cause: unknown,
): unknown {
  if (cause instanceof ModelLoadError) return cause;
  const message = cause instanceof Error ? cause.message : String(cause);
  const opts = { cause };
  if (/Unsupported model type/i.test(message)) {
    return new ModelLoadError(
      "unsupported-architecture",
      model,
      `Model '${model}' uses an architecture Transformers.js does not support (${message}). ` +
        `Pick another encoder model, see \`nachos-embeddings models\` for tested ones.`,
      opts,
    );
  }
  const file = /(?:locate|access to) file: "([^"]+)"/.exec(message)?.[1] ?? "";
  if (/\.onnx(_data)?$/.test(file)) {
    return new ModelLoadError(
      "no-onnx-weights",
      model,
      `Model '${model}' has no ONNX weights for dtype '${dtype ?? "fp32"}' (missing ${file.split("/resolve/main/")[1] ?? file}). ` +
        `Try another --dtype, or use a repository that ships an ONNX export (Xenova/..., onnx-community/..., or a model with an onnx/ folder); ` +
        `you can also export your own with \`optimum-cli export onnx\` and pass its directory.`,
      opts,
    );
  }
  if (/Unauthorized access|not a valid model identifier|404/i.test(message)) {
    return new ModelLoadError(
      "not-found",
      model,
      `Model '${model}' was not found on the Hugging Face hub. Check the id for typos; ` +
        `private or gated models need HF_TOKEN in the environment. (${message})`,
      opts,
    );
  }
  return cause;
}

/** Extra options to merge into the Transformers.js `from_pretrained` call. */
export interface ModelLoadOptions {
  local_files_only?: boolean;
  /** Folder holding the ONNX weights; `""` for models with `model.onnx` at the top. */
  subfolder?: string;
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

async function readConfig(directory: string): Promise<ConfigLike | undefined> {
  try {
    return JSON.parse(await readFile(join(directory, "config.json"), "utf8"));
  } catch {
    return undefined;
  }
}

/** Run `load` with the Transformers.js globals set, restoring them afterwards. */
async function withEnv<T>(
  settings: { local: boolean; remote?: boolean; localModelPath?: string },
  run: () => Promise<T>,
): Promise<T> {
  const saved = {
    allowLocalModels: env.allowLocalModels,
    allowRemoteModels: env.allowRemoteModels,
    localModelPath: env.localModelPath,
  };
  const release = await acquireLoadLock();
  try {
    env.allowLocalModels = settings.local;
    if (settings.remote !== undefined) env.allowRemoteModels = settings.remote;
    if (settings.localModelPath !== undefined) {
      env.localModelPath = settings.localModelPath;
    }
    return await run();
  } finally {
    env.allowLocalModels = saved.allowLocalModels;
    env.allowRemoteModels = saved.allowRemoteModels;
    env.localModelPath = saved.localModelPath;
    release();
  }
}

/** A model directory on disk: complete files, no network, ever. */
async function loadLocalDirectory<T>(
  target: ModelLoadTarget,
  load: (options: ModelLoadOptions) => Promise<T>,
): Promise<T> {
  const { model, dtype } = target;
  const nested = await missingModelFiles(model, dtype);
  const flat = await missingModelFiles(model, dtype, "");
  const folder = flat.length < nested.length ? "" : "onnx";
  const missing = folder === "" ? flat : nested;
  if (missing.length > 0) {
    throw new ModelLoadError(
      "local-model-incomplete",
      model,
      `Local model directory '${model}' is missing: ${missing.join(", ")}. ` +
        `It needs config.json, tokenizer.json, tokenizer_config.json and the ONNX weights ` +
        `(${weightsFile(dtype)} or ${weightsFile(dtype, "")}) for dtype '${dtype ?? "fp32"}'.`,
    );
  }
  assertEncoderArchitecture(model, (await readConfig(model)) ?? {});
  return withEnv({ local: true, remote: false }, () =>
    load({
      local_files_only: true,
      ...(folder === "" && { subfolder: "" }),
    }).catch((cause: unknown) => {
      throw explainLoadError(model, dtype, cause);
    }),
  );
}

/**
 * Load a model, never touching the network when it can be avoided.
 *
 * - A local directory path loads from that directory only.
 * - A hub model whose files are all cached loads with `local_files_only`
 *   (no request). Otherwise, unless offline, it downloads; offline it fails
 *   with {@link ModelNotCachedError}.
 *
 * During a cache-only load, local lookups are pointed at the cache directory
 * (so a stray `./models/<id>` folder is never picked up) and remote access is
 * off; during a download local lookups are disabled for the same reason.
 * Unsupported architectures and missing ONNX weights surface as
 * {@link ModelLoadError} with an actionable message.
 */
export async function loadModel<T>(
  target: ModelLoadTarget,
  load: (options: ModelLoadOptions) => Promise<T>,
): Promise<T> {
  if (isLocalModelPath(target.model)) return loadLocalDirectory(target, load);
  const missing = await missingCacheFiles(target);
  if (missing.length > 0 && target.offline) {
    throw new ModelNotCachedError(
      target.model,
      target.cacheDir,
      new Error(`missing: ${missing.join(", ")}`),
    );
  }
  if (missing.length === 0) {
    const cachedConfig = await readConfig(join(target.cacheDir, target.model));
    if (cachedConfig) assertEncoderArchitecture(target.model, cachedConfig);
    try {
      return await withEnv(
        { local: true, remote: false, localModelPath: target.cacheDir },
        () => load({ local_files_only: true }),
      );
    } catch (cause) {
      if (target.offline) {
        throw new ModelNotCachedError(target.model, target.cacheDir, cause);
      }
    }
  }
  return withEnv({ local: false }, async () => {
    try {
      // Fetch the small config first: refuse unusable architectures before
      // downloading hundreds of MB of weights.
      const config = await AutoConfig.from_pretrained(target.model, {
        cache_dir: target.cacheDir,
      });
      assertEncoderArchitecture(target.model, config);
      return await load({});
    } catch (cause) {
      throw explainLoadError(target.model, target.dtype, cause);
    }
  });
}
