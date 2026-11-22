/**
 * Inspecting models on disk: what is cached, and what a given model id is
 * (profile, files, dimension) without downloading anything.
 */

import { readFile, readdir, stat } from "node:fs/promises";
import { join } from "node:path";
import {
  DTYPE_SUFFIX,
  isLocalModelPath,
  missingModelFiles,
  resolveModelRef,
} from "./providers/model-loading.js";
import {
  findCatalogEntry,
  profileFor,
  type CatalogEntry,
  type EffectiveProfile,
  type ModelKind,
} from "./providers/model-catalog.js";
import { looksLikeReranker } from "./providers/reranker-profiles.js";

export interface CachedModel {
  id: string;
  kind: ModelKind;
  /** dtypes whose ONNX weights are cached, e.g. `["q8"]`. */
  dtypes: string[];
  /** Total bytes of the model's cached files. */
  bytes: number;
  modelType?: string;
  dimension?: number;
  /** True when the config, tokenizer and at least one weights file are present. */
  complete: boolean;
}

export interface ModelInfo {
  id: string;
  kind: ModelKind;
  source: "hub" | "local";
  catalog?: CatalogEntry;
  profile: EffectiveProfile;
  /** Where the files live (cache entry or local directory). */
  directory: string;
  cached: boolean;
  dtypes: string[];
  /** Files missing for the requested dtype (empty when ready to load). */
  missing: string[];
  dtype: string;
  modelType?: string;
  architectures?: string[];
  dimension?: number;
  maxPositions?: number;
}

interface RawConfig {
  model_type?: string;
  architectures?: string[];
  hidden_size?: number;
  d_model?: number;
  dim?: number;
  max_position_embeddings?: number;
}

const SUFFIX_TO_DTYPE = new Map(
  Object.entries(DTYPE_SUFFIX)
    .filter(([name]) => name !== "auto")
    .map(([name, suffix]) => [suffix, name]),
);

async function readJson<T>(path: string): Promise<T | undefined> {
  try {
    return JSON.parse(await readFile(path, "utf8")) as T;
  } catch {
    return undefined;
  }
}

async function listDir(path: string): Promise<string[]> {
  try {
    return await readdir(path);
  } catch {
    return [];
  }
}

/** dtypes of the ONNX weights present in a model directory (top or `onnx/`). */
async function cachedDtypes(directory: string): Promise<string[]> {
  const found = new Set<string>();
  for (const folder of [join(directory, "onnx"), directory]) {
    for (const file of await listDir(folder)) {
      const match = /^model(_[a-z0-9]+)?\.onnx$/.exec(file);
      if (!match) continue;
      const dtype = SUFFIX_TO_DTYPE.get(match[1] ?? "");
      if (dtype) found.add(dtype);
    }
  }
  return [...found].sort();
}

async function directoryBytes(directory: string): Promise<number> {
  let total = 0;
  for (const name of await listDir(directory)) {
    const path = join(directory, name);
    const info = await stat(path).catch(() => undefined);
    if (info?.isDirectory()) total += await directoryBytes(path);
    else if (info) total += info.size;
  }
  return total;
}

function dimensionOf(config: RawConfig | undefined): number | undefined {
  return config?.hidden_size ?? config?.d_model ?? config?.dim;
}

/** Models in the cache directory (`<org>/<name>/` with a config.json). */
export async function listCachedModels(
  cacheDir: string,
): Promise<CachedModel[]> {
  const models: CachedModel[] = [];
  for (const org of await listDir(cacheDir)) {
    for (const name of await listDir(join(cacheDir, org))) {
      const directory = join(cacheDir, org, name);
      const config = await readJson<RawConfig>(join(directory, "config.json"));
      if (!config) continue;
      const id = `${org}/${name}`;
      const dtypes = await cachedDtypes(directory);
      const kind = kindOf(id, config);
      const dimension = kind === "embedding" ? dimensionOf(config) : undefined;
      models.push({
        id,
        kind,
        dtypes,
        bytes: await directoryBytes(directory),
        ...(config.model_type && { modelType: config.model_type }),
        ...(dimension !== undefined && { dimension }),
        complete:
          dtypes.length > 0 &&
          (await missingModelFiles(directory, dtypes[0])).length === 0,
      });
    }
  }
  return models.sort((a, b) => a.id.localeCompare(b.id));
}

function kindOf(id: string, config?: RawConfig): ModelKind {
  const entry = findCatalogEntry(id);
  if (entry) return entry.kind;
  const sequenceClassifier = config?.architectures?.some((a) =>
    /ForSequenceClassification$/.test(a),
  );
  return looksLikeReranker(id) || sequenceClassifier ? "rerank" : "embedding";
}

/** What a config.json says, in the fields `inspectModel` reports. */
function configFacts(
  config: RawConfig | undefined,
  kind: ModelKind,
  fallbackDimension: number | undefined,
): Partial<ModelInfo> {
  const dimension = dimensionOf(config) ?? fallbackDimension;
  return {
    ...(config?.model_type && { modelType: config.model_type }),
    ...(config?.architectures && { architectures: config.architectures }),
    ...(dimension !== undefined && kind === "embedding" && { dimension }),
    ...(config?.max_position_embeddings !== undefined && {
      maxPositions: config.max_position_embeddings,
    }),
  };
}

/**
 * Describe a model without loading or downloading it: its profile, whether
 * the files for `dtype` are on disk, and what its config says.
 */
export async function inspectModel(
  model: string,
  options: { dtype?: string; cacheDir: string; kind?: ModelKind },
): Promise<ModelInfo> {
  const id = resolveModelRef(model);
  const local = isLocalModelPath(model);
  const directory = local ? id : join(options.cacheDir, id);
  const config = await readJson<RawConfig>(join(directory, "config.json"));
  const kind = options.kind ?? kindOf(id, config);
  // The library defaults: rerankers q8, embedding models fp32 (dtype unset).
  const dtype = options.dtype ?? (kind === "rerank" ? "q8" : "fp32");
  const missing = await missingModelFiles(directory, dtype);
  const catalog = findCatalogEntry(id);
  return {
    id,
    kind,
    source: local ? "local" : "hub",
    ...(catalog && { catalog }),
    profile: profileFor(id, kind),
    directory,
    cached: missing.length === 0,
    dtypes: await cachedDtypes(directory),
    missing,
    dtype,
    ...configFacts(config, kind, catalog?.dimension),
  };
}
