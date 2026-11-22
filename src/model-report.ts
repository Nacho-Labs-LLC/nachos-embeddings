/** Data and text for the `models` and `info <model>` commands. */

import {
  MODEL_CATALOG,
  profileFor,
  type CatalogEntry,
  type EffectiveProfile,
} from "./providers/model-catalog.js";
import {
  listCachedModels,
  type CachedModel,
  type ModelInfo,
} from "./model-cache.js";

export interface BuiltinModel extends CatalogEntry {
  profile: EffectiveProfile;
  /** dtypes of this model that are in the cache. */
  cachedDtypes: string[];
}

export interface ModelsReport {
  cacheDir: string;
  builtin: BuiltinModel[];
  /** Everything in the cache, including models outside the catalogue. */
  cached: CachedModel[];
}

export async function buildModelsReport(
  cacheDir: string,
): Promise<ModelsReport> {
  const cached = await listCachedModels(cacheDir);
  const builtin = MODEL_CATALOG.map((entry) => ({
    ...entry,
    profile: profileFor(entry.id, entry.kind),
    cachedDtypes:
      cached.find((c) => c.id.toLowerCase() === entry.id.toLowerCase())
        ?.dtypes ?? [],
  }));
  return { cacheDir, builtin, cached };
}

const mb = (bytes: number) => `${(bytes / 1e6).toFixed(1)} MB`;

function describeProfile(profile: EffectiveProfile): string {
  const parts: string[] = [];
  if (profile.pooling) parts.push(`${profile.pooling} pooling`);
  if (profile.queryPrefix)
    parts.push(`query prefix ${JSON.stringify(profile.queryPrefix)}`);
  if (profile.documentPrefix)
    parts.push(`document prefix ${JSON.stringify(profile.documentPrefix)}`);
  if (profile.layerNorm) parts.push("layer norm");
  if (profile.maxLength !== undefined) {
    parts.push(
      `max length ${profile.maxLength}` +
        (profile.maxPositions ? ` (up to ${profile.maxPositions})` : ""),
    );
  }
  return parts.join(", ") || "no prefixes";
}

export function formatModelsText(report: ModelsReport): string {
  const lines = [
    "Built-in profiles (tested models; any ONNX encoder or a local directory also works):",
    "",
  ];
  for (const model of report.builtin) {
    const sizes = Object.entries(model.sizesMB)
      .map(([dtype, size]) => `${dtype} ${size} MB`)
      .join(", ");
    const cached = model.cachedDtypes.length
      ? `  [cached: ${model.cachedDtypes.join(", ")}]`
      : "";
    lines.push(
      `${model.id}  (${model.kind}${model.dimension ? `, ${model.dimension} dims` : ""}, ${model.license})${cached}`,
      `    ${describeProfile(model.profile)}`,
      `    downloads: ${sizes}`,
      `    ${model.notes}`,
    );
  }
  lines.push("", `Cached models in ${report.cacheDir}:`, "");
  if (report.cached.length === 0) lines.push("    (none)");
  for (const model of report.cached) {
    lines.push(
      `${model.id}  (${model.kind}${model.dimension ? `, ${model.dimension} dims` : ""}) ` +
        `${model.dtypes.join(", ") || "no weights"}  ${mb(model.bytes)}` +
        (model.complete ? "" : "  [incomplete]"),
    );
  }
  return lines.join("\n") + "\n";
}

export function formatInfoText(info: ModelInfo): string {
  const lines = [
    `model: ${info.id}`,
    `kind: ${info.kind}`,
    `source: ${info.source === "local" ? "local directory" : "Hugging Face hub (cache)"}`,
    `directory: ${info.directory}`,
    `status: ${info.cached ? `ready (${info.dtype})` : `not usable offline for dtype ${info.dtype}: missing ${info.missing.join(", ")}`}`,
    `cached dtypes: ${info.dtypes.join(", ") || "-"}`,
  ];
  if (info.modelType) lines.push(`architecture: ${info.modelType}`);
  if (info.dimension) lines.push(`dimension: ${info.dimension}`);
  if (info.maxPositions) lines.push(`max positions: ${info.maxPositions}`);
  lines.push(`profile: ${describeProfile(info.profile)}`);
  if (info.catalog) {
    lines.push(
      `license: ${info.catalog.license}`,
      `notes: ${info.catalog.notes}`,
    );
  } else {
    lines.push(
      "not in the built-in list: defaults apply (mean pooling, no prefixes); override with --pooling / --query-prefix / --document-prefix",
    );
  }
  return lines.join("\n") + "\n";
}
