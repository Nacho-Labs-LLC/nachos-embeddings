/**
 * Curated catalogue of models this package has tested, with the facts a user
 * needs to choose one: vector size, licence, download sizes per dtype, and the
 * built-in pooling/prefix/length profile that is applied by default.
 *
 * Any other Hugging Face ONNX encoder or cross-encoder (or a local directory)
 * also works; this list is just the tested set. Sizes are the ONNX file sizes
 * on the hub (checked 2026-11), in MB (10^6 bytes). The licence is the one the
 * model's authors publish.
 */

import { getModelProfile } from "./model-profiles.js";
import { getRerankerProfile } from "./reranker-profiles.js";
import type { TransformersDtype } from "./transformers-provider.js";

export type ModelKind = "embedding" | "rerank";

export interface CatalogEntry {
  id: string;
  kind: ModelKind;
  /** Output vector size (embedding models). */
  dimension?: number;
  license: string;
  /** ONNX download size in MB per dtype available on the hub. */
  sizesMB: Partial<Record<TransformersDtype, number>>;
  notes: string;
}

/** Download sizes (MB) of a Xenova-style export: int8 and uint8 match q8. */
interface Sizes {
  fp32: number;
  fp16: number;
  q8: number;
  q4: number;
  q4f16?: number;
  bnb4?: number;
}

const dtypes = ({
  fp32,
  fp16,
  q8,
  q4,
  q4f16,
  bnb4,
}: Sizes): CatalogEntry["sizesMB"] => ({
  fp32,
  fp16,
  q8,
  int8: q8,
  uint8: q8,
  q4,
  ...(q4f16 !== undefined && { q4f16 }),
  ...(bnb4 !== undefined && { bnb4 }),
});

export const MODEL_CATALOG: readonly CatalogEntry[] = [
  {
    id: "Xenova/all-MiniLM-L6-v2",
    kind: "embedding",
    dimension: 384,
    license: "Apache-2.0",
    sizesMB: dtypes({
      fp32: 90.4,
      fp16: 45.3,
      q8: 23.0,
      q4: 54.6,
      q4f16: 30.0,
      bnb4: 53.9,
    }),
    notes: "Default. Smallest and fastest; no prefixes.",
  },
  {
    id: "Xenova/bge-small-en-v1.5",
    kind: "embedding",
    dimension: 384,
    license: "MIT",
    sizesMB: dtypes({
      fp32: 133.1,
      fp16: 66.7,
      q8: 34.0,
      q4: 61.5,
      q4f16: 36.2,
      bnb4: 60.1,
    }),
    notes: "cls pooling, query instruction. English.",
  },
  {
    id: "Xenova/e5-small-v2",
    kind: "embedding",
    dimension: 384,
    license: "MIT",
    sizesMB: dtypes({
      fp32: 133.1,
      fp16: 66.7,
      q8: 34.0,
      q4: 61.5,
      q4f16: 36.2,
      bnb4: 60.1,
    }),
    notes: "`query: ` / `passage: ` prefixes. English.",
  },
  {
    id: "Snowflake/snowflake-arctic-embed-s",
    kind: "embedding",
    dimension: 384,
    license: "Apache-2.0",
    sizesMB: dtypes({
      fp32: 133.1,
      fp16: 66.7,
      q8: 34.0,
      q4: 61.4,
      bnb4: 60.1,
    }),
    notes: "cls pooling, query instruction; needs the query type. English.",
  },
  {
    id: "nomic-ai/nomic-embed-text-v1.5",
    kind: "embedding",
    dimension: 768,
    license: "Apache-2.0",
    sizesMB: dtypes({
      fp32: 547.3,
      fp16: 273.9,
      q8: 137.3,
      q4: 165.1,
      q4f16: 111.1,
      bnb4: 158.0,
    }),
    notes: "`search_query: ` / `search_document: ` prefixes, layer norm.",
  },
  {
    id: "Alibaba-NLP/gte-modernbert-base",
    kind: "embedding",
    dimension: 768,
    license: "Apache-2.0",
    sizesMB: dtypes({
      fp32: 596.4,
      fp16: 298.4,
      q8: 150.2,
      q4: 224.2,
      q4f16: 139.8,
      bnb4: 217.3,
    }),
    notes:
      "cls pooling, no prefixes, long context. Best recall in the benchmark.",
  },
  {
    id: "jinaai/jina-reranker-v1-tiny-en",
    kind: "rerank",
    license: "Apache-2.0",
    sizesMB: dtypes({
      fp32: 132.4,
      fp16: 66.3,
      q8: 33.4,
      q4: 100.5,
      bnb4: 99.9,
    }),
    notes:
      "Default reranker. 4 layers, 8192 positions (ALiBi); maxLength defaults to 1024.",
  },
  {
    id: "Xenova/ms-marco-MiniLM-L-6-v2",
    kind: "rerank",
    license: "Apache-2.0",
    sizesMB: dtypes({
      fp32: 91.0,
      fp16: 45.6,
      q8: 23.1,
      q4: 55.2,
      q4f16: 30.3,
      bnb4: 54.5,
    }),
    notes: "Fallback reranker. 512 tokens maximum.",
  },
];

/** What the package applies to a model unless overridden. */
export interface EffectiveProfile {
  pooling?: string;
  queryPrefix?: string;
  documentPrefix?: string;
  layerNorm?: boolean;
  maxLength?: number;
  maxPositions?: number;
}

/** Built-in profile for a model id of either kind. */
export function profileFor(id: string, kind: ModelKind): EffectiveProfile {
  if (kind === "rerank") {
    const profile = getRerankerProfile(id);
    return {
      maxLength: profile.maxLength,
      ...(profile.maxPositions !== undefined && {
        maxPositions: profile.maxPositions,
      }),
    };
  }
  const { pooling, queryPrefix, documentPrefix, layerNorm } =
    getModelProfile(id);
  return { pooling, queryPrefix, documentPrefix, layerNorm };
}

/** Catalogue entry for an id (case-insensitive exact match). */
export function findCatalogEntry(id: string): CatalogEntry | undefined {
  const wanted = id.toLowerCase();
  return MODEL_CATALOG.find((entry) => entry.id.toLowerCase() === wanted);
}
