/**
 * Built-in per-model embedding conventions: pooling and the query/document
 * prefixes a model was trained with. Matched by model id (case-insensitive),
 * so `Xenova/bge-small-en-v1.5` and `BAAI/bge-small-en-v1.5` both hit.
 * Unknown models (including the default MiniLM) get mean pooling, no prefixes.
 */

/** How token embeddings are reduced to one vector. */
export type PoolingMode = "mean" | "cls";

export const POOLING_MODES: readonly PoolingMode[] = ["mean", "cls"];

/** What a text is for: a search query, or a document/passage being indexed. */
export type InputType = "query" | "document";

export const INPUT_TYPES: readonly InputType[] = ["query", "document"];

export interface ModelProfile {
  pooling: PoolingMode;
  queryPrefix: string;
  documentPrefix: string;
}

const NO_PROFILE: ModelProfile = {
  pooling: "mean",
  queryPrefix: "",
  documentPrefix: "",
};

const PROFILES: ReadonlyArray<{ match: RegExp; profile: ModelProfile }> = [
  {
    match: /(^|[/_-])e5-/,
    profile: {
      pooling: "mean",
      queryPrefix: "query: ",
      documentPrefix: "passage: ",
    },
  },
  {
    match: /bge-.*-en-v1\.5/,
    profile: {
      pooling: "cls",
      queryPrefix: "Represent this sentence for searching relevant passages: ",
      documentPrefix: "",
    },
  },
  {
    match: /nomic-embed-text-v1(\.5)?(?![\d.])/,
    profile: {
      pooling: "mean",
      queryPrefix: "search_query: ",
      documentPrefix: "search_document: ",
    },
  },
];

/** Built-in profile for a model id (defaults when the model is not known). */
export function getModelProfile(modelId: string): ModelProfile {
  const id = modelId.toLowerCase();
  const hit = PROFILES.find(({ match }) => match.test(id));
  return { ...(hit?.profile ?? NO_PROFILE) };
}

export function assertInputType(type: unknown): asserts type is InputType {
  if (!(INPUT_TYPES as readonly unknown[]).includes(type)) {
    throw new RangeError(
      `Invalid input type '${String(type)}'. Expected one of: ${INPUT_TYPES.join(", ")}`,
    );
  }
}

export function assertPooling(
  pooling: unknown,
): asserts pooling is PoolingMode {
  if (!(POOLING_MODES as readonly unknown[]).includes(pooling)) {
    throw new RangeError(
      `Invalid pooling '${String(pooling)}'. Expected one of: ${POOLING_MODES.join(", ")}`,
    );
  }
}
