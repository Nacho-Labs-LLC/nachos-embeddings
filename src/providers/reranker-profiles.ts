/**
 * Built-in per-model reranker conventions, matched on the model id
 * (case-insensitive), like `model-profiles.ts` does for embedding models.
 */

export interface RerankerProfile {
  /** Default truncation length (tokens, query + passage) for this model. */
  maxLength: number;
  /**
   * Longest input the model itself supports, when that exceeds the
   * tokenizer's `model_max_length`. Only models whose positions are not
   * learned (e.g. ALiBi) can be raised; for everything else this is
   * `undefined` and the tokenizer limit (512) applies.
   */
  maxPositions: number | undefined;
}

/** Fallback for models without a profile: the tokenizer limit applies. */
export const DEFAULT_RERANKER_PROFILE: RerankerProfile = {
  maxLength: 512,
  maxPositions: undefined,
};

const PROFILES: ReadonlyArray<{ match: RegExp; profile: RerankerProfile }> = [
  // ALiBi position biases: the ONNX graph accepts sequences far beyond the
  // tokenizer's 512. Verified up to 1024 here (scores for short inputs are
  // unchanged; a relevant sentence after token 512 is now seen). The default
  // is 1024 because memory bodies are usually a few paragraphs.
  {
    match: /jina-reranker-v1-tiny-en/,
    profile: { maxLength: 1024, maxPositions: 8192 },
  },
  // MiniLM / MS MARCO cross-encoders have learned absolute positions: the
  // ONNX graph fails above 512 tokens, so they stay at the default profile.
];

/** Built-in reranker profile for a model id (defaults when unknown). */
export function getRerankerProfile(modelId: string): RerankerProfile {
  const id = modelId.toLowerCase();
  const hit = PROFILES.find(({ match }) => match.test(id));
  return { ...(hit?.profile ?? DEFAULT_RERANKER_PROFILE) };
}

/** True for model ids that are known (or named like) cross-encoder rerankers. */
export function looksLikeReranker(modelId: string): boolean {
  return (
    PROFILES.some(({ match }) => match.test(modelId.toLowerCase())) ||
    /rerank|cross-encoder|ms-marco/i.test(modelId)
  );
}
