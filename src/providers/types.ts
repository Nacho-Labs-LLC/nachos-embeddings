import type { InputType } from "./model-profiles.js";

/** Per-call embedding options. Providers without prefixes ignore them. */
export interface EmbedOptions {
  /**
   * `"query"` for search queries, `"document"` (default) for indexed text.
   * Models trained with asymmetric prefixes (e5, bge, nomic) use this.
   */
  type?: InputType;
}

export interface EmbeddingProvider {
  init(): Promise<void>;
  embed(text: string, options?: EmbedOptions): Promise<number[]>;
  embedBatch(texts: string[], options?: EmbedOptions): Promise<number[][]>;
  getDimension(): Promise<number | null>;
  isInitialized(): boolean;
  readonly name: string;
  /**
   * Optional: identifier of the model actually in use (including any variant
   * such as a quantization suffix). Consumers can persist it and re-embed
   * when it changes.
   */
  getModelId?(): string;
  /** Optional: release resources (model sessions, HTTP clients). */
  dispose?(): Promise<void>;
}

export interface BaseProviderConfig {
  progressLogging?: boolean;
}
