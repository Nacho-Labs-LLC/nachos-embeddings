export interface EmbeddingProvider {
  init(): Promise<void>;
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
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
