export interface EmbeddingProvider {
  init(): Promise<void>;
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
  getDimension(): Promise<number | null>;
  isInitialized(): boolean;
  readonly name: string;
}

export interface BaseProviderConfig {
  progressLogging?: boolean;
}
