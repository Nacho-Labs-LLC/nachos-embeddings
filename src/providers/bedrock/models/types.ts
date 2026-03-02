export interface BedrockModelAdapter {
  /** Human-readable model name */
  readonly modelName: string;
  /** Default vector dimension for this model */
  readonly defaultDimension: number;
  /** Format the InvokeModel request body JSON string */
  formatRequest(text: string, options?: Record<string, unknown>): string;
  /** Parse the InvokeModel response body JSON string into a number[] vector */
  parseResponse(responseBody: string): number[];
  /** Optional: validate model-specific options, throw on invalid */
  validateOptions?(options: Record<string, unknown>): void;
}
