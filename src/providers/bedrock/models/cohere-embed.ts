import type { BedrockModelAdapter } from './types.js';

const VALID_INPUT_TYPES = [
  'search_document',
  'search_query',
  'classification',
  'clustering',
] as const;
type CohereInputType = (typeof VALID_INPUT_TYPES)[number];

const VALID_TRUNCATE = ['NONE', 'START', 'END'] as const;
type CohereTruncate = (typeof VALID_TRUNCATE)[number];

interface CohereRequest {
  texts: string[];
  input_type: string;
  truncate?: string;
}

interface CohereResponse {
  embeddings: number[][];
  id: string;
  response_type: string;
  texts: string[];
}

export class CohereEmbedAdapter implements BedrockModelAdapter {
  readonly modelName = 'Cohere Embed v3';
  readonly defaultDimension = 1024;

  formatRequest(text: string, options?: Record<string, unknown>): string {
    const inputType = (options?.['inputType'] as string | undefined) ?? 'search_document';
    const truncate = (options?.['truncate'] as string | undefined) ?? 'END';

    const request: CohereRequest = {
      texts: [text],
      input_type: inputType,
      truncate,
    };

    return JSON.stringify(request);
  }

  parseResponse(responseBody: string): number[] {
    const parsed = JSON.parse(responseBody) as CohereResponse;
    const firstEmbedding = parsed.embeddings[0];
    if (!firstEmbedding) {
      throw new Error('Cohere response contained no embeddings');
    }
    return firstEmbedding;
  }

  validateOptions(options: Record<string, unknown>): void {
    const inputType = options['inputType'];
    if (inputType !== undefined) {
      if (
        typeof inputType !== 'string' ||
        !VALID_INPUT_TYPES.includes(inputType as CohereInputType)
      ) {
        throw new Error(
          `Invalid inputType for Cohere Embed: ${String(inputType)}. ` +
            `Valid values: ${VALID_INPUT_TYPES.join(', ')}`,
        );
      }
    }

    const truncate = options['truncate'];
    if (truncate !== undefined) {
      if (
        typeof truncate !== 'string' ||
        !VALID_TRUNCATE.includes(truncate as CohereTruncate)
      ) {
        throw new Error(
          `Invalid truncate for Cohere Embed: ${String(truncate)}. ` +
            `Valid values: ${VALID_TRUNCATE.join(', ')}`,
        );
      }
    }
  }
}
