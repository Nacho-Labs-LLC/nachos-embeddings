import type { BedrockModelAdapter } from './types.js';

const VALID_DIMENSIONS = [256, 512, 1024] as const;
type TitanDimension = (typeof VALID_DIMENSIONS)[number];

interface TitanRequest {
  inputText: string;
  dimensions?: TitanDimension;
  normalize?: boolean;
}

interface TitanResponse {
  embedding: number[];
  inputTextTokenCount: number;
}

export class TitanV2Adapter implements BedrockModelAdapter {
  readonly modelName = 'Amazon Titan Text Embeddings V2';
  readonly defaultDimension = 1024;

  formatRequest(text: string, options?: Record<string, unknown>): string {
    const request: TitanRequest = {
      inputText: text,
    };

    const dimensions = options?.['dimensions'];
    if (dimensions !== undefined) {
      request.dimensions = dimensions as TitanDimension;
    }

    const normalize = options?.['normalize'];
    if (normalize !== undefined) {
      request.normalize = normalize as boolean;
    } else {
      request.normalize = true;
    }

    return JSON.stringify(request);
  }

  parseResponse(responseBody: string): number[] {
    const parsed = JSON.parse(responseBody) as TitanResponse;
    return parsed.embedding;
  }

  validateOptions(options: Record<string, unknown>): void {
    const dimensions = options['dimensions'];
    if (dimensions !== undefined) {
      if (
        typeof dimensions !== 'number' ||
        !VALID_DIMENSIONS.includes(dimensions as TitanDimension)
      ) {
        throw new Error(
          `Invalid dimensions for Titan V2: ${String(dimensions)}. ` +
            `Valid values: ${VALID_DIMENSIONS.join(', ')}`,
        );
      }
    }

    const normalize = options['normalize'];
    if (normalize !== undefined && typeof normalize !== 'boolean') {
      throw new Error(
        `Invalid normalize for Titan V2: expected boolean, got ${typeof normalize}`,
      );
    }
  }
}
