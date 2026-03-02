import { TitanV2Adapter } from './titan-v2.js';
import { CohereEmbedAdapter } from './cohere-embed.js';
import type { BedrockModelAdapter } from './types.js';

const BUILTIN_ADAPTERS: Record<string, () => BedrockModelAdapter> = {
  'amazon.titan-embed-text-v2:0': () => new TitanV2Adapter(),
  'cohere.embed-english-v3': () => new CohereEmbedAdapter(),
  'cohere.embed-multilingual-v3': () => new CohereEmbedAdapter(),
};

export function resolveModelAdapter(modelId: string): BedrockModelAdapter {
  const factory = BUILTIN_ADAPTERS[modelId];
  if (!factory) {
    throw new Error(
      `Unknown Bedrock model: "${modelId}". ` +
        `Supported: ${Object.keys(BUILTIN_ADAPTERS).join(', ')}. ` +
        `For custom models, provide a modelAdapter in config.`,
    );
  }
  return factory();
}

export { TitanV2Adapter } from './titan-v2.js';
export { CohereEmbedAdapter } from './cohere-embed.js';
export type { BedrockModelAdapter } from './types.js';
