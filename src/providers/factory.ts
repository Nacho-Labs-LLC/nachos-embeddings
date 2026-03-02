import type { EmbeddingProvider } from './types.js';
import type { TransformersProviderConfig } from './transformers-provider.js';
import type { BedrockProviderConfig } from './bedrock/bedrock-provider.js';

export type ProviderType = 'transformers' | 'bedrock';

export type ProviderConfigMap = {
  transformers: TransformersProviderConfig;
  bedrock: BedrockProviderConfig;
};

export async function createEmbedder<T extends ProviderType>(
  type: T,
  config?: ProviderConfigMap[T]
): Promise<EmbeddingProvider> {
  switch (type) {
    case 'transformers': {
      const { TransformersProvider } = await import('./transformers-provider.js');
      return new TransformersProvider(config as TransformersProviderConfig);
    }
    case 'bedrock': {
      // Dynamic import avoids pulling in @aws-sdk/client-bedrock-runtime at module level.
      // The variable prevents TypeScript from statically resolving the specifier.
      const specifier = './bedrock/bedrock-provider.js';
      const mod: any = await import(specifier);
      return new mod.BedrockProvider(config as BedrockProviderConfig) as EmbeddingProvider;
    }
    default:
      throw new Error(`Unknown provider type: "${type as string}". Supported: transformers, bedrock`);
  }
}
