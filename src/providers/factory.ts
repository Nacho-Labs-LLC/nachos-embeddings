import type { EmbeddingProvider } from './types.js';
import type { TransformersProviderConfig } from './transformers-provider.js';

export type ProviderType = 'transformers' | 'bedrock';

export type ProviderConfigMap = {
  transformers: TransformersProviderConfig;
  bedrock: Record<string, unknown>;
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
      // Use a variable to prevent TypeScript from statically resolving the module.
      // The bedrock provider is loaded on demand to avoid pulling in
      // @aws-sdk/client-bedrock-runtime at module level.
      const specifier = './bedrock/bedrock-provider.js';
      const mod: any = await import(specifier);
      return new mod.BedrockProvider(config) as EmbeddingProvider;
    }
    default:
      throw new Error(`Unknown provider type: "${type as string}". Supported: transformers, bedrock`);
  }
}
