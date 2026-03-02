import { describe, it, expect, vi } from 'vitest';
import { createEmbedder } from '../../src/providers/factory.js';
import { TransformersProvider } from '../../src/providers/transformers-provider.js';

describe('createEmbedder', () => {
  it('creates a TransformersProvider for "transformers"', async () => {
    const provider = await createEmbedder('transformers');
    expect(provider).toBeInstanceOf(TransformersProvider);
    expect(provider.name).toBe('transformers');
  });

  it('passes config to TransformersProvider', async () => {
    const provider = await createEmbedder('transformers', {
      model: 'Xenova/all-mpnet-base-v2',
      progressLogging: true,
    });
    expect(provider.name).toBe('transformers');
  });

  it('creates a BedrockProvider for "bedrock"', async () => {
    const provider = await createEmbedder('bedrock', {
      region: 'us-west-2',
      modelId: 'amazon.titan-embed-text-v2:0',
    });
    expect(provider.name).toBe('bedrock');
  });

  it('throws for unknown provider type', async () => {
    await expect(
      createEmbedder('unknown' as any, {}),
    ).rejects.toThrow('Unknown provider type');
  });
});
