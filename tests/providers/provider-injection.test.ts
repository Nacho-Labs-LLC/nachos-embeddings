import { describe, it, expect, vi } from 'vitest';
import { SemanticSearch } from '../../src/semantic-search.js';
import type { EmbeddingProvider } from '../../src/providers/types.js';

function createMockProvider(dimension = 3): EmbeddingProvider {
  const vector = Array.from({ length: dimension }, (_, i) => (i + 1) * 0.1);
  return {
    name: 'mock',
    init: vi.fn().mockResolvedValue(undefined),
    embed: vi.fn().mockResolvedValue(vector),
    embedBatch: vi.fn().mockImplementation((texts: string[]) =>
      Promise.resolve(texts.map(() => [...vector])),
    ),
    getDimension: vi.fn().mockResolvedValue(dimension),
    isInitialized: vi.fn().mockReturnValue(true),
  };
}

describe('SemanticSearch with custom provider', () => {
  it('accepts an EmbeddingProvider instance', async () => {
    const mockProvider = createMockProvider();

    const search = new SemanticSearch({ provider: mockProvider });
    await search.init();

    expect(mockProvider.init).toHaveBeenCalled();
  });

  it('uses the custom provider for embedding', async () => {
    const mockProvider = createMockProvider();

    const search = new SemanticSearch({ provider: mockProvider });
    await search.init();

    await search.addDocument({ id: 'doc1', text: 'hello world' });
    expect(mockProvider.embed).toHaveBeenCalledWith('hello world');
  });

  it('uses the custom provider for search', async () => {
    const mockProvider = createMockProvider();

    const search = new SemanticSearch({
      provider: mockProvider,
      minSimilarity: 0.0, // allow all results
    });
    await search.init();

    await search.addDocument({ id: 'doc1', text: 'hello' });
    const results = await search.search('query');

    expect(mockProvider.embed).toHaveBeenCalledWith('query');
    expect(results).toHaveLength(1);
    expect(results[0]!.id).toBe('doc1');
  });

  it('uses custom provider for batch operations', async () => {
    const mockProvider = createMockProvider();

    const search = new SemanticSearch({ provider: mockProvider });
    await search.init();

    await search.addDocuments([
      { id: 'a', text: 'alpha' },
      { id: 'b', text: 'beta' },
    ]);

    expect(mockProvider.embedBatch).toHaveBeenCalledWith(['alpha', 'beta']);
    expect(search.size()).toBe(2);
  });

  it('reports initialization from custom provider', () => {
    const mockProvider = createMockProvider();
    (mockProvider.isInitialized as ReturnType<typeof vi.fn>).mockReturnValue(false);

    const search = new SemanticSearch({ provider: mockProvider });
    expect(search.isInitialized()).toBe(false);
  });

  it('falls back to Embedder when no provider given', () => {
    // Should not throw — creates default Embedder internally
    const search = new SemanticSearch();
    expect(search.isInitialized()).toBe(false);
  });

  it('falls back to Embedder when provider is undefined', () => {
    const search = new SemanticSearch({ provider: undefined });
    expect(search.isInitialized()).toBe(false);
  });
});
