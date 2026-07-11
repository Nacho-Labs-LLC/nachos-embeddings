import { describe, it, expect } from 'vitest';
import { SemanticSearch } from '../src/semantic-search.js';

describe('SemanticSearch', () => {
  it('is not initialized before init()', () => {
    const search = new SemanticSearch();
    expect(search.isInitialized()).toBe(false);
  });

  it('starts with zero documents', () => {
    const search = new SemanticSearch();
    expect(search.size()).toBe(0);
  });

  it('reads one document without exposing its vector', () => {
    const search = new SemanticSearch<{ tag: string }>();
    search.import([
      {
        id: 'doc1',
        text: 'hello',
        vector: [1, 0, 0],
        metadata: { tag: 'greeting' },
      },
    ]);

    expect(search.getDocument('doc1')).toEqual({
      id: 'doc1',
      text: 'hello',
      metadata: { tag: 'greeting' },
    });
    expect(search.getDocument('doc1')).not.toHaveProperty('vector');
  });

  it('returns undefined when reading a missing document', () => {
    const search = new SemanticSearch();

    expect(search.getDocument('missing')).toBeUndefined();
  });

  it('lists document IDs and metadata without exposing document text or vectors', () => {
    const search = new SemanticSearch<{ tag: string }>();
    search.import([
      { id: 'a', text: 'alpha', vector: [1, 0], metadata: { tag: 'first' } },
      { id: 'b', text: 'beta', vector: [0, 1], metadata: { tag: 'second' } },
    ]);

    const summaries = search.listDocuments();

    expect(summaries).toEqual([
      { id: 'a', metadata: { tag: 'first' } },
      { id: 'b', metadata: { tag: 'second' } },
    ]);
    expect(summaries[0]).not.toHaveProperty('text');
    expect(summaries[0]).not.toHaveProperty('vector');
  });

  it('removes documents by id', async () => {
    const search = new SemanticSearch();
    // Import pre-computed data to test remove without needing model
    search.import([
      { id: 'doc1', text: 'hello', vector: [1, 0, 0], metadata: undefined },
      { id: 'doc2', text: 'world', vector: [0, 1, 0], metadata: undefined },
    ]);
    expect(search.size()).toBe(2);
    expect(await search.remove('doc1')).toBe(true);
    expect(search.size()).toBe(1);
    expect(await search.remove('nonexistent')).toBe(false);
  });

  it('clears all documents', async () => {
    const search = new SemanticSearch();
    search.import([
      { id: 'doc1', text: 'hello', vector: [1, 0, 0], metadata: undefined },
      { id: 'doc2', text: 'world', vector: [0, 1, 0], metadata: undefined },
    ]);
    await search.clear();
    expect(search.size()).toBe(0);
  });

  it('exports and imports round-trip', () => {
    const search = new SemanticSearch<{ tag: string }>();
    const data = [
      { id: 'a', text: 'alpha', vector: [1, 0], metadata: { tag: 'first' } },
      { id: 'b', text: 'beta', vector: [0, 1], metadata: { tag: 'second' } },
    ];
    search.import(data);

    const exported = search.export();
    expect(exported).toHaveLength(2);
    expect(exported[0]!.text).toBe('alpha');
    expect(exported[1]!.metadata).toEqual({ tag: 'second' });

    const newSearch = new SemanticSearch<{ tag: string }>();
    newSearch.import(exported);
    expect(newSearch.size()).toBe(2);
  });
});
