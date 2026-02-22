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

  it('removes documents by id', () => {
    const search = new SemanticSearch();
    // Import pre-computed data to test remove without needing model
    search.import([
      { id: 'doc1', text: 'hello', vector: [1, 0, 0], metadata: undefined },
      { id: 'doc2', text: 'world', vector: [0, 1, 0], metadata: undefined },
    ]);
    expect(search.size()).toBe(2);
    expect(search.remove('doc1')).toBe(true);
    expect(search.size()).toBe(1);
    expect(search.remove('nonexistent')).toBe(false);
  });

  it('clears all documents', () => {
    const search = new SemanticSearch();
    search.import([
      { id: 'doc1', text: 'hello', vector: [1, 0, 0], metadata: undefined },
      { id: 'doc2', text: 'world', vector: [0, 1, 0], metadata: undefined },
    ]);
    search.clear();
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
