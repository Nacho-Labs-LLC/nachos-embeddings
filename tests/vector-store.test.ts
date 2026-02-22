import { describe, it, expect, beforeEach } from 'vitest';
import { VectorStore, cosineSimilarity, normalizeVector } from '../src/vector-store.js';

describe('cosineSimilarity', () => {
  it('returns 1 for identical vectors', () => {
    const v = [1, 0, 0];
    expect(cosineSimilarity(v, v)).toBeCloseTo(1, 5);
  });

  it('returns 0 for orthogonal vectors', () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0, 5);
  });

  it('returns -1 for opposite vectors', () => {
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1, 5);
  });

  it('handles normalized vectors correctly', () => {
    const a = normalizeVector([3, 4]);
    const b = normalizeVector([4, 3]);
    const sim = cosineSimilarity(a, b);
    expect(sim).toBeGreaterThan(0.9);
    expect(sim).toBeLessThan(1);
  });

  it('throws on dimension mismatch', () => {
    expect(() => cosineSimilarity([1, 2], [1, 2, 3])).toThrow('dimension mismatch');
  });

  it('returns 0 for empty vectors', () => {
    expect(cosineSimilarity([], [])).toBe(0);
  });

  it('returns 0 for zero vectors', () => {
    expect(cosineSimilarity([0, 0, 0], [1, 2, 3])).toBe(0);
  });
});

describe('normalizeVector', () => {
  it('normalizes to unit length', () => {
    const n = normalizeVector([3, 4]);
    const magnitude = Math.sqrt(n.reduce((s, v) => s + v * v, 0));
    expect(magnitude).toBeCloseTo(1, 5);
  });

  it('returns copy of zero vector', () => {
    const v = [0, 0, 0];
    const n = normalizeVector(v);
    expect(n).toEqual([0, 0, 0]);
    expect(n).not.toBe(v); // Should be a copy
  });

  it('preserves direction', () => {
    const v = [3, 4, 0];
    const n = normalizeVector(v);
    // Ratios should be preserved
    expect(n[0]! / n[1]!).toBeCloseTo(3 / 4, 5);
  });
});

describe('VectorStore', () => {
  let store: VectorStore<{ label: string }>;

  beforeEach(() => {
    store = new VectorStore({ minSimilarity: 0.5 });
  });

  it('adds and retrieves vectors', () => {
    store.add('v1', [1, 0, 0], { label: 'x-axis' });
    const entry = store.get('v1');
    expect(entry).toBeDefined();
    expect(entry!.id).toBe('v1');
    expect(entry!.vector).toEqual([1, 0, 0]);
    expect(entry!.metadata).toEqual({ label: 'x-axis' });
  });

  it('reports correct size', () => {
    expect(store.size()).toBe(0);
    store.add('v1', [1, 0, 0]);
    store.add('v2', [0, 1, 0]);
    expect(store.size()).toBe(2);
  });

  it('removes vectors', () => {
    store.add('v1', [1, 0, 0]);
    expect(store.remove('v1')).toBe(true);
    expect(store.get('v1')).toBeUndefined();
    expect(store.remove('nonexistent')).toBe(false);
  });

  it('clears all vectors', () => {
    store.add('v1', [1, 0, 0]);
    store.add('v2', [0, 1, 0]);
    store.clear();
    expect(store.size()).toBe(0);
  });

  it('lists keys', () => {
    store.add('a', [1, 0, 0]);
    store.add('b', [0, 1, 0]);
    expect(store.keys().sort()).toEqual(['a', 'b']);
  });

  describe('search', () => {
    beforeEach(() => {
      // Three orthogonal unit vectors
      store.add('x', [1, 0, 0], { label: 'x-axis' });
      store.add('y', [0, 1, 0], { label: 'y-axis' });
      store.add('z', [0, 0, 1], { label: 'z-axis' });
      // A vector close to x
      store.add('near-x', [0.95, 0.31, 0], { label: 'near-x' });
    });

    it('finds the most similar vector', () => {
      const results = store.search([1, 0, 0]);
      expect(results[0]!.id).toBe('x');
      expect(results[0]!.similarity).toBeCloseTo(1, 5);
    });

    it('ranks by similarity descending', () => {
      const results = store.search([0.9, 0.1, 0]);
      expect(results.length).toBeGreaterThan(0);
      for (let i = 1; i < results.length; i++) {
        expect(results[i]!.similarity).toBeLessThanOrEqual(results[i - 1]!.similarity);
      }
    });

    it('respects minSimilarity threshold', () => {
      const results = store.search([1, 0, 0], { minSimilarity: 0.9 });
      // Only 'x' and 'near-x' should pass 0.9 threshold
      expect(results.every((r) => r.similarity >= 0.9)).toBe(true);
    });

    it('respects limit', () => {
      const results = store.search([0.5, 0.5, 0.5], { limit: 2, minSimilarity: 0 });
      expect(results.length).toBe(2);
    });

    it('filters by metadata', () => {
      const results = store.search([1, 0, 0], {
        minSimilarity: 0,
        filter: (m) => m?.label === 'y-axis',
      });
      expect(results.length).toBe(1);
      expect(results[0]!.id).toBe('y');
    });
  });

  describe('addBatch', () => {
    it('adds multiple vectors at once', () => {
      store.addBatch([
        { id: 'a', vector: [1, 0, 0], metadata: { label: 'a' } },
        { id: 'b', vector: [0, 1, 0], metadata: { label: 'b' } },
      ]);
      expect(store.size()).toBe(2);
    });
  });

  describe('export / import', () => {
    it('round-trips data', () => {
      store.add('v1', [1, 0, 0], { label: 'hello' });
      store.add('v2', [0, 1, 0], { label: 'world' });

      const exported = store.export();
      expect(exported).toHaveLength(2);

      const newStore = new VectorStore<{ label: string }>();
      newStore.import(exported);
      expect(newStore.size()).toBe(2);
      expect(newStore.get('v1')!.metadata).toEqual({ label: 'hello' });
    });
  });
});
