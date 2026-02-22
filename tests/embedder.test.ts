import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { Embedder, getGlobalEmbedder, resetGlobalEmbedder } from '../src/embedder.js';

describe('Embedder', () => {
  it('is not initialized before init()', () => {
    const embedder = new Embedder();
    expect(embedder.isInitialized()).toBe(false);
  });

  it('throws when embed() called before init()', async () => {
    const embedder = new Embedder();
    await expect(embedder.embed('test')).rejects.toThrow('not initialized');
  });

  it('throws when embedBatch() called before init()', async () => {
    const embedder = new Embedder();
    await expect(embedder.embedBatch(['test'])).rejects.toThrow('not initialized');
  });

  it('returns null dimension when not initialized', async () => {
    const embedder = new Embedder();
    expect(await embedder.getDimension()).toBeNull();
  });

  it('uses default config when none provided', () => {
    const embedder = new Embedder();
    const config = embedder.getConfig();
    expect(config.model).toBe('Xenova/all-MiniLM-L6-v2');
    expect(config.cacheDir).toBe('.cache/transformers');
    expect(config.progressLogging).toBe(false);
  });

  it('accepts custom config', () => {
    const embedder = new Embedder({
      model: 'Xenova/all-mpnet-base-v2',
      cacheDir: '/tmp/models',
      progressLogging: true,
    });
    const config = embedder.getConfig();
    expect(config.model).toBe('Xenova/all-mpnet-base-v2');
    expect(config.cacheDir).toBe('/tmp/models');
    expect(config.progressLogging).toBe(true);
  });
});

describe('globalEmbedder', () => {
  afterEach(() => {
    resetGlobalEmbedder();
  });

  it('returns the same instance on repeated calls', () => {
    const a = getGlobalEmbedder();
    const b = getGlobalEmbedder();
    expect(a).toBe(b);
  });

  it('creates a new instance after reset', () => {
    const a = getGlobalEmbedder();
    resetGlobalEmbedder();
    const b = getGlobalEmbedder();
    expect(a).not.toBe(b);
  });
});
