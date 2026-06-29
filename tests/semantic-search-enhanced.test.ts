import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { EnhancedSemanticSearch } from '../src/semantic-search-enhanced.js';
import * as fs from 'node:fs';
import * as fsPromises from 'node:fs/promises';

vi.mock('node:fs', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:fs')>();
  return {
    ...actual,
    existsSync: vi.fn(),
  };
});

vi.mock('node:fs/promises', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:fs/promises')>();
  return {
    ...actual,
    readFile: vi.fn(),
    writeFile: vi.fn(),
    mkdir: vi.fn(),
  };
});

describe('EnhancedSemanticSearch', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('load()', () => {
    it('handles errors when loading semantic search store fails', async () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const storePath = '.dummy-store.json';
      vi.mocked(fs.existsSync).mockReturnValue(true);
      const testError = new Error('Test read error');
      vi.mocked(fsPromises.readFile).mockRejectedValue(testError);

      const search = new EnhancedSemanticSearch({
        verbose: true,
        storePath,
      });

      await search.load();

      expect(fs.existsSync).toHaveBeenCalledWith(storePath);
      expect(fsPromises.readFile).toHaveBeenCalledWith(storePath, 'utf-8');
      expect(warnSpy).toHaveBeenCalledWith(
        `[EnhancedSemanticSearch] Failed to load from ${storePath}:`,
        testError
      );
    });
  });
});
