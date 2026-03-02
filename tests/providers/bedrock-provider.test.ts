import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { BedrockModelAdapter } from '../../src/providers/bedrock/models/types.js';

// Mock the AWS SDK
const mockSend = vi.fn();
vi.mock('@aws-sdk/client-bedrock-runtime', () => ({
  BedrockRuntimeClient: vi.fn().mockImplementation(() => ({
    send: mockSend,
  })),
  InvokeModelCommand: vi.fn().mockImplementation((input: unknown) => input),
}));

vi.mock('@aws-sdk/credential-provider-ini', () => ({
  fromIni: vi.fn().mockImplementation((opts: unknown) => opts),
}));

vi.mock('@aws-sdk/credential-providers', () => ({
  fromTemporaryCredentials: vi.fn().mockImplementation((opts: unknown) => opts),
}));

import { BedrockProvider } from '../../src/providers/bedrock/bedrock-provider.js';
import { BedrockRuntimeClient } from '@aws-sdk/client-bedrock-runtime';

function mockEmbeddingResponse(embedding: number[]): void {
  mockSend.mockResolvedValueOnce({
    body: new TextEncoder().encode(JSON.stringify({ embedding })),
  });
}

describe('BedrockProvider', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('has correct name', () => {
    const provider = new BedrockProvider();
    expect(provider.name).toBe('bedrock');
  });

  it('is not initialized before init()', () => {
    const provider = new BedrockProvider();
    expect(provider.isInitialized()).toBe(false);
  });

  it('returns null dimension before init()', async () => {
    const provider = new BedrockProvider();
    expect(await provider.getDimension()).toBeNull();
  });

  it('throws when embed() called before init()', async () => {
    const provider = new BedrockProvider();
    await expect(provider.embed('test')).rejects.toThrow('not initialized');
  });

  it('throws when embedBatch() called before init()', async () => {
    const provider = new BedrockProvider();
    await expect(provider.embedBatch(['test'])).rejects.toThrow('not initialized');
  });

  it('initializes with default config', async () => {
    // Probe dimension call during init
    mockEmbeddingResponse([0.1, 0.2, 0.3]);

    const provider = new BedrockProvider();
    await provider.init();

    expect(provider.isInitialized()).toBe(true);
    expect(await provider.getDimension()).toBe(3);
    expect(BedrockRuntimeClient).toHaveBeenCalledWith(
      expect.objectContaining({ region: 'us-east-1' }),
    );
  });

  it('skips re-initialization', async () => {
    mockEmbeddingResponse([0.1, 0.2]);

    const provider = new BedrockProvider();
    await provider.init();
    await provider.init(); // second call should be no-op

    // Only one probe embedding call
    expect(mockSend).toHaveBeenCalledTimes(1);
  });

  it('uses custom region', async () => {
    mockEmbeddingResponse([0.1]);

    const provider = new BedrockProvider({ region: 'eu-west-1' });
    await provider.init();

    expect(BedrockRuntimeClient).toHaveBeenCalledWith(
      expect.objectContaining({ region: 'eu-west-1' }),
    );
  });

  it('uses custom endpoint', async () => {
    mockEmbeddingResponse([0.1]);

    const provider = new BedrockProvider({ endpoint: 'http://localhost:4566' });
    await provider.init();

    expect(BedrockRuntimeClient).toHaveBeenCalledWith(
      expect.objectContaining({ endpoint: 'http://localhost:4566' }),
    );
  });

  it('embeds text using the model adapter', async () => {
    mockEmbeddingResponse([0.1, 0.2]); // init probe
    mockEmbeddingResponse([0.5, 0.6]); // actual embed

    const provider = new BedrockProvider();
    await provider.init();

    const result = await provider.embed('Hello world');
    expect(result).toEqual([0.5, 0.6]);
    expect(mockSend).toHaveBeenCalledTimes(2);
  });

  it('batch embeds multiple texts', async () => {
    mockEmbeddingResponse([0.1]); // init probe
    mockEmbeddingResponse([0.2]); // text 1
    mockEmbeddingResponse([0.3]); // text 2
    mockEmbeddingResponse([0.4]); // text 3

    const provider = new BedrockProvider({ maxConcurrency: 1 });
    await provider.init();

    const results = await provider.embedBatch(['a', 'b', 'c']);
    expect(results).toHaveLength(3);
    expect(results[0]).toEqual([0.2]);
    expect(results[1]).toEqual([0.3]);
    expect(results[2]).toEqual([0.4]);
  });

  it('uses explicit credentials', async () => {
    mockEmbeddingResponse([0.1]);

    const provider = new BedrockProvider({
      credentials: {
        strategy: 'explicit',
        accessKeyId: 'AKID',
        secretAccessKey: 'SECRET',
        sessionToken: 'TOKEN',
      },
    });
    await provider.init();

    expect(BedrockRuntimeClient).toHaveBeenCalledWith(
      expect.objectContaining({
        credentials: {
          accessKeyId: 'AKID',
          secretAccessKey: 'SECRET',
          sessionToken: 'TOKEN',
        },
      }),
    );
  });

  it('throws on explicit credentials without keys', async () => {
    const provider = new BedrockProvider({
      credentials: { strategy: 'explicit' },
    });

    await expect(provider.init()).rejects.toThrow('accessKeyId');
  });

  it('throws on role strategy without roleArn', async () => {
    const provider = new BedrockProvider({
      credentials: { strategy: 'role' },
    });

    await expect(provider.init()).rejects.toThrow('roleArn');
  });

  it('accepts custom model adapter', async () => {
    const customAdapter: BedrockModelAdapter = {
      modelName: 'Custom Model',
      defaultDimension: 128,
      formatRequest: (text) => JSON.stringify({ input: text }),
      parseResponse: (body) => {
        const parsed = JSON.parse(body) as { vector: number[] };
        return parsed.vector;
      },
    };

    mockSend.mockResolvedValueOnce({
      body: new TextEncoder().encode(JSON.stringify({ vector: [1, 2, 3] })),
    });

    const provider = new BedrockProvider({
      modelAdapter: customAdapter,
      modelId: 'custom.model-v1',
    });
    await provider.init();

    expect(provider.isInitialized()).toBe(true);
    expect(await provider.getDimension()).toBe(3);
  });

  it('retries on throttling errors', async () => {
    mockEmbeddingResponse([0.1]); // init probe

    // First call fails with throttling, second succeeds
    const throttleError = new Error('Rate exceeded');
    throttleError.name = 'ThrottlingException';
    mockSend
      .mockRejectedValueOnce(throttleError)
      .mockResolvedValueOnce({
        body: new TextEncoder().encode(JSON.stringify({ embedding: [0.5] })),
      });

    const provider = new BedrockProvider({
      retry: { maxAttempts: 3, backoffMs: 1 },
    });
    await provider.init();

    const result = await provider.embed('test');
    expect(result).toEqual([0.5]);
  });

  it('does not retry non-retryable errors', async () => {
    mockEmbeddingResponse([0.1]); // init probe

    const validationError = new Error('ValidationException');
    validationError.name = 'ValidationException';
    mockSend.mockRejectedValueOnce(validationError);

    const provider = new BedrockProvider({
      retry: { maxAttempts: 3, backoffMs: 1 },
    });
    await provider.init();

    await expect(provider.embed('test')).rejects.toThrow('ValidationException');
    // Only called twice: init probe + one failed attempt (no retry)
    expect(mockSend).toHaveBeenCalledTimes(2);
  });

  it('returns config', () => {
    const provider = new BedrockProvider({ region: 'ap-southeast-1' });
    const config = provider.getConfig();
    expect(config.region).toBe('ap-southeast-1');
    expect(config.modelId).toBe('amazon.titan-embed-text-v2:0');
    expect(config.retry.maxAttempts).toBe(3);
  });

  it('validates model options on construction', () => {
    expect(
      () => new BedrockProvider({ modelOptions: { dimensions: 999 } }),
    ).toThrow('Invalid dimensions');
  });

  it('throws for unknown model without custom adapter', () => {
    expect(
      () => new BedrockProvider({ modelId: 'unknown.model' }),
    ).toThrow('Unknown Bedrock model');
  });

  it('returns empty array for empty batch', async () => {
    mockEmbeddingResponse([0.1]); // init probe

    const provider = new BedrockProvider();
    await provider.init();

    const results = await provider.embedBatch([]);
    expect(results).toEqual([]);
    // Only the init probe call, no batch calls
    expect(mockSend).toHaveBeenCalledTimes(1);
  });

  it('preserves order in concurrent batch embedding', async () => {
    mockEmbeddingResponse([0.1]); // init probe

    // Mock responses with delays to test ordering
    // Text at index 0 should get [1], index 1 -> [2], etc.
    for (let i = 1; i <= 5; i++) {
      mockSend.mockResolvedValueOnce({
        body: new TextEncoder().encode(JSON.stringify({ embedding: [i] })),
      });
    }

    const provider = new BedrockProvider({ maxConcurrency: 3 });
    await provider.init();

    const results = await provider.embedBatch(['a', 'b', 'c', 'd', 'e']);
    expect(results).toHaveLength(5);
    expect(results[0]).toEqual([1]);
    expect(results[1]).toEqual([2]);
    expect(results[2]).toEqual([3]);
    expect(results[3]).toEqual([4]);
    expect(results[4]).toEqual([5]);
  });

  it('uses profile credential strategy', async () => {
    mockEmbeddingResponse([0.1]); // init probe
    const { fromIni } = await import('@aws-sdk/credential-provider-ini');

    const provider = new BedrockProvider({
      credentials: { strategy: 'profile', profile: 'my-profile' },
    });
    await provider.init();

    expect(fromIni).toHaveBeenCalledWith({ profile: 'my-profile' });
  });

  it('uses role credential strategy', async () => {
    mockEmbeddingResponse([0.1]); // init probe
    const { fromTemporaryCredentials } = await import('@aws-sdk/credential-providers');

    const provider = new BedrockProvider({
      credentials: {
        strategy: 'role',
        roleArn: 'arn:aws:iam::123:role/test',
        roleSessionName: 'test-session',
        externalId: 'ext-123',
      },
    });
    await provider.init();

    expect(fromTemporaryCredentials).toHaveBeenCalledWith(
      expect.objectContaining({
        params: expect.objectContaining({
          RoleArn: 'arn:aws:iam::123:role/test',
          RoleSessionName: 'test-session',
          ExternalId: 'ext-123',
        }),
      }),
    );
  });
});
