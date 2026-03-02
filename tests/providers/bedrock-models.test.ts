import { describe, it, expect } from 'vitest';
import { TitanV2Adapter } from '../../src/providers/bedrock/models/titan-v2.js';
import { CohereEmbedAdapter } from '../../src/providers/bedrock/models/cohere-embed.js';
import { resolveModelAdapter } from '../../src/providers/bedrock/models/index.js';

describe('TitanV2Adapter', () => {
  const adapter = new TitanV2Adapter();

  it('has correct model metadata', () => {
    expect(adapter.modelName).toBe('Amazon Titan Text Embeddings V2');
    expect(adapter.defaultDimension).toBe(1024);
  });

  it('formats request with defaults', () => {
    const body = JSON.parse(adapter.formatRequest('Hello world'));
    expect(body).toEqual({
      inputText: 'Hello world',
      normalize: true,
    });
  });

  it('formats request with custom dimensions', () => {
    const body = JSON.parse(adapter.formatRequest('Hello', { dimensions: 256 }));
    expect(body.inputText).toBe('Hello');
    expect(body.dimensions).toBe(256);
    expect(body.normalize).toBe(true);
  });

  it('formats request with normalize disabled', () => {
    const body = JSON.parse(adapter.formatRequest('Hello', { normalize: false }));
    expect(body.normalize).toBe(false);
  });

  it('parses response correctly', () => {
    const response = JSON.stringify({
      embedding: [0.1, 0.2, 0.3, 0.4],
      inputTextTokenCount: 2,
    });
    expect(adapter.parseResponse(response)).toEqual([0.1, 0.2, 0.3, 0.4]);
  });

  it('validates valid dimensions', () => {
    expect(() => adapter.validateOptions({ dimensions: 256 })).not.toThrow();
    expect(() => adapter.validateOptions({ dimensions: 512 })).not.toThrow();
    expect(() => adapter.validateOptions({ dimensions: 1024 })).not.toThrow();
  });

  it('rejects invalid dimensions', () => {
    expect(() => adapter.validateOptions({ dimensions: 128 })).toThrow('Invalid dimensions');
    expect(() => adapter.validateOptions({ dimensions: 999 })).toThrow('Invalid dimensions');
    expect(() => adapter.validateOptions({ dimensions: 'big' })).toThrow('Invalid dimensions');
  });

  it('rejects non-boolean normalize', () => {
    expect(() => adapter.validateOptions({ normalize: 'yes' })).toThrow('Invalid normalize');
  });

  it('accepts empty options', () => {
    expect(() => adapter.validateOptions({})).not.toThrow();
  });
});

describe('CohereEmbedAdapter', () => {
  const adapter = new CohereEmbedAdapter();

  it('has correct model metadata', () => {
    expect(adapter.modelName).toBe('Cohere Embed v3');
    expect(adapter.defaultDimension).toBe(1024);
  });

  it('formats request with defaults', () => {
    const body = JSON.parse(adapter.formatRequest('Hello world'));
    expect(body).toEqual({
      texts: ['Hello world'],
      input_type: 'search_document',
      truncate: 'END',
    });
  });

  it('formats request with custom input type', () => {
    const body = JSON.parse(adapter.formatRequest('query text', { inputType: 'search_query' }));
    expect(body.input_type).toBe('search_query');
  });

  it('formats request with custom truncate', () => {
    const body = JSON.parse(adapter.formatRequest('text', { truncate: 'NONE' }));
    expect(body.truncate).toBe('NONE');
  });

  it('parses response correctly', () => {
    const response = JSON.stringify({
      embeddings: [[0.5, 0.6, 0.7]],
      id: 'test-id',
      response_type: 'embeddings_floats',
      texts: ['Hello'],
    });
    expect(adapter.parseResponse(response)).toEqual([0.5, 0.6, 0.7]);
  });

  it('throws on empty embeddings response', () => {
    const response = JSON.stringify({
      embeddings: [],
      id: 'test-id',
      response_type: 'embeddings_floats',
      texts: [],
    });
    expect(() => adapter.parseResponse(response)).toThrow('no embeddings');
  });

  it('validates valid input types', () => {
    expect(() => adapter.validateOptions({ inputType: 'search_document' })).not.toThrow();
    expect(() => adapter.validateOptions({ inputType: 'search_query' })).not.toThrow();
    expect(() => adapter.validateOptions({ inputType: 'classification' })).not.toThrow();
    expect(() => adapter.validateOptions({ inputType: 'clustering' })).not.toThrow();
  });

  it('rejects invalid input type', () => {
    expect(() => adapter.validateOptions({ inputType: 'invalid' })).toThrow('Invalid inputType');
  });

  it('validates valid truncate values', () => {
    expect(() => adapter.validateOptions({ truncate: 'NONE' })).not.toThrow();
    expect(() => adapter.validateOptions({ truncate: 'START' })).not.toThrow();
    expect(() => adapter.validateOptions({ truncate: 'END' })).not.toThrow();
  });

  it('rejects invalid truncate', () => {
    expect(() => adapter.validateOptions({ truncate: 'MIDDLE' })).toThrow('Invalid truncate');
  });
});

describe('resolveModelAdapter', () => {
  it('resolves titan v2', () => {
    const adapter = resolveModelAdapter('amazon.titan-embed-text-v2:0');
    expect(adapter.modelName).toBe('Amazon Titan Text Embeddings V2');
  });

  it('resolves cohere english', () => {
    const adapter = resolveModelAdapter('cohere.embed-english-v3');
    expect(adapter.modelName).toBe('Cohere Embed v3');
  });

  it('resolves cohere multilingual', () => {
    const adapter = resolveModelAdapter('cohere.embed-multilingual-v3');
    expect(adapter.modelName).toBe('Cohere Embed v3');
  });

  it('throws for unknown model', () => {
    expect(() => resolveModelAdapter('unknown.model')).toThrow('Unknown Bedrock model');
    expect(() => resolveModelAdapter('unknown.model')).toThrow('unknown.model');
  });

  it('lists supported models in error message', () => {
    expect(() => resolveModelAdapter('bad')).toThrow('amazon.titan-embed-text-v2:0');
    expect(() => resolveModelAdapter('bad')).toThrow('cohere.embed-english-v3');
  });
});
