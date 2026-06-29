import { EnhancedSemanticSearch } from './src/semantic-search-enhanced.js';
import { Embedder } from './src/embedder.js';

// Mock the embedder to make the benchmark faster
class MockProvider {
  async init() {}
  async embed(text: string) { return [0.1, 0.2, 0.3]; }
  async embedBatch(texts: string[]) { return texts.map(() => [0.1, 0.2, 0.3]); }
  isInitialized() { return true; }
}

async function run() {
  const search = new EnhancedSemanticSearch({
    storePath: '.benchmark.json',
    autoSave: false,
    provider: new MockProvider(),
  });

  await search.init();

  console.log('Generating documents...');
  const docs = [];
  for (let i = 0; i < 10000; i++) {
    docs.push({
      id: `doc-${i}`,
      text: `This is the document number ${i} with some random text to make it unique ${Math.random()}`,
    });
  }
  await search.addDocuments(docs);

  console.log('Benchmarking remove...');
  const start = performance.now();
  for (let i = 0; i < 1000; i++) {
    await search.remove(`doc-${i}`);
  }
  const end = performance.now();

  console.log(`Time taken to remove 1000 documents: ${(end - start).toFixed(2)} ms`);
}

run().catch(console.error);
