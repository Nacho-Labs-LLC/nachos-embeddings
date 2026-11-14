import type { EmbeddingProvider } from "../src/providers/types.js";

const DIM = 16;

function hash(word: string): number {
  let h = 0;
  for (const ch of word) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return h;
}

/** Deterministic bag-of-words embedding; no model download. */
export function fakeVector(text: string): number[] {
  const v = new Array<number>(DIM).fill(0);
  for (const w of text.toLowerCase().split(/\W+/).filter(Boolean)) {
    const i = hash(w) % DIM;
    v[i] = (v[i] ?? 0) + 1;
  }
  return v;
}

export class FakeProvider implements EmbeddingProvider {
  readonly name = "fake";
  embedCalls = 0;
  batchCalls = 0;
  initCalls = 0;
  delayMs = 0;
  private ready = false;

  async init(): Promise<void> {
    this.initCalls++;
    this.ready = true;
  }

  async embed(text: string): Promise<number[]> {
    this.embedCalls++;
    if (this.delayMs) await new Promise((r) => setTimeout(r, this.delayMs));
    return fakeVector(text);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    this.batchCalls++;
    if (this.delayMs) await new Promise((r) => setTimeout(r, this.delayMs));
    return texts.map(fakeVector);
  }

  async getDimension(): Promise<number | null> {
    return DIM;
  }

  isInitialized(): boolean {
    return this.ready;
  }
}
