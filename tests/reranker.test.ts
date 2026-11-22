import { beforeEach, describe, expect, it, vi } from "vitest";

/**
 * Fake model: the logit of a pair is the number of query words that appear in
 * the passage, minus the length in words / 100. Tokenizer calls and model
 * batches are recorded so batching and ordering can be asserted.
 */
const calls = vi.hoisted(() => ({
  tokenizer: [] as { texts: string[]; options: Record<string, unknown> }[],
  loads: [] as { id: string; options: Record<string, unknown> }[],
  tokenizers: [] as { _tokenizerConfig: { model_max_length: number } }[],
  disposed: 0,
}));

vi.mock("@huggingface/transformers", () => {
  const makeTokenizer = () =>
    Object.assign(
      (texts: string[], options: Record<string, unknown>) => {
        calls.tokenizer.push({ texts, options });
        return {
          pairs: texts.map((q, i) => [
            q,
            (options["text_pair"] as string[])[i],
          ]),
        };
      },
      { _tokenizerConfig: { model_max_length: 512 } },
    );
  return {
    env: {
      allowLocalModels: true,
      allowRemoteModels: true,
      localModelPath: "./models/",
    },
    AutoTokenizer: {
      from_pretrained: (id: string, options: Record<string, unknown>) => {
        calls.loads.push({ id, options });
        const tokenizer = makeTokenizer();
        calls.tokenizers.push(tokenizer);
        return Promise.resolve(tokenizer);
      },
    },
    AutoModelForSequenceClassification: {
      from_pretrained: (id: string, options: Record<string, unknown>) => {
        calls.loads.push({ id, options });
        const model = Object.assign(
          (inputs: { pairs: [string, string][] }) => {
            const data = Float32Array.from(inputs.pairs, ([q, p]) => {
              const words = p.split(/\s+/).filter(Boolean);
              const hits = q
                .split(/\s+/)
                .filter((w) => words.includes(w)).length;
              return hits * 4 - 2 - words.length / 100;
            });
            return Promise.resolve({
              logits: { data, dims: [data.length, 1] },
            });
          },
          {
            dispose: () => {
              calls.disposed++;
              return Promise.resolve();
            },
          },
        );
        return Promise.resolve(model);
      },
    },
  };
});

import {
  Reranker,
  rerank,
  disposeRerankers,
  RERANK_DEFAULT_MODEL,
  RERANK_FALLBACK_MODEL,
} from "../src/reranker.js";

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

beforeEach(async () => {
  await disposeRerankers();
  calls.tokenizer.length = 0;
  calls.loads.length = 0;
  calls.tokenizers.length = 0;
  calls.disposed = 0;
});

describe("Reranker", () => {
  it("returns sigmoid scores in input order, relevant passage highest", async () => {
    const reranker = new Reranker({ cacheDir: "/tmp/x" });
    const scores = await reranker.rerank("red apple pie", [
      "the weather is nice",
      "red apple pie recipe",
      "an apple",
    ]);
    expect(scores).toHaveLength(3);
    expect(scores[1]).toBeGreaterThan(scores[2] as number);
    expect(scores[2]).toBeGreaterThan(scores[0] as number);
    expect(scores[1]).toBeCloseTo(sigmoid(3 * 4 - 2 - 4 / 100), 6);
    for (const s of scores) expect(s).toBeGreaterThan(0);
  });

  it("keeps input order across length-sorted chunks of 8", async () => {
    const reranker = new Reranker({ cacheDir: "/tmp/x" });
    // 20 passages, longest first; every passage has a unique length.
    const passages = Array.from({ length: 20 }, (_, i) =>
      Array(20 - i)
        .fill("w")
        .join(" "),
    );
    const scores = await reranker.rerank("w", passages);
    expect(scores).toHaveLength(20);
    scores.forEach((score, i) => {
      expect(score).toBeCloseTo(sigmoid(4 - 2 - (20 - i) / 100), 6);
    });
    expect(calls.tokenizer.map((c) => c.texts.length)).toEqual([8, 8, 4]);
    // Shortest passages go first (length-sorted chunks).
    const firstChunk = calls.tokenizer[0]?.options["text_pair"] as string[];
    expect(firstChunk[0]).toBe("w");
    expect(calls.tokenizer[0]?.texts.every((q) => q === "w")).toBe(true);
  });

  it("passes padding, truncation and max_length to the tokenizer", async () => {
    await new Reranker({ maxLength: 128 }).rerank("q", ["a"]);
    expect(calls.tokenizer[0]?.options).toMatchObject({
      padding: true,
      truncation: true,
      max_length: 128,
    });
    calls.tokenizer.length = 0;
    await new Reranker().rerank("q", ["a"]);
    expect(calls.tokenizer[0]?.options["max_length"]).toBe(1024);
  });

  it("raises the tokenizer limit for models whose profile allows it (jina), default 1024", async () => {
    const reranker = new Reranker({ cacheDir: "/tmp/x" });
    expect(reranker.getConfig().maxLength).toBe(1024);
    await reranker.init();
    expect(calls.tokenizers[0]?._tokenizerConfig.model_max_length).toBe(1024);

    const long = new Reranker({ maxLength: 4096, cacheDir: "/tmp/x" });
    await long.init();
    expect(long.getConfig().maxLength).toBe(4096);
    expect(calls.tokenizers[1]?._tokenizerConfig.model_max_length).toBe(4096);
    expect(new Reranker({ maxLength: 99999 }).getConfig().maxLength).toBe(8192);
  });

  it("keeps 512 for models without a profile (MiniLM): the limit is never raised", async () => {
    const reranker = new Reranker({
      model: RERANK_FALLBACK_MODEL,
      maxLength: 2048,
      cacheDir: "/tmp/x",
    });
    expect(
      new Reranker({ model: RERANK_FALLBACK_MODEL }).getConfig().maxLength,
    ).toBe(512);
    await reranker.init();
    // The tokenizer config is untouched, so Transformers.js still caps at 512.
    expect(calls.tokenizers[0]?._tokenizerConfig.model_max_length).toBe(512);
  });

  it("resolves offline from the option or NACHOS_EMBEDDINGS_OFFLINE", () => {
    expect(new Reranker().getConfig().offline).toBe(false);
    expect(new Reranker({ offline: true }).getConfig().offline).toBe(true);
    vi.stubEnv("NACHOS_EMBEDDINGS_OFFLINE", "1");
    expect(new Reranker().getConfig().offline).toBe(true);
    expect(new Reranker({ offline: false }).getConfig().offline).toBe(false);
    vi.unstubAllEnvs();
  });

  it("handles an empty list without loading the model", async () => {
    const reranker = new Reranker();
    expect(await reranker.rerank("q", [])).toEqual([]);
    expect(calls.loads).toEqual([]);
    expect(reranker.isInitialized()).toBe(false);
  });

  it("scores empty passages and an empty query", async () => {
    const reranker = new Reranker();
    const scores = await reranker.rerank("", ["", "text"]);
    expect(scores).toHaveLength(2);
    scores.forEach((s) => expect(Number.isFinite(s)).toBe(true));
  });

  it("rejects non-string input", async () => {
    const reranker = new Reranker();
    await expect(reranker.rerank(1 as never, [])).rejects.toThrow(TypeError);
    await expect(reranker.rerank("q", [1] as never)).rejects.toThrow(TypeError);
    await expect(reranker.rerank("q", "x" as never)).rejects.toThrow(TypeError);
  });

  it("loads the default model at q8 once, lazily, with its own cache dir", async () => {
    const reranker = new Reranker({ cacheDir: "/tmp/rr" });
    expect(calls.loads).toEqual([]);
    await Promise.all([
      reranker.rerank("q", ["a"]),
      reranker.rerank("q", ["b"]),
    ]);
    expect(calls.loads.map((l) => l.id)).toEqual([
      RERANK_DEFAULT_MODEL,
      RERANK_DEFAULT_MODEL,
    ]);
    expect(calls.loads[1]?.options).toEqual({
      cache_dir: "/tmp/rr",
      dtype: "q8",
    });
    expect(reranker.getModelId()).toBe(RERANK_DEFAULT_MODEL);
    expect(new Reranker({ dtype: "fp32" }).getModelId()).toBe(
      `${RERANK_DEFAULT_MODEL}:fp32`,
    );
  });

  it("validates options and disposes the model", async () => {
    expect(() => new Reranker({ dtype: "q9" as never })).toThrow(RangeError);
    expect(() => new Reranker({ maxLength: 0 })).toThrow(RangeError);
    const reranker = new Reranker();
    await reranker.init();
    await reranker.dispose();
    expect(calls.disposed).toBe(1);
    expect(reranker.isInitialized()).toBe(false);
  });
});

describe("rerank()", () => {
  it("caches one model per option set and reuses it", async () => {
    await rerank("q", ["a"], { model: RERANK_FALLBACK_MODEL });
    await rerank("q", ["b"], { model: RERANK_FALLBACK_MODEL });
    expect(
      calls.loads.filter((l) => l.id === RERANK_FALLBACK_MODEL),
    ).toHaveLength(2); // tokenizer + model, once
    await rerank("q", ["c"]);
    expect(calls.loads.map((l) => l.id)).toContain(RERANK_DEFAULT_MODEL);
    await disposeRerankers();
    expect(calls.disposed).toBe(2);
  });

  it("returns [] for an empty list", async () => {
    expect(await rerank("q", [])).toEqual([]);
  });
});
