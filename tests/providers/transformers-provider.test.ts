import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

const { mockPipeline, mockEnv, mockDispose } = vi.hoisted(() => ({
  mockPipeline: vi.fn(),
  mockEnv: { cacheDir: "untouched", allowLocalModels: true },
  mockDispose: vi.fn(),
}));

vi.mock("@huggingface/transformers", () => ({
  pipeline: mockPipeline,
  env: mockEnv,
}));

import {
  layerNormThenNormalize,
  TransformersProvider,
  resolveDefaultCacheDir,
  TRANSFORMERS_DEFAULT_MODEL,
  TRANSFORMERS_DEFAULT_DIMENSION,
} from "../../src/providers/transformers-provider.js";

type ExtractorCall = (
  input: string | string[],
  options?: unknown,
) => Promise<{ data: Float32Array; dims: number[] }>;
type TokenizerCall = (texts: unknown, options?: unknown) => object;

/** Fake feature-extraction pipeline: one row `[length, 0.5, 0.25]` per text. */
function makeExtractor() {
  const extractor = vi.fn<ExtractorCall>(async (input) => {
    const texts = Array.isArray(input) ? input : [input];
    return {
      data: new Float32Array(texts.flatMap((t) => [t.length, 0.5, 0.25])),
      dims: [texts.length, 3],
    };
  });
  return Object.assign(extractor, {
    dispose: mockDispose,
    tokenizer: vi.fn<TokenizerCall>(() => ({})),
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.stubEnv("NACHOS_EMBEDDINGS_CACHE", "");
  vi.stubEnv("XDG_CACHE_HOME", "");
  mockPipeline.mockImplementation(async () => makeExtractor());
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("resolveDefaultCacheDir", () => {
  const base = { homeDir: "/home/u", platform: "linux" };

  it("prefers NACHOS_EMBEDDINGS_CACHE", () => {
    expect(
      resolveDefaultCacheDir({
        ...base,
        env: { NACHOS_EMBEDDINGS_CACHE: "/n", XDG_CACHE_HOME: "/x" },
      }),
    ).toBe("/n");
  });

  it("then XDG_CACHE_HOME", () => {
    expect(
      resolveDefaultCacheDir({ ...base, env: { XDG_CACHE_HOME: "/x" } }),
    ).toBe("/x/nachos-embeddings");
  });

  it("then ~/.cache", () => {
    expect(resolveDefaultCacheDir({ ...base, env: {} })).toBe(
      "/home/u/.cache/nachos-embeddings",
    );
  });

  it("treats empty variables as unset", () => {
    expect(
      resolveDefaultCacheDir({
        ...base,
        env: { NACHOS_EMBEDDINGS_CACHE: "", XDG_CACHE_HOME: "" },
      }),
    ).toBe("/home/u/.cache/nachos-embeddings");
  });

  it("uses LOCALAPPDATA on Windows, ignoring XDG", () => {
    expect(
      resolveDefaultCacheDir({
        platform: "win32",
        homeDir: "C:\\Users\\u",
        env: { LOCALAPPDATA: "C:\\L", XDG_CACHE_HOME: "/x" },
      }),
    ).toBe("C:\\L\\nachos-embeddings");
  });

  it("falls back to AppData\\Local on Windows", () => {
    expect(
      resolveDefaultCacheDir({
        platform: "win32",
        homeDir: "C:\\Users\\u",
        env: {},
      }),
    ).toBe("C:\\Users\\u\\AppData\\Local\\nachos-embeddings");
  });

  it("is used by the provider unless cacheDir is explicit", () => {
    vi.stubEnv("NACHOS_EMBEDDINGS_CACHE", "/from-env");
    expect(new TransformersProvider().getConfig().cacheDir).toBe("/from-env");
    expect(
      new TransformersProvider({ cacheDir: "/explicit" }).getConfig().cacheDir,
    ).toBe("/explicit");
  });

  it("falls back to XDG_CACHE_HOME via the process environment", () => {
    vi.stubEnv("XDG_CACHE_HOME", "/xdg");
    if (process.platform !== "win32") {
      expect(new TransformersProvider().getConfig().cacheDir).toBe(
        "/xdg/nachos-embeddings",
      );
    }
  });
});

describe("TransformersProvider", () => {
  it("exports defaults", () => {
    expect(TRANSFORMERS_DEFAULT_MODEL).toBe("Xenova/all-MiniLM-L6-v2");
    expect(TRANSFORMERS_DEFAULT_DIMENSION).toBe(384);
    expect(new TransformersProvider().getConfig().model).toBe(
      TRANSFORMERS_DEFAULT_MODEL,
    );
  });

  it("does not mutate global env in the constructor", () => {
    new TransformersProvider({ cacheDir: "/a" });
    expect(mockEnv.cacheDir).toBe("untouched");
    expect(mockEnv.allowLocalModels).toBe(true);
  });

  it("passes cache_dir per pipeline call and restores allowLocalModels", async () => {
    let seenDuringLoad: boolean | undefined;
    mockPipeline.mockImplementationOnce(async () => {
      seenDuringLoad = mockEnv.allowLocalModels;
      return makeExtractor();
    });
    const provider = new TransformersProvider({ cacheDir: "/cache" });
    await provider.init();

    expect(mockPipeline).toHaveBeenCalledWith(
      "feature-extraction",
      TRANSFORMERS_DEFAULT_MODEL,
      { cache_dir: "/cache" },
    );
    expect(seenDuringLoad).toBe(false);
    expect(mockEnv.allowLocalModels).toBe(true);
    expect(mockEnv.cacheDir).toBe("untouched");
  });

  it("restores allowLocalModels when the load fails", async () => {
    mockPipeline.mockRejectedValueOnce(new Error("boom"));
    await expect(new TransformersProvider().init()).rejects.toThrow("boom");
    expect(mockEnv.allowLocalModels).toBe(true);
  });

  it("serializes loads across instances", async () => {
    const order: string[] = [];
    mockPipeline.mockImplementation(async (_t: string, model: string) => {
      order.push(`start:${model}`);
      await new Promise((r) => setTimeout(r, 5));
      order.push(`end:${model}`);
      return makeExtractor();
    });
    await Promise.all([
      new TransformersProvider({ model: "a" }).init(),
      new TransformersProvider({ model: "b" }).init(),
    ]);
    expect(order).toEqual(["start:a", "end:a", "start:b", "end:b"]);
  });

  it("passes dtype through and omits it when unset", async () => {
    await new TransformersProvider({ dtype: "q8" }).init();
    expect(mockPipeline.mock.calls[0]?.[2]).toMatchObject({ dtype: "q8" });

    mockPipeline.mockClear();
    await new TransformersProvider().init();
    expect(mockPipeline.mock.calls[0]?.[2]).not.toHaveProperty("dtype");
  });

  it("rejects an unknown dtype", () => {
    expect(() => new TransformersProvider({ dtype: "bogus" as never })).toThrow(
      RangeError,
    );
  });

  it("includes non-default dtype in the model id", () => {
    expect(new TransformersProvider().getModelId()).toBe(
      TRANSFORMERS_DEFAULT_MODEL,
    );
    expect(new TransformersProvider({ dtype: "fp32" }).getModelId()).toBe(
      TRANSFORMERS_DEFAULT_MODEL,
    );
    expect(new TransformersProvider({ dtype: "auto" }).getModelId()).toBe(
      TRANSFORMERS_DEFAULT_MODEL,
    );
    expect(
      new TransformersProvider({ model: "m", dtype: "q4" }).getModelId(),
    ).toBe("m:q4");
  });

  it("memoizes concurrent init() calls into one pipeline load", async () => {
    const provider = new TransformersProvider();
    await Promise.all([provider.init(), provider.init(), provider.init()]);
    expect(mockPipeline).toHaveBeenCalledTimes(1);
    expect(provider.isInitialized()).toBe(true);

    await provider.init();
    expect(mockPipeline).toHaveBeenCalledTimes(1);
  });

  it("allows retry after a failed init", async () => {
    mockPipeline.mockRejectedValueOnce(new Error("network down"));
    const provider = new TransformersProvider();

    await expect(provider.init()).rejects.toThrow("network down");
    expect(provider.isInitialized()).toBe(false);

    await provider.init();
    expect(provider.isInitialized()).toBe(true);
    expect(mockPipeline).toHaveBeenCalledTimes(2);
  });

  it("shares a failure between concurrent callers", async () => {
    mockPipeline.mockRejectedValueOnce(new Error("nope"));
    const provider = new TransformersProvider();
    const results = await Promise.allSettled([
      provider.init(),
      provider.init(),
    ]);
    expect(results.map((r) => r.status)).toEqual(["rejected", "rejected"]);
    expect(mockPipeline).toHaveBeenCalledTimes(1);
  });

  it("embeds single texts and batches, preserving order", async () => {
    const provider = new TransformersProvider();
    await provider.init();

    expect(await provider.embed("abc")).toEqual([3, 0.5, 0.25]);

    const texts = Array.from({ length: 70 }, (_, i) => "x".repeat(i + 1));
    const vectors = await provider.embedBatch(texts);
    expect(vectors).toHaveLength(70);
    expect(vectors.map((v) => v[0])).toEqual(texts.map((t) => t.length));
    expect(await provider.getDimension()).toBe(3);
  });

  it("throws before init", async () => {
    const provider = new TransformersProvider();
    await expect(provider.embed("a")).rejects.toThrow("not initialized");
    await expect(provider.embedBatch(["a"])).rejects.toThrow("not initialized");
    expect(await provider.getDimension()).toBeNull();
  });

  it("logs progress when enabled", async () => {
    const log = vi.spyOn(console, "log").mockImplementation(() => undefined);
    await new TransformersProvider({ progressLogging: true }).init();
    expect(log).toHaveBeenCalledWith(
      expect.stringContaining("Model loaded successfully"),
    );
    log.mockRestore();
  });

  it("dispose releases the pipeline and allows re-init", async () => {
    const provider = new TransformersProvider();
    await provider.init();
    await provider.dispose();

    expect(mockDispose).toHaveBeenCalledTimes(1);
    expect(provider.isInitialized()).toBe(false);
    await expect(provider.embed("a")).rejects.toThrow("not initialized");

    await provider.init();
    expect(mockPipeline).toHaveBeenCalledTimes(2);
  });

  it("dispose before init is a no-op", async () => {
    await expect(new TransformersProvider().dispose()).resolves.toBeUndefined();
    expect(mockDispose).not.toHaveBeenCalled();
  });

  it("dispose waits for an in-flight init", async () => {
    const provider = new TransformersProvider();
    const init = provider.init();
    await provider.dispose();
    await init;
    expect(mockDispose).toHaveBeenCalledTimes(1);
    expect(provider.isInitialized()).toBe(false);
  });

  it("dispose after a failed in-flight init does not throw", async () => {
    mockPipeline.mockRejectedValueOnce(new Error("x"));
    const provider = new TransformersProvider();
    const init = provider.init().catch(() => undefined);
    await expect(provider.dispose()).resolves.toBeUndefined();
    await init;
  });
});

describe("TransformersProvider prefixes, pooling and batching", () => {
  async function load(
    config: ConstructorParameters<typeof TransformersProvider>[0],
  ) {
    const provider = new TransformersProvider(config);
    await provider.init();
    const extractor = (await mockPipeline.mock.results.at(-1)!
      .value) as ReturnType<typeof makeExtractor>;
    return { provider, extractor };
  }

  it("defaults to document type, mean pooling and no prefix for MiniLM", async () => {
    const { provider, extractor } = await load({});
    expect(provider.getConfig()).toMatchObject({
      pooling: "mean",
      queryPrefix: "",
      documentPrefix: "",
    });
    await provider.embed("abc");
    expect(extractor).toHaveBeenCalledWith("abc", {
      pooling: "mean",
      normalize: true,
    });
  });

  it("applies the built-in e5 prefixes by input type", async () => {
    const { provider, extractor } = await load({
      model: "intfloat/e5-small-v2",
    });
    await provider.embed("abc");
    await provider.embed("abc", { type: "query" });
    await provider.embedBatch(["a"], { type: "query" });
    expect(extractor.mock.calls.map((c) => c[0])).toEqual([
      "passage: abc",
      "query: abc",
      ["query: a"],
    ]);
  });

  it("uses cls pooling and the instruction prefix for bge, none for documents", async () => {
    const { provider, extractor } = await load({
      model: "Xenova/bge-small-en-v1.5",
    });
    expect(provider.getConfig().pooling).toBe("cls");
    await provider.embed("abc", { type: "query" });
    await provider.embed("abc");
    expect(extractor.mock.calls[0]).toEqual([
      "Represent this sentence for searching relevant passages: abc",
      { pooling: "cls", normalize: true },
    ]);
    expect(extractor.mock.calls[1]?.[0]).toBe("abc");
  });

  it("applies the nomic prefixes", async () => {
    const { provider, extractor } = await load({
      model: "nomic-ai/nomic-embed-text-v1.5",
    });
    await provider.embed("x", { type: "query" });
    await provider.embed("x", { type: "document" });
    expect(extractor.mock.calls.map((c) => c[0])).toEqual([
      "search_query: x",
      "search_document: x",
    ]);
  });

  it("lets config override prefixes (including empty) and pooling", async () => {
    const { provider, extractor } = await load({
      model: "intfloat/e5-small-v2",
      queryPrefix: "",
      documentPrefix: "doc: ",
      pooling: "cls",
    });
    await provider.embed("a", { type: "query" });
    await provider.embed("a");
    expect(extractor.mock.calls[0]).toEqual([
      "a",
      { pooling: "cls", normalize: true },
    ]);
    expect(extractor.mock.calls[1]?.[0]).toBe("doc: a");
  });

  it("reflects vector-changing overrides in the model id", () => {
    expect(new TransformersProvider().getModelId()).toBe(
      TRANSFORMERS_DEFAULT_MODEL,
    );
    expect(
      new TransformersProvider({
        model: "BAAI/bge-small-en-v1.5",
      }).getModelId(),
    ).toBe("BAAI/bge-small-en-v1.5");
    expect(new TransformersProvider({ pooling: "cls" }).getModelId()).toBe(
      `${TRANSFORMERS_DEFAULT_MODEL}:pooling=cls`,
    );
    expect(
      new TransformersProvider({ documentPrefix: "d: " }).getModelId(),
    ).toBe(`${TRANSFORMERS_DEFAULT_MODEL}:docprefix="d: "`);
    // A query prefix does not affect stored vectors.
    expect(new TransformersProvider({ queryPrefix: "q: " }).getModelId()).toBe(
      TRANSFORMERS_DEFAULT_MODEL,
    );
  });

  it("rejects invalid type, pooling and maxLength", async () => {
    expect(() => new TransformersProvider({ pooling: "max" as never })).toThrow(
      RangeError,
    );
    expect(() => new TransformersProvider({ maxLength: 0 })).toThrow(
      RangeError,
    );
    expect(() => new TransformersProvider({ maxLength: 1.5 })).toThrow(
      RangeError,
    );
    const { provider } = await load({});
    await expect(
      provider.embed("a", { type: "other" as never }),
    ).rejects.toThrow("Invalid input type");
  });

  it("makes one padded pipeline call per chunk of 8, in input order", async () => {
    const { provider, extractor } = await load({});
    const texts = Array.from({ length: 20 }, (_, i) => "x".repeat(20 - i));
    const vectors = await provider.embedBatch(texts);
    expect(extractor).toHaveBeenCalledTimes(3);
    expect(extractor.mock.calls.every((c) => Array.isArray(c[0]))).toBe(true);
    // Chunks are length-sorted, but results are restored to input order.
    expect(vectors.map((v) => v[0])).toEqual(texts.map((t) => t.length));
    expect(await provider.embedBatch([])).toEqual([]);
  });

  it("passes max_length to the tokenizer only when configured", async () => {
    const tokenizers: Array<ReturnType<typeof vi.fn>> = [];
    mockPipeline.mockImplementation(async () => {
      const extractor = makeExtractor();
      tokenizers.push(extractor.tokenizer);
      return extractor;
    });

    const plain = await load({});
    plain.extractor.tokenizer("t", { padding: true, truncation: true });
    expect(tokenizers[0]).toHaveBeenCalledWith("t", {
      padding: true,
      truncation: true,
    });
    expect(plain.provider.getConfig().maxLength).toBeUndefined();

    const limited = await load({ maxLength: 64 });
    limited.extractor.tokenizer("t", { padding: true, truncation: true });
    expect(tokenizers[1]).toHaveBeenCalledWith("t", {
      padding: true,
      truncation: true,
      max_length: 64,
    });
    expect(limited.provider.getConfig().maxLength).toBe(64);
  });

  it("layer-norms then normalizes nomic vectors, and leaves others to the pipeline", async () => {
    const nomic = await load({ model: "nomic-ai/nomic-embed-text-v1.5" });
    expect(nomic.extractor.mock.calls.length).toBe(0);
    const vec = await nomic.provider.embed("abcd", { type: "query" });
    expect(nomic.extractor).toHaveBeenCalledWith("search_query: abcd", {
      pooling: "mean",
      normalize: false,
    });
    expect(Math.hypot(...vec)).toBeCloseTo(1, 6);
    expect(vec).toEqual(layerNormThenNormalize([18, 0.5, 0.25]));
    const batch = await nomic.provider.embedBatch(["ab", "abcd"]);
    expect(Math.hypot(...(batch[1] as number[]))).toBeCloseTo(1, 6);

    const bge = await load({ model: "BAAI/bge-small-en-v1.5" });
    await bge.provider.embed("x");
    expect(bge.extractor.mock.calls[0]?.[1]).toEqual({
      pooling: "cls",
      normalize: true,
    });
  });
});

describe("layerNormThenNormalize", () => {
  it("matches a hand-computed layer norm followed by L2 normalization", () => {
    // [1,3]: mean 2, var 1 -> [-1,1]/sqrt(1+1e-5) -> L2 normalized [-s, s]
    const out = layerNormThenNormalize([1, 3]);
    expect(out[0]).toBeCloseTo(-Math.SQRT1_2, 6);
    expect(out[1]).toBeCloseTo(Math.SQRT1_2, 6);
  });

  it("returns zeros for a constant row instead of NaN", () => {
    expect(layerNormThenNormalize([2, 2, 2])).toEqual([0, 0, 0]);
  });
});
