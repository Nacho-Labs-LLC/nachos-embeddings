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
  TransformersProvider,
  resolveDefaultCacheDir,
  TRANSFORMERS_DEFAULT_MODEL,
  TRANSFORMERS_DEFAULT_DIMENSION,
} from "../../src/providers/transformers-provider.js";

function makeExtractor() {
  const extractor = vi.fn(async (text: string) => ({
    data: new Float32Array([text.length, 0.5, 0.25]),
  }));
  return Object.assign(extractor, { dispose: mockDispose });
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
