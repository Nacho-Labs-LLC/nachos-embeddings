import { describe, it, expect, vi, afterEach } from "vitest";
import { Embedder } from "../src/embedder.js";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("Embedder", () => {
  it("is not initialized before init()", () => {
    const embedder = new Embedder();
    expect(embedder.isInitialized()).toBe(false);
  });

  it("throws when embed() called before init()", async () => {
    const embedder = new Embedder();
    await expect(embedder.embed("test")).rejects.toThrow("not initialized");
  });

  it("throws when embedBatch() called before init()", async () => {
    const embedder = new Embedder();
    await expect(embedder.embedBatch(["test"])).rejects.toThrow(
      "not initialized",
    );
  });

  it("returns null dimension when not initialized", async () => {
    const embedder = new Embedder();
    expect(await embedder.getDimension()).toBeNull();
  });

  it("uses default config when none provided", () => {
    vi.stubEnv("NACHOS_EMBEDDINGS_CACHE", "/tmp/nachos-default-cache");
    const embedder = new Embedder();
    const config = embedder.getConfig();
    expect(config.model).toBe("Xenova/all-MiniLM-L6-v2");
    expect(config.cacheDir).toBe("/tmp/nachos-default-cache");
    expect(config.dtype).toBeUndefined();
    expect(config.progressLogging).toBe(false);
  });

  it("accepts custom config", () => {
    const embedder = new Embedder({
      model: "Xenova/all-mpnet-base-v2",
      cacheDir: "/tmp/models",
      progressLogging: true,
    });
    const config = embedder.getConfig();
    expect(config.model).toBe("Xenova/all-mpnet-base-v2");
    expect(config.cacheDir).toBe("/tmp/models");
    expect(config.progressLogging).toBe(true);
  });

  it("reports the model id with a non-default dtype", () => {
    expect(new Embedder().getModelId()).toBe("Xenova/all-MiniLM-L6-v2");
    expect(new Embedder({ dtype: "q8" }).getModelId()).toBe(
      "Xenova/all-MiniLM-L6-v2:q8",
    );
  });

  it("dispose is safe before init", async () => {
    await expect(new Embedder().dispose()).resolves.toBeUndefined();
  });
});
