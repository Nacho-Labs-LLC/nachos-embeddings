import { describe, it, expect, afterEach, beforeEach } from "vitest";
import { mkdtemp, rm, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  EnhancedSemanticSearch,
  type EnhancedSemanticSearchConfig,
} from "../src/semantic-search-enhanced.js";
import { FakeProvider } from "./fake-provider.js";

let dir: string;
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "enh-"));
});
afterEach(async () => {
  await rm(dir, { recursive: true, force: true });
});

describe("EnhancedSemanticSearch (deprecated alias)", () => {
  it("maps the old config: normalized exact dedup by default", async () => {
    const s = new EnhancedSemanticSearch({ provider: new FakeProvider() });
    await s.addDocument({ id: "a", text: "Hello   World" });
    await s.addDocument({ id: "b", text: "hello world" });
    expect(s.size()).toBe(1);
    expect(s.getConfig().deduplicate).toBe("normalized");
  });

  it("maps deduplicateExact:false to exact and deduplicateSimilarity to similarity", () => {
    const p = new FakeProvider();
    expect(
      new EnhancedSemanticSearch({
        provider: p,
        deduplicateExact: false,
      }).getConfig().deduplicate,
    ).toBe("exact");
    expect(
      new EnhancedSemanticSearch({
        provider: p,
        deduplicateSimilarity: 0.9,
      }).getConfig().deduplicate,
    ).toEqual({ similarity: 0.9 });
  });

  it("accepts the full old mcp-semantic-search config and round-trips via a store", async () => {
    const storePath = join(dir, "sub", "store.json");
    const cfg: EnhancedSemanticSearchConfig = {
      provider: new FakeProvider(),
      minSimilarity: 0.1,
      autoSave: true,
      storePath,
      autoChunk: true,
      deduplicateExact: true,
      deduplicateSimilarity: 0,
      temporalBoost: true,
      verbose: false,
    };
    const s = new EnhancedSemanticSearch<{ kind?: string }>(cfg);
    await s.init();
    await s.addDocuments([
      { id: "a", text: "tacos for breakfast", metadata: { kind: "pref" } },
      { id: "b", text: "pizza for dinner" },
    ]);
    const s2 = new EnhancedSemanticSearch({
      ...cfg,
      provider: new FakeProvider(),
    });
    await s2.init();
    expect(s2.size()).toBe(2);
    const res = await s2.search("breakfast tacos", { limit: 1 });
    expect(res[0]?.id).toBe("a");
    expect(s2.export()).toHaveLength(2);
  });

  it("uses the legacy default store path when autoSave has no path", () => {
    const s = new EnhancedSemanticSearch({
      provider: new FakeProvider(),
      autoSave: true,
    });
    expect(s.getConfig().storePath).toBe(".semantic-store.json");
  });

  it("forceSave writes the store, and is a no-op without storePath", async () => {
    const none = new EnhancedSemanticSearch({ provider: new FakeProvider() });
    await expect(none.forceSave()).resolves.toBeUndefined();

    const storePath = join(dir, "s.json");
    const s = new EnhancedSemanticSearch({
      provider: new FakeProvider(),
      storePath,
    });
    await s.addDocument({ id: "a", text: "x y z" });
    await s.forceSave();
    expect(JSON.parse(await readFile(storePath, "utf-8"))).toHaveLength(1);
  });

  it("reads the old on-disk array format", async () => {
    const storePath = join(dir, "old.json");
    await writeFile(
      storePath,
      JSON.stringify([
        { id: "a", text: "alpha", vector: [1, 0], metadata: { k: 1 } },
      ]),
    );
    const s = new EnhancedSemanticSearch({
      provider: new FakeProvider(),
      storePath,
    });
    await s.init();
    expect(s.getDocument("a")).toEqual({
      id: "a",
      text: "alpha",
      metadata: { k: 1 },
    });
  });
});
