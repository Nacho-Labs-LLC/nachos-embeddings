import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";
import {
  mkdtemp,
  rm,
  readFile,
  writeFile,
  readdir,
  mkdir,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { SemanticSearch, StoreCorruptError } from "../src/semantic-search.js";
import { FakeProvider } from "./fake-provider.js";

type Meta = {
  timestamp?: number;
  kind?: string;
  chunkIndex?: number;
  parentId?: string;
};

const hooks = vi.hoisted(() => ({
  renameFailures: [] as unknown[],
  alwaysFail: undefined as unknown,
}));

vi.mock("node:fs/promises", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs/promises")>();
  return {
    ...actual,
    rename: async (from: string, to: string) => {
      if (hooks.alwaysFail !== undefined) throw hooks.alwaysFail;
      if (hooks.renameFailures.length > 0) throw hooks.renameFailures.shift();
      return actual.rename(from, to);
    },
  };
});

let dir: string;
let provider: FakeProvider;
beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), "ss-"));
  provider = new FakeProvider();
});
afterEach(async () => {
  vi.restoreAllMocks();
  hooks.renameFailures = [];
  hooks.alwaysFail = undefined;
  await rm(dir, { recursive: true, force: true });
});

const make = (o: ConstructorParameters<typeof SemanticSearch<Meta>>[0] = {}) =>
  new SemanticSearch<Meta>({ provider, minSimilarity: 0, ...o });

describe("deduplicate modes", () => {
  it("exact (default) only skips byte-identical text", async () => {
    const s = make();
    await s.addDocument({ id: "a", text: "Hello world" });
    await s.addDocument({ id: "b", text: "Hello world" });
    await s.addDocument({ id: "c", text: "hello world" });
    expect(s.listDocuments().map((d) => d.id)).toEqual(["a", "c"]);
  });

  it("off keeps everything", async () => {
    const s = make({ deduplicate: "off" });
    await s.addDocuments([
      { id: "a", text: "same" },
      { id: "b", text: "same" },
    ]);
    expect(s.size()).toBe(2);
  });

  it("legacy deduplication:false maps to off, deduplicate wins over legacy", () => {
    expect(make({ deduplication: false }).getConfig().deduplicate).toBe("off");
    expect(make({ deduplication: true }).getConfig().deduplicate).toBe("exact");
    expect(
      make({ deduplicate: "normalized", deduplication: false }).getConfig()
        .deduplicate,
    ).toBe("normalized");
  });

  it("normalized skips case/whitespace variants, within a batch too", async () => {
    const s = make({ deduplicate: "normalized" });
    await s.addDocuments([
      { id: "a", text: "Hello  World" },
      { id: "b", text: " hello world " },
      { id: "c", text: "other" },
    ]);
    expect(s.listDocuments().map((d) => d.id)).toEqual(["a", "c"]);
    await s.addDocument({ id: "d", text: "HELLO WORLD" });
    expect(s.size()).toBe(2);
  });

  it("similarity skips near duplicates but not different docs", async () => {
    const s = make({ deduplicate: { similarity: 0.8 } });
    await s.addDocument({
      id: "a",
      text: "the quick brown fox jumps over the lazy dog",
    });
    await s.addDocument({
      id: "b",
      text: "the quick brown fox jumps over the lazy dog again",
    });
    await s.addDocument({
      id: "c",
      text: "completely unrelated sentence about tacos",
    });
    expect(s.listDocuments().map((d) => d.id)).toEqual(["a", "c"]);
    // exact (normalized) match is also caught
    await s.addDocument({
      id: "d",
      text: "The Quick brown fox jumps over the lazy dog",
    });
    expect(s.size()).toBe(2);
  });

  it("legacy deduplicateSimilarity maps to similarity; 0 is ignored", () => {
    expect(
      make({ deduplicateSimilarity: 0.7 }).getConfig().deduplicate,
    ).toEqual({
      similarity: 0.7,
    });
    expect(make({ deduplicateSimilarity: 0 }).getConfig().deduplicate).toBe(
      "exact",
    );
  });

  it("re-adding the same id with new text replaces and frees the old text", async () => {
    const s = make();
    await s.addDocument({ id: "a", text: "first" });
    await s.addDocument({ id: "a", text: "second" });
    expect(s.getDocument("a")?.text).toBe("second");
    await s.addDocument({ id: "b", text: "first" });
    expect(s.size()).toBe(2);
  });

  it("same id with same text is a no-op", async () => {
    const s = make();
    await s.addDocument({ id: "a", text: "first" });
    await s.addDocument({ id: "a", text: "first" });
    expect(s.size()).toBe(1);
  });
});

describe("chunking", () => {
  const long = Array.from(
    { length: 40 },
    (_, i) => `Sentence number ${i} is here.`,
  ).join(" ");

  it("splits long documents with chunk ids and metadata", async () => {
    const s = make({ autoChunk: true, maxChunkTokens: 50, chunkOverlap: 10 });
    await s.addDocument({ id: "doc", text: long, metadata: { kind: "k" } });
    const ids = s.listDocuments().map((d) => d.id);
    expect(ids.length).toBeGreaterThan(1);
    expect(ids[0]).toBe("doc#chunk0");
    const meta = s.getDocument("doc#chunk1")?.metadata;
    expect(meta).toMatchObject({ kind: "k", chunkIndex: 1, parentId: "doc" });
  });

  it("does not chunk short docs or when disabled", async () => {
    const s = make({ autoChunk: true, maxChunkTokens: 500 });
    await s.addDocument({ id: "short", text: "tiny" });
    expect(s.listDocuments().map((d) => d.id)).toEqual(["short"]);
    const off = make({ maxChunkTokens: 5 });
    await off.addDocument({ id: "x", text: long });
    expect(off.listDocuments().map((d) => d.id)).toEqual(["x"]);
  });

  it("single-chunk results keep the original id", async () => {
    const s = make({ autoChunk: true, maxChunkTokens: 5 });
    await s.addDocument({ id: "one", text: "abcdefghijklmnopqrstuvwxyz" });
    expect(s.listDocuments().map((d) => d.id)).toEqual(["one"]);
    expect(s.getDocument("one")?.metadata).toMatchObject({
      chunkIndex: 0,
      parentId: "one",
    });
  });

  it("dedupes a chunked document by its whole text, and remove(parent) drops chunks", async () => {
    const s = make({ autoChunk: true, maxChunkTokens: 50, chunkOverlap: 10 });
    await s.addDocument({ id: "doc", text: long });
    const n = s.size();
    await s.addDocument({ id: "dup", text: long });
    expect(s.size()).toBe(n);
    expect(await s.remove("doc")).toBe(true);
    expect(s.size()).toBe(0);
    await s.addDocument({ id: "again", text: long });
    expect(s.size()).toBe(n);
  });

  it("removing a single chunk keeps its siblings; replacing a parent drops old chunks", async () => {
    const s = make({ autoChunk: true, maxChunkTokens: 50, chunkOverlap: 10 });
    await s.addDocument({ id: "doc", text: long });
    const n = s.size();
    expect(await s.remove("doc#chunk0")).toBe(true);
    expect(s.size()).toBe(n - 1);
    await s.addDocument({ id: "doc", text: `${long} extra tail sentence.` });
    expect(s.getDocument("doc#chunk0")).toBeDefined();
    expect(s.listDocuments().every((d) => d.id.startsWith("doc#chunk"))).toBe(
      true,
    );
  });

  it("chunk links survive export/import", async () => {
    const s = make({ autoChunk: true, maxChunkTokens: 50, chunkOverlap: 10 });
    await s.addDocument({ id: "doc", text: long });
    const copy = make();
    copy.import(s.export());
    expect(await copy.remove("doc")).toBe(true);
    expect(copy.size()).toBe(0);
  });
});

describe("temporal boost", () => {
  it("stamps timestamps only when enabled and keeps user timestamps", async () => {
    const on = make({ temporalBoost: true, deduplicate: "off" });
    await on.addDocument({ id: "a", text: "x" });
    await on.addDocument({ id: "b", text: "y", metadata: { timestamp: 5 } });
    expect(typeof on.getDocument("a")?.metadata?.timestamp).toBe("number");
    expect(on.getDocument("b")?.metadata?.timestamp).toBe(5);
    const off = make();
    await off.addDocument({ id: "a", text: "x" });
    expect(off.getDocument("a")?.metadata).toBeUndefined();
  });

  it("re-ranks recent documents above older equally-similar ones, and respects limit afterwards", async () => {
    const s = make({ deduplicate: "off" });
    const day = 86_400_000;
    await s.addDocuments([
      {
        id: "old",
        text: "tacos",
        metadata: { timestamp: Date.now() - 400 * day },
      },
      { id: "new", text: "tacos", metadata: { timestamp: Date.now() } },
      {
        id: "mid",
        text: "tacos",
        metadata: { timestamp: Date.now() - 30 * day },
      },
      { id: "none", text: "tacos" },
    ]);
    const plain = await s.search("tacos", { limit: 4 });
    expect(
      plain.every((r) => Math.abs(r.similarity - plain[0]!.similarity) < 1e-9),
    ).toBe(true);
    const boosted = await s.search("tacos", { temporalBoost: true, limit: 3 });
    expect(boosted).toHaveLength(3);
    expect(boosted.map((r) => r.id)).not.toContain("old");
    const all = await s.search("tacos", { temporalBoost: true });
    expect(all.map((r) => r.id).slice(2)).toEqual(["mid", "old"]);
    expect(
      all
        .slice(0, 2)
        .map((r) => r.id)
        .sort(),
    ).toEqual(["new", "none"]);
  });

  it("instance flag enables it and per-search false disables it", async () => {
    const s = make({ deduplicate: "off", temporalBoost: true });
    await s.addDocuments([
      { id: "old", text: "tacos", metadata: { timestamp: 1 } },
      { id: "new", text: "tacos", metadata: { timestamp: Date.now() } },
    ]);
    expect((await s.search("tacos"))[0]?.id).toBe("new");
    const raw = await s.search("tacos", { temporalBoost: false });
    expect(raw[0]?.similarity).toBeCloseTo(raw[1]!.similarity);
  });
});

describe("search", () => {
  it("applies filter, minSimilarity and limit", async () => {
    const s = make({ minSimilarity: undefined });
    await s.addDocuments([
      { id: "a", text: "apple pie", metadata: { kind: "x" } },
      { id: "b", text: "apple tart", metadata: { kind: "y" } },
      { id: "c", text: "zzz qqq" },
    ]);
    const r = await s.search("apple pie", {
      filter: (m) => m?.kind === "y",
      minSimilarity: 0.1,
    });
    expect(r.map((x) => x.id)).toEqual(["b"]);
    expect(r[0]?.text).toBe("apple tart");
    expect(await s.search("apple pie", { limit: 1 })).toHaveLength(1);
    expect(await s.search("apple pie", { minSimilarity: 0.99 })).toHaveLength(
      1,
    );
  });

  it("init() initializes the provider", async () => {
    const s = make();
    expect(s.isInitialized()).toBe(false);
    await s.init();
    expect(s.isInitialized()).toBe(true);
  });
});

describe("concurrency", () => {
  it("concurrent adds of the same text dedupe correctly", async () => {
    provider.delayMs = 5;
    const s = make({ deduplicate: "normalized" });
    await Promise.all(
      Array.from({ length: 10 }, (_, i) =>
        s.addDocument({ id: `d${i}`, text: "Same Text" }),
      ),
    );
    expect(s.size()).toBe(1);
    expect(s.listDocuments()[0]?.id).toBe("d0");
  });

  it("concurrent add/remove/clear stay consistent", async () => {
    provider.delayMs = 2;
    const s = make();
    const p = [
      s.addDocument({ id: "a", text: "alpha" }),
      s.remove("a"),
      s.addDocument({ id: "b", text: "beta" }),
      s.clear(),
      s.addDocument({ id: "c", text: "gamma" }),
    ];
    const res = await Promise.all(p);
    expect(res[1]).toBe(true);
    expect(s.listDocuments().map((d) => d.id)).toEqual(["c"]);
  });

  it("a failed embed does not wedge later mutations", async () => {
    const s = make();
    vi.spyOn(provider, "embed").mockRejectedValueOnce(new Error("boom"));
    await expect(s.addDocument({ id: "a", text: "x" })).rejects.toThrow("boom");
    await s.addDocument({ id: "a", text: "x" });
    expect(s.size()).toBe(1);
  });
});

describe("remove / import / export", () => {
  it("remove returns false for unknown ids and frees dedup text", async () => {
    const s = make();
    await s.addDocument({ id: "a", text: "hello" });
    expect(await s.remove("nope")).toBe(false);
    expect(await s.remove("a")).toBe(true);
    await s.addDocument({ id: "b", text: "hello" });
    expect(s.getDocument("b")).toBeDefined();
  });

  it("export/import round-trips documents, vectors and metadata", async () => {
    const s = make();
    await s.addDocuments([
      { id: "a", text: "alpha beta", metadata: { kind: "k" } },
      { id: "b", text: "gamma" },
    ]);
    const data = s.export();
    const copy = make();
    copy.import(JSON.parse(JSON.stringify(data)));
    expect(copy.export()).toEqual(JSON.parse(JSON.stringify(data)));
    expect((await copy.search("alpha beta"))[0]?.id).toBe("a");
  });
});

describe("logging", () => {
  it("is silent by default", async () => {
    const spies = (["log", "warn", "error"] as const).map((k) =>
      vi.spyOn(console, k).mockImplementation(() => {}),
    );
    const s = make();
    await s.addDocuments([
      { id: "a", text: "same" },
      { id: "b", text: "same" },
    ]);
    spies.forEach((sp) => expect(sp).not.toHaveBeenCalled());
  });

  it("verbose maps to console; a custom logger takes precedence", async () => {
    const log = vi.spyOn(console, "log").mockImplementation(() => {});
    const v = make({ verbose: true });
    await v.addDocuments([
      { id: "a", text: "same" },
      { id: "b", text: "same" },
    ]);
    expect(log).toHaveBeenCalled();
    log.mockClear();
    const debug = vi.fn();
    const c = make({ verbose: true, logger: { debug } });
    await c.addDocuments([
      { id: "a", text: "same" },
      { id: "b", text: "same" },
    ]);
    expect(debug).toHaveBeenCalled();
    expect(log).not.toHaveBeenCalled();
    // partial logger: missing methods are no-ops
    await expect(
      make({ logger: {}, deduplicate: "off" }).addDocument({
        id: "z",
        text: "z",
      }),
    ).resolves.toBeUndefined();
  });

  it("verbose routes warn/error to console", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    const path = join(dir, "bad.json");
    await writeFile(path, "{nope");
    const q = make({
      storePath: path,
      verbose: true,
      onCorruptStore: "quarantine",
    });
    await q.init();
    expect(warn).toHaveBeenCalled();
    const bad = make({
      storePath: join(dir, "x.json"),
      autoSave: true,
      verbose: true,
    });
    await mkdir(join(dir, "x.json"));
    await bad.addDocument({ id: "a", text: "a" });
    expect(error).toHaveBeenCalled();
  });
});

describe("persistence", () => {
  it("requires storePath for autoSave and persist", async () => {
    expect(() => new SemanticSearch({ provider, autoSave: true })).toThrow(
      /storePath/,
    );
    await expect(make().persist()).rejects.toThrow(/storePath/);
  });

  it("concurrent autoSave adds produce one consistent file and no temp leftovers", async () => {
    const storePath = join(dir, "nested", "store.json");
    const s = make({ autoSave: true, storePath });
    await s.init();
    await Promise.all(
      Array.from({ length: 20 }, (_, i) =>
        s.addDocument({ id: `d${i}`, text: `doc number ${i} text` }),
      ),
    );
    const onDisk = JSON.parse(await readFile(storePath, "utf-8")) as Array<{
      id: string;
    }>;
    expect(onDisk).toHaveLength(20);
    expect(onDisk.map((d) => d.id).sort()).toEqual(
      Array.from({ length: 20 }, (_, i) => `d${i}`).sort(),
    );
    expect(await readdir(join(dir, "nested"))).toEqual(["store.json"]);
  });

  it("autoSave covers addDocuments, remove and clear, and a new instance reloads", async () => {
    const storePath = join(dir, "s.json");
    const s = make({ autoSave: true, storePath });
    await s.init();
    await s.addDocuments([
      { id: "a", text: "alpha" },
      { id: "b", text: "beta" },
    ]);
    await s.remove("a");
    expect(
      JSON.parse(await readFile(storePath, "utf-8")).map(
        (d: { id: string }) => d.id,
      ),
    ).toEqual(["b"]);
    const s2 = make({ autoSave: true, storePath });
    await s2.init();
    expect(s2.size()).toBe(1);
    await s2.addDocument({ id: "dup", text: "beta" }); // dedup index rebuilt on load
    expect(s2.size()).toBe(1);
    await s2.clear();
    expect(JSON.parse(await readFile(storePath, "utf-8"))).toEqual([]);
  });

  it("a missing file is an empty store; loadOnInit:false skips loading", async () => {
    const storePath = join(dir, "missing.json");
    const s = make({ storePath });
    await s.init();
    expect(s.size()).toBe(0);
    await writeFile(
      storePath,
      JSON.stringify([{ id: "a", text: "t", vector: [1] }]),
    );
    const skip = make({ storePath, loadOnInit: false });
    await skip.init();
    expect(skip.size()).toBe(0);
    await s.load();
    expect(s.size()).toBe(1);
  });

  it("load keeps in-memory documents over file entries; load() without storePath is a no-op", async () => {
    const storePath = join(dir, "s.json");
    await writeFile(
      storePath,
      JSON.stringify([{ id: "a", text: "old", vector: [1] }]),
    );
    const s = make({ storePath, loadOnInit: false });
    s.import([{ id: "a", text: "new", vector: [1] }]);
    await s.load();
    expect(s.getDocument("a")?.text).toBe("new");
    await expect(make().load()).resolves.toBeUndefined();
  });

  it("a corrupt file throws StoreCorruptError and is never overwritten", async () => {
    const storePath = join(dir, "s.json");
    await writeFile(storePath, '[{"id":"a","te');
    const s = make({ storePath, autoSave: true });
    await expect(s.init()).rejects.toBeInstanceOf(StoreCorruptError);
    await expect(s.init()).rejects.toThrow(/corrupt/);

    // even without init, a write must not clobber the unreadable file
    const errors: Error[] = [];
    const s2 = make({
      storePath,
      autoSave: true,
      onError: (e) => errors.push(e),
    });
    await s2.addDocument({ id: "x", text: "new data" });
    expect(await readFile(storePath, "utf-8")).toBe('[{"id":"a","te');
    expect(errors[0]).toBeInstanceOf(StoreCorruptError);
    await expect(s2.persist()).rejects.toBeInstanceOf(StoreCorruptError);
  });

  it.each([
    ["non-array JSON", '{"a":1}'],
    ["malformed entry", '[{"id":"a","text":"t"}]'],
    ["non-object entry", "[1]"],
  ])("rejects %s", async (_n, content) => {
    const storePath = join(dir, "s.json");
    await writeFile(storePath, content);
    await expect(make({ storePath }).init()).rejects.toBeInstanceOf(
      StoreCorruptError,
    );
    expect(await readFile(storePath, "utf-8")).toBe(content);
  });

  it("onCorruptStore: quarantine moves the file aside and starts empty", async () => {
    const storePath = join(dir, "s.json");
    await writeFile(storePath, "garbage");
    const warn = vi.fn();
    const s = make({
      storePath,
      autoSave: true,
      onCorruptStore: "quarantine",
      logger: { warn },
    });
    await s.init();
    expect(s.size()).toBe(0);
    expect(warn).toHaveBeenCalled();
    const files = await readdir(dir);
    const aside = files.find((f) => f.startsWith("s.json.corrupt-"));
    expect(aside).toBeDefined();
    expect(await readFile(join(dir, aside!), "utf-8")).toBe("garbage");
    await s.addDocument({ id: "a", text: "fresh" });
    expect(JSON.parse(await readFile(storePath, "utf-8"))).toHaveLength(1);
  });

  it("non-corruption read errors propagate (EISDIR)", async () => {
    const storePath = join(dir, "isdir");
    await mkdir(storePath);
    await expect(make({ storePath }).init()).rejects.toThrow();
  });

  it("atomic write: a failed rename leaves the previous file intact and no temp files", async () => {
    const storePath = join(dir, "s.json");
    const s = make({ storePath });
    await s.addDocument({ id: "a", text: "alpha" });
    await s.persist();
    const before = await readFile(storePath, "utf-8");

    await s.addDocument({ id: "b", text: "beta" });
    hooks.renameFailures.push(new Error("disk gone"));
    await expect(s.persist()).rejects.toThrow("disk gone");

    expect(await readFile(storePath, "utf-8")).toBe(before);
    expect(JSON.parse(before)).toHaveLength(1);
    expect(await readdir(dir)).toEqual(["s.json"]);
    await s.persist(); // recovers
    expect(JSON.parse(await readFile(storePath, "utf-8"))).toHaveLength(2);
  });

  it("autoSave failures go to onError/logger, add still resolves, next persist rethrows once", async () => {
    const storePath = join(dir, "s.json");
    const onError = vi.fn();
    const error = vi.fn();
    const s = make({ storePath, autoSave: true, onError, logger: { error } });
    await s.init();
    hooks.renameFailures.push(new Error("no space"));
    await s.addDocument({ id: "a", text: "alpha" });
    expect(s.size()).toBe(1);
    expect(onError).toHaveBeenCalledTimes(1);
    expect(error).toHaveBeenCalled();

    await expect(s.persist()).rejects.toThrow("no space");
    // the write itself succeeded, and the error is reported only once
    expect(JSON.parse(await readFile(storePath, "utf-8"))).toHaveLength(1);
    await expect(s.persist()).resolves.toBeUndefined();
  });

  it("wraps non-Error rejections", async () => {
    const storePath = join(dir, "s.json");
    const onError = vi.fn();
    const s = make({ storePath, autoSave: true, onError });
    hooks.renameFailures.push("string failure");
    await s.addDocument({ id: "a", text: "alpha" });
    expect(onError.mock.calls[0]?.[0]).toBeInstanceOf(Error);
  });

  it("persist() fails its own write and clears the deferred error", async () => {
    const storePath = join(dir, "s.json");
    const s = make({ storePath, autoSave: true, onError: () => {} });
    hooks.alwaysFail = new Error("down");
    await s.addDocument({ id: "a", text: "alpha" });
    await expect(s.persist()).rejects.toThrow("down");
    hooks.alwaysFail = undefined;
    await expect(s.persist()).resolves.toBeUndefined();
  });

  it("import does not trigger autoSave", async () => {
    const storePath = join(dir, "s.json");
    const s = make({ storePath, autoSave: true });
    s.import([{ id: "a", text: "t", vector: [1] }]);
    await expect(readFile(storePath)).rejects.toThrow();
  });
});
