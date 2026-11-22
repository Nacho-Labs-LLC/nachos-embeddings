import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { AutoTokenizer, env } from "@huggingface/transformers";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { isOfflineEnv } from "../src/offline.js";
import {
  ModelNotCachedError,
  loadModel,
  missingCacheFiles,
} from "../src/providers/model-loading.js";

/**
 * These tests use the real Transformers.js with a tiny hand-made WordPiece
 * tokenizer and a fake `env.fetch`, so they show what actually reaches the
 * network without downloading anything.
 */
const MODEL = "test/tiny-model";

const TOKENIZER_JSON = JSON.stringify({
  version: "1.0",
  truncation: null,
  padding: null,
  added_tokens: [
    { id: 0, content: "[PAD]", special: true },
    { id: 1, content: "[UNK]", special: true },
    { id: 2, content: "[CLS]", special: true },
    { id: 3, content: "[SEP]", special: true },
  ].map((t) => ({
    ...t,
    single_word: false,
    lstrip: false,
    rstrip: false,
    normalized: false,
  })),
  normalizer: { type: "BertNormalizer", lowercase: true },
  pre_tokenizer: { type: "BertPreTokenizer" },
  post_processor: null,
  decoder: null,
  model: {
    type: "WordPiece",
    unk_token: "[UNK]",
    continuing_subword_prefix: "##",
    max_input_chars_per_word: 100,
    vocab: { "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, hello: 4 },
  },
});
const TOKENIZER_CONFIG = JSON.stringify({
  tokenizer_class: "BertTokenizer",
  model_max_length: 512,
  unk_token: "[UNK]",
  pad_token: "[PAD]",
  cls_token: "[CLS]",
  sep_token: "[SEP]",
});
const FILES: Record<string, string> = {
  "config.json": JSON.stringify({ model_type: "bert" }),
  "tokenizer.json": TOKENIZER_JSON,
  "tokenizer_config.json": TOKENIZER_CONFIG,
  "onnx/model_quantized.onnx": "not a real model",
};

let cacheDir: string;
let requests: string[];
const realFetch = env.fetch;

/** Fake hub: serves the tokenizer files, 404 for everything else. */
const fakeFetch = (async (input: unknown) => {
  const url = String(input);
  requests.push(url);
  const file = Object.keys(FILES).find((name) => url.endsWith(`/${name}`));
  return file && file.startsWith("tokenizer")
    ? new Response(FILES[file], { status: 200 })
    : new Response("not found", { status: 404 });
}) as typeof env.fetch;

function seedCache(files = Object.keys(FILES), model = MODEL) {
  for (const name of files) {
    const path = join(cacheDir, model, name);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, FILES[name] as string);
  }
}

const loadTokenizer = (offline: boolean) =>
  loadModel({ model: MODEL, cacheDir, dtype: "q8", offline }, (extra) =>
    AutoTokenizer.from_pretrained(MODEL, { cache_dir: cacheDir, ...extra }),
  );

beforeEach(() => {
  cacheDir = mkdtempSync(join(tmpdir(), "nachos-offline-test-"));
  requests = [];
  env.fetch = fakeFetch;
});

afterEach(() => {
  env.fetch = realFetch;
  rmSync(cacheDir, { recursive: true, force: true });
});

describe("cache-first loading", () => {
  it("a fully cached model loads with zero network requests", async () => {
    seedCache();
    const tokenizer = await loadTokenizer(false);
    expect(tokenizer("hello").input_ids.dims).toEqual([1, 1]);
    expect(requests).toEqual([]);
  });

  it("control: the plain Transformers.js load of a cached model does hit the network", async () => {
    // A model id no other test touches: Transformers.js memoizes its
    // tokenizer-file lookups per process.
    const model = "test/control-model";
    seedCache(undefined, model);
    const saved = env.allowLocalModels;
    env.allowLocalModels = false;
    try {
      await AutoTokenizer.from_pretrained(model, { cache_dir: cacheDir });
    } finally {
      env.allowLocalModels = saved;
    }
    expect(requests.length).toBeGreaterThan(0);
  });

  it("downloads only when files are missing, then never again", async () => {
    // Only config + weights cached: the tokenizer files must be fetched.
    seedCache(["config.json", "onnx/model_quantized.onnx"]);
    await loadTokenizer(false);
    expect(requests.length).toBeGreaterThan(0);
    expect(
      await missingCacheFiles({ model: MODEL, cacheDir, dtype: "q8" }),
    ).toEqual([]);

    requests.length = 0;
    await loadTokenizer(false);
    expect(requests).toEqual([]);
  });

  it("restores the global Transformers.js env after loading", async () => {
    seedCache();
    const before = {
      allowLocalModels: env.allowLocalModels,
      allowRemoteModels: env.allowRemoteModels,
      localModelPath: env.localModelPath,
    };
    await loadTokenizer(false);
    await loadTokenizer(true);
    expect({
      allowLocalModels: env.allowLocalModels,
      allowRemoteModels: env.allowRemoteModels,
      localModelPath: env.localModelPath,
    }).toEqual(before);
  });
});

describe("offline mode", () => {
  it("loads a cached model with zero network requests", async () => {
    seedCache();
    await loadTokenizer(true);
    expect(requests).toEqual([]);
  });

  it("fails clearly, without any request, when the model is not cached", async () => {
    const error = await loadTokenizer(true).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ModelNotCachedError);
    const message = (error as Error).message;
    expect(message).toContain(MODEL);
    expect(message).toContain("offline mode is on");
    expect(message).toContain(`nachos-embeddings pull ${MODEL}`);
    expect(message).toContain("config.json");
    expect(requests).toEqual([]);
  });

  it("names the missing weights file for the requested dtype", async () => {
    seedCache(["config.json", "tokenizer.json", "tokenizer_config.json"]);
    expect(
      await missingCacheFiles({ model: MODEL, cacheDir, dtype: "q8" }),
    ).toEqual(["onnx/model_quantized.onnx"]);
    expect(await missingCacheFiles({ model: MODEL, cacheDir })).toEqual([
      "onnx/model.onnx",
    ]);
    await expect(loadTokenizer(true)).rejects.toThrow(/model_quantized\.onnx/);
    expect(requests).toEqual([]);
  });

  it("recognises the offline environment variable", () => {
    expect(isOfflineEnv({})).toBe(false);
    expect(isOfflineEnv({ NACHOS_EMBEDDINGS_OFFLINE: "0" })).toBe(false);
    for (const value of ["1", "true", "TRUE", "yes", "on"]) {
      expect(isOfflineEnv({ NACHOS_EMBEDDINGS_OFFLINE: value })).toBe(true);
    }
  });
});
