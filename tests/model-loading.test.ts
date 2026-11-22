import { mkdtempSync, rmSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { AutoTokenizer, env } from "@huggingface/transformers";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  ModelLoadError,
  type ModelLoadOptions,
  assertEncoderArchitecture,
  explainLoadError,
  isLocalModelPath,
  loadModel,
  resolveModelRef,
  weightsFile,
} from "../src/providers/model-loading.js";
import {
  TOKENIZER_CONFIG,
  TOKENIZER_JSON,
  writeTinyModel,
} from "./tiny-model.js";

let root: string;
let requests: string[];
const realFetch = env.fetch;

/** Fake hub that counts requests and serves a config and tokenizer for `hub/model`. */
let hubConfig: Record<string, unknown>;
const fakeFetch = (async (input: unknown) => {
  const url = String(input);
  requests.push(url);
  if (url.endsWith("/config.json")) {
    return new Response(JSON.stringify(hubConfig), { status: 200 });
  }
  if (url.endsWith("/tokenizer.json")) return new Response(TOKENIZER_JSON);
  if (url.endsWith("/tokenizer_config.json")) {
    return new Response(TOKENIZER_CONFIG);
  }
  return new Response("not found", { status: 404 });
}) as typeof env.fetch;

const tokenize = (model: string, extra: ModelLoadOptions) =>
  AutoTokenizer.from_pretrained(model, { cache_dir: root, ...extra });

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), "nachos-load-test-"));
  requests = [];
  hubConfig = { model_type: "bert", architectures: ["BertModel"] };
  env.fetch = fakeFetch;
});

afterEach(() => {
  env.fetch = realFetch;
  rmSync(root, { recursive: true, force: true });
});

describe("local model directories", () => {
  it("recognises explicit paths only, so org/name stays a hub id", () => {
    for (const path of [
      "/abs/dir",
      "./dir",
      "../dir",
      "~/dir",
      "C:\\dir",
      "file:///x",
    ]) {
      expect(isLocalModelPath(path)).toBe(true);
    }
    for (const id of [
      "org/name",
      "Xenova/all-MiniLM-L6-v2",
      "name",
      "models/x",
    ]) {
      expect(isLocalModelPath(id)).toBe(false);
    }
    expect(resolveModelRef("org/name")).toBe("org/name");
    expect(resolveModelRef("~/m")).toBe(join(homedir(), "m"));
    expect(resolveModelRef("file:///tmp/m")).toBe("/tmp/m");
  });

  it("loads a complete directory with zero network requests, even when not offline", async () => {
    const dir = join(root, "mine");
    writeTinyModel(dir);
    const tokenizer = await loadModel(
      { model: dir, cacheDir: "/unused", dtype: "q8", offline: false },
      (extra) => tokenize(dir, extra),
    );
    expect(tokenizer("hello").input_ids.dims).toEqual([1, 1]);
    expect(requests).toEqual([]);
  });

  it("accepts weights at the top level (optimum export layout) by dropping the subfolder", async () => {
    const dir = join(root, "flat");
    writeTinyModel(dir, { weights: "model_quantized.onnx" });
    let seen: ModelLoadOptions = {};
    await loadModel(
      { model: dir, cacheDir: "/unused", dtype: "q8", offline: true },
      (extra) => {
        seen = extra;
        return tokenize(dir, extra);
      },
    );
    expect(seen).toMatchObject({ local_files_only: true, subfolder: "" });
  });

  it("names exactly what an incomplete directory is missing", async () => {
    const dir = join(root, "partial");
    writeTinyModel(dir, { weights: null });
    const error = await loadModel(
      { model: dir, cacheDir: "/unused", dtype: "q8", offline: false },
      () => Promise.reject(new Error("should not load")),
    ).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ModelLoadError);
    expect((error as ModelLoadError).code).toBe("local-model-incomplete");
    expect((error as Error).message).toContain("onnx/model_quantized.onnx");
    expect((error as Error).message).toContain(dir);
    expect(requests).toEqual([]);
  });

  it("rejects an unsupported architecture in a local directory", async () => {
    const dir = join(root, "gpt");
    writeTinyModel(dir, {
      config: { model_type: "gpt2", architectures: ["GPT2LMHeadModel"] },
    });
    await expect(
      loadModel(
        { model: dir, cacheDir: "/x", dtype: "q8", offline: true },
        () => Promise.resolve(1),
      ),
    ).rejects.toMatchObject({ code: "unsupported-architecture" });
  });
});

describe("architecture check", () => {
  it("accepts encoders and rejects generative and encoder-decoder models", () => {
    expect(() =>
      assertEncoderArchitecture("m", {
        model_type: "bert",
        architectures: ["BertModel"],
      }),
    ).not.toThrow();
    expect(() =>
      assertEncoderArchitecture("m", {
        model_type: "bert",
        architectures: ["BertForSequenceClassification"],
      }),
    ).not.toThrow();
    expect(() =>
      assertEncoderArchitecture("m", {
        model_type: "t5",
        is_encoder_decoder: true,
      }),
    ).toThrow(/encoder-decoder/);
    expect(() =>
      assertEncoderArchitecture("m", {
        model_type: "llama",
        architectures: ["LlamaForCausalLM"],
      }),
    ).toThrow(/LlamaForCausalLM/);
    expect(() => assertEncoderArchitecture("m", { architectures: [] })).toThrow(
      /no model_type/,
    );
  });

  it("refuses a cached decoder model before loading anything", async () => {
    writeTinyModel(join(root, "org", "gen"), {
      config: { model_type: "gpt2", architectures: ["GPT2LMHeadModel"] },
    });
    let loaded = false;
    await expect(
      loadModel(
        { model: "org/gen", cacheDir: root, dtype: "q8", offline: false },
        () => {
          loaded = true;
          return Promise.resolve(1);
        },
      ),
    ).rejects.toMatchObject({ code: "unsupported-architecture" });
    expect(loaded).toBe(false);
    expect(requests).toEqual([]);
  });

  it("checks the config before downloading weights for an uncached hub model", async () => {
    hubConfig = { model_type: "t5", is_encoder_decoder: true };
    let loaded = false;
    await expect(
      loadModel(
        { model: "hub/t5", cacheDir: root, dtype: "q8", offline: false },
        () => {
          loaded = true;
          return Promise.resolve(1);
        },
      ),
    ).rejects.toMatchObject({ code: "unsupported-architecture" });
    expect(loaded).toBe(false);
    // Only the small config was requested, never the weights.
    expect(requests.every((url) => url.endsWith("config.json"))).toBe(true);
  });
});

describe("explainLoadError", () => {
  it("maps missing ONNX weights to an actionable message", () => {
    const cause = new Error(
      'Could not locate file: "https://huggingface.co/intfloat/e5-small-v2/resolve/main/onnx/model_quantized.onnx".',
    );
    const error = explainLoadError(
      "intfloat/e5-small-v2",
      "q8",
      cause,
    ) as ModelLoadError;
    expect(error.code).toBe("no-onnx-weights");
    expect(error.message).toContain("onnx/model_quantized.onnx");
    expect(error.message).toContain("--dtype");
    expect(error.message).toContain("optimum");
    expect(error.cause).toBe(cause);
  });

  it("maps an unknown repo to not-found (the hub answers 401)", () => {
    const error = explainLoadError(
      "nobody/nothing",
      undefined,
      new Error(
        'Unauthorized access to file: "https://huggingface.co/nobody/nothing/resolve/main/config.json".',
      ),
    ) as ModelLoadError;
    expect(error.code).toBe("not-found");
    expect(error.message).toContain("HF_TOKEN");
  });

  it("maps unsupported model types and leaves other errors alone", () => {
    const unsupported = explainLoadError(
      "m",
      "q8",
      new Error("Unsupported model type: foo"),
    ) as ModelLoadError;
    expect(unsupported.code).toBe("unsupported-architecture");
    const other = new Error("fetch failed");
    expect(explainLoadError("m", "q8", other)).toBe(other);
    const already = new ModelLoadError("not-found", "m", "x");
    expect(explainLoadError("m", "q8", already)).toBe(already);
  });

  it("builds weight file names per dtype", () => {
    expect(weightsFile("q8")).toBe("onnx/model_quantized.onnx");
    expect(weightsFile(undefined)).toBe("onnx/model.onnx");
    expect(weightsFile("fp16", "")).toBe("model_fp16.onnx");
  });
});
