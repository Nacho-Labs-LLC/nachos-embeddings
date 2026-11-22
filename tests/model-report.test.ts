import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { inspectModel, listCachedModels } from "../src/model-cache.js";
import {
  buildModelsReport,
  formatInfoText,
  formatModelsText,
} from "../src/model-report.js";
import {
  MODEL_CATALOG,
  findCatalogEntry,
  profileFor,
} from "../src/providers/model-catalog.js";
import { writeTinyModel } from "./tiny-model.js";

let cacheDir: string;

beforeEach(() => {
  cacheDir = mkdtempSync(join(tmpdir(), "nachos-report-test-"));
});
afterEach(() => rmSync(cacheDir, { recursive: true, force: true }));

describe("catalogue", () => {
  it("lists tested models with licence, sizes and profile", () => {
    expect(MODEL_CATALOG.length).toBeGreaterThanOrEqual(8);
    for (const entry of MODEL_CATALOG) {
      expect(entry.license).not.toBe("");
      expect(entry.sizesMB.q8).toBeGreaterThan(0);
      expect(entry.sizesMB.fp32).toBeGreaterThan(entry.sizesMB.q8 as number);
      if (entry.kind === "embedding")
        expect(entry.dimension).toBeGreaterThan(0);
    }
    expect(findCatalogEntry("xenova/ALL-minilm-l6-v2")?.dimension).toBe(384);
    expect(findCatalogEntry("nobody/x")).toBeUndefined();
  });

  it("derives the profile from the same tables the loaders use", () => {
    expect(
      profileFor("Alibaba-NLP/gte-modernbert-base", "embedding"),
    ).toMatchObject({
      pooling: "cls",
      queryPrefix: "",
    });
    expect(profileFor("jinaai/jina-reranker-v1-tiny-en", "rerank")).toEqual({
      maxLength: 1024,
      maxPositions: 8192,
    });
    expect(profileFor("Xenova/ms-marco-MiniLM-L-6-v2", "rerank")).toEqual({
      maxLength: 512,
    });
  });
});

describe("listCachedModels / inspectModel", () => {
  it("finds cached models, their dtypes, kind and dimension", async () => {
    writeTinyModel(join(cacheDir, "Xenova", "all-MiniLM-L6-v2"));
    writeTinyModel(join(cacheDir, "Xenova", "all-MiniLM-L6-v2"), {
      weights: "onnx/model.onnx",
    });
    writeTinyModel(join(cacheDir, "someone", "my-reranker"), {
      config: {
        model_type: "bert",
        architectures: ["BertForSequenceClassification"],
        hidden_size: 8,
      },
    });
    mkdirSync(join(cacheDir, "empty", "dir"), { recursive: true });
    const cached = await listCachedModels(cacheDir);
    expect(cached.map((m) => m.id)).toEqual([
      "someone/my-reranker",
      "Xenova/all-MiniLM-L6-v2",
    ]);
    const [reranker, minilm] = cached;
    expect(minilm).toMatchObject({
      kind: "embedding",
      dtypes: ["fp32", "q8"],
      dimension: 8,
      complete: true,
      modelType: "bert",
    });
    expect(reranker).toMatchObject({ kind: "rerank", dtypes: ["q8"] });
    expect(reranker?.dimension).toBeUndefined();
    expect(minilm?.bytes).toBeGreaterThan(0);
  });

  it("returns [] for a missing cache directory", async () => {
    expect(await listCachedModels(join(cacheDir, "nope"))).toEqual([]);
  });

  it("reports readiness per dtype, never downloading", async () => {
    writeTinyModel(join(cacheDir, "Xenova", "all-MiniLM-L6-v2"));
    const ready = await inspectModel("Xenova/all-MiniLM-L6-v2", {
      cacheDir,
      dtype: "q8",
    });
    expect(ready).toMatchObject({
      cached: true,
      missing: [],
      source: "hub",
      dimension: 8,
    });
    expect(ready.catalog?.license).toBe("Apache-2.0");
    const other = await inspectModel("Xenova/all-MiniLM-L6-v2", {
      cacheDir,
      dtype: "fp32",
    });
    expect(other.cached).toBe(false);
    expect(other.missing).toEqual(["onnx/model.onnx"]);
    const unknown = await inspectModel("nobody/x", { cacheDir });
    expect(unknown).toMatchObject({
      cached: false,
      kind: "embedding",
      dtype: "fp32",
    });
    expect(unknown.catalog).toBeUndefined();
  });

  it("describes a local directory and uses the reranker default dtype", async () => {
    const dir = join(cacheDir, "local-model");
    writeTinyModel(dir);
    const info = await inspectModel(dir, { cacheDir: "/unused", dtype: "q8" });
    expect(info).toMatchObject({
      source: "local",
      cached: true,
      directory: dir,
    });
    const rerank = await inspectModel("jinaai/jina-reranker-v1-tiny-en", {
      cacheDir,
    });
    expect(rerank).toMatchObject({ kind: "rerank", dtype: "q8" });
  });
});

describe("reports", () => {
  it("builds JSON and text with catalogue and cache sections", async () => {
    writeTinyModel(join(cacheDir, "Xenova", "all-MiniLM-L6-v2"));
    writeFileSync(join(cacheDir, "stray.txt"), "x");
    const report = await buildModelsReport(cacheDir);
    expect(
      report.builtin.find((m) => m.id === "Xenova/all-MiniLM-L6-v2")
        ?.cachedDtypes,
    ).toEqual(["q8"]);
    expect(report.cached).toHaveLength(1);
    const text = formatModelsText(report);
    expect(text).toContain("Apache-2.0");
    expect(text).toContain("jinaai/jina-reranker-v1-tiny-en");
    expect(text).toContain("[cached: q8]");
    expect(text).toContain("max length 1024 (up to 8192)");
    expect(text).toContain(`Cached models in ${cacheDir}`);
    const info = formatInfoText(await inspectModel("nobody/x", { cacheDir }));
    expect(info).toContain("not usable offline");
    expect(info).toContain("not in the built-in list");
  });
});
