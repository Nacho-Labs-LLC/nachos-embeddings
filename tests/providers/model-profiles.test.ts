import { describe, it, expect } from "vitest";
import {
  getModelProfile,
  assertInputType,
  assertPooling,
} from "../../src/providers/model-profiles.js";

const BGE_QUERY = "Represent this sentence for searching relevant passages: ";

describe("getModelProfile", () => {
  it.each([
    ["intfloat/e5-small-v2", "mean", "query: ", "passage: "],
    ["Xenova/e5-small-v2", "mean", "query: ", "passage: "],
    ["intfloat/multilingual-e5-small", "mean", "query: ", "passage: "],
    ["BAAI/bge-small-en-v1.5", "cls", BGE_QUERY, ""],
    ["Xenova/bge-base-en-v1.5", "cls", BGE_QUERY, ""],
    [
      "nomic-ai/nomic-embed-text-v1.5",
      "mean",
      "search_query: ",
      "search_document: ",
    ],
    ["Xenova/all-MiniLM-L6-v2", "mean", "", ""],
    ["some/unknown-model", "mean", "", ""],
  ])("%s", (model, pooling, queryPrefix, documentPrefix) => {
    expect(getModelProfile(model)).toEqual({
      pooling,
      queryPrefix,
      documentPrefix,
    });
  });

  it("is case-insensitive and returns a fresh copy", () => {
    expect(getModelProfile("BAAI/BGE-SMALL-EN-V1.5").pooling).toBe("cls");
    const a = getModelProfile("x");
    a.pooling = "cls";
    expect(getModelProfile("x").pooling).toBe("mean");
  });

  it("does not treat bge v1 (no instruction convention) or e5-less names as known", () => {
    expect(getModelProfile("BAAI/bge-small-en").pooling).toBe("mean");
    expect(getModelProfile("my/free5-model").queryPrefix).toBe("");
  });
});

describe("validators", () => {
  it("accepts valid values and rejects others", () => {
    expect(() => assertInputType("query")).not.toThrow();
    expect(() => assertInputType("document")).not.toThrow();
    expect(() => assertInputType("x")).toThrow(RangeError);
    expect(() => assertPooling("cls")).not.toThrow();
    expect(() => assertPooling("max")).toThrow(RangeError);
  });
});
