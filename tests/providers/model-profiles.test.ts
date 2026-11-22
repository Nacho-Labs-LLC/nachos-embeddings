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
    ["Snowflake/snowflake-arctic-embed-xs", "cls", BGE_QUERY, ""],
    ["Snowflake/snowflake-arctic-embed-s", "cls", BGE_QUERY, ""],
    ["Snowflake/snowflake-arctic-embed-m", "cls", BGE_QUERY, ""],
    ["Snowflake/snowflake-arctic-embed-m-v1.5", "cls", BGE_QUERY, ""],
    ["Snowflake/snowflake-arctic-embed-m-long", "cls", BGE_QUERY, ""],
    ["Snowflake/snowflake-arctic-embed-l", "cls", BGE_QUERY, ""],
    ["Snowflake/snowflake-arctic-embed-m-v2.0", "cls", "query: ", ""],
    ["Snowflake/snowflake-arctic-embed-l-v2.0", "cls", "query: ", ""],
    ["Alibaba-NLP/gte-modernbert-base", "cls", "", ""],
    ["Xenova/all-MiniLM-L6-v2", "mean", "", ""],
    ["some/unknown-model", "mean", "", ""],
  ])("%s", (model, pooling, queryPrefix, documentPrefix) => {
    expect(getModelProfile(model)).toMatchObject({
      pooling,
      queryPrefix,
      documentPrefix,
    });
  });

  it("requires layer norm only for nomic-embed-text-v1.5", () => {
    expect(getModelProfile("nomic-ai/nomic-embed-text-v1.5").layerNorm).toBe(
      true,
    );
    for (const id of [
      "BAAI/bge-small-en-v1.5",
      "intfloat/e5-small-v2",
      "Snowflake/snowflake-arctic-embed-s",
      "Alibaba-NLP/gte-modernbert-base",
      "Xenova/all-MiniLM-L6-v2",
    ]) {
      expect(getModelProfile(id).layerNorm).toBe(false);
    }
  });

  it("does not treat other gte models as modernbert", () => {
    expect(getModelProfile("thenlper/gte-base").pooling).toBe("mean");
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
