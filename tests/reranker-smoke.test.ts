import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { performance } from "node:perf_hooks";
import { afterAll, describe, expect, it } from "vitest";
import {
  Reranker,
  RERANK_DEFAULT_MODEL,
  RERANK_FALLBACK_MODEL,
} from "../src/reranker.js";

/**
 * Real-model smoke test, opt in with SMOKE_MODEL=1. Downloads both rerankers
 * (about 35 MB and 24 MB at q8) into $NACHOS_EMBEDDINGS_CACHE, or a temp dir.
 */
const enabled = process.env["SMOKE_MODEL"] === "1";
const cacheDir =
  process.env["NACHOS_EMBEDDINGS_CACHE"] ??
  mkdtempSync(join(tmpdir(), "nachos-rerank-smoke-"));

const query = "How many people live in Berlin?";
const relevant = "Berlin has a population of 3,520,031 registered inhabitants.";
const distractors = [
  "New York City is famous for the Metropolitan Museum of Art.",
  "The recipe calls for two cups of flour and a pinch of salt.",
  "Photosynthesis converts light energy into chemical energy in plants.",
  "The 2018 World Cup final was played in Moscow.",
];

const filler =
  "Regional planners compared transit ridership, housing density and park access across several districts, " +
  "noting seasonal commuting patterns, freight corridors and the cost of maintaining older bridges. ";

/** 25 passages of roughly 100-300 words/tokens, one of them relevant. */
function benchmarkPassages(): string[] {
  return Array.from({ length: 25 }, (_, i) => {
    const text = filler.repeat(3 + (i % 5)).trim();
    return i === 12 ? `${relevant} ${text}` : text;
  });
}

const median = (xs: number[]) =>
  [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] as number;

describe.skipIf(!enabled)("reranker smoke (real models)", () => {
  const reranker: Reranker[] = [];
  afterAll(async () => {
    await Promise.all(reranker.map((r) => r.dispose()));
  });

  for (const model of [RERANK_DEFAULT_MODEL, RERANK_FALLBACK_MODEL]) {
    it(`${model}: ranks the relevant passage first; 25-passage latency`, async () => {
      const r = new Reranker({ model, cacheDir });
      reranker.push(r);
      const loadStart = performance.now();
      await r.init();
      const loadMs = performance.now() - loadStart;

      const passages = [
        ...distractors.slice(0, 2),
        relevant,
        ...distractors.slice(2),
      ];
      const scores = await r.rerank(query, passages);
      const relevantScore = scores[2] as number;
      for (const [i, score] of scores.entries()) {
        if (i !== 2) expect(relevantScore).toBeGreaterThan(score);
      }

      const big = benchmarkPassages();
      const timings: number[] = [];
      let bigScores: number[] = [];
      for (let run = 0; run < 5; run++) {
        const start = performance.now();
        bigScores = await r.rerank(query, big);
        timings.push(performance.now() - start);
      }
      const best = bigScores.indexOf(Math.max(...bigScores));
      expect(best).toBe(12);
      // Recorded for the README / CHANGELOG; vitest prints it.
      console.log(
        `[rerank-smoke] ${model} load=${loadMs.toFixed(0)}ms median25=${median(timings).toFixed(0)}ms runs=${timings.map((t) => t.toFixed(0)).join(",")}`,
      );
    }, 600_000);
  }
});
