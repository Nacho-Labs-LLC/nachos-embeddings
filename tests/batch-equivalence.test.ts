import { describe, it, expect } from "vitest";
import { Embedder } from "../src/embedder.js";

/**
 * Real-model check, run with SMOKE_MODEL=1 (downloads ~90MB once;
 * honours NACHOS_EMBEDDINGS_CACHE).
 * Padded batches must match single-text calls despite different padding.
 */
const cosine = (a: number[], b: number[]) =>
  a.reduce((sum, x, i) => sum + x * (b[i] ?? 0), 0);

describe.skipIf(process.env["SMOKE_MODEL"] !== "1")(
  "embedBatch vs embed (real model)",
  () => {
    it("padded batches match single-text vectors (cosine > 0.999)", async () => {
      const embedder = new Embedder({ model: "Xenova/all-MiniLM-L6-v2" });
      await embedder.init();
      const texts = [
        "hi",
        "Breakfast tacos are the user's favorite morning meal.",
        "word ".repeat(200).trim(),
        "The deployment pipeline runs on GitHub Actions and ships on tags.",
        "",
        "short one",
        "Kubernetes pods restart when they run out of memory limits.",
        "She enjoys hiking in the mountains on weekends.",
        "another moderately sized sentence for padding variety",
        "x",
      ];
      const batch = await embedder.embedBatch(texts);
      expect(batch).toHaveLength(texts.length);
      for (const [i, text] of texts.entries()) {
        const single = await embedder.embed(text);
        expect(cosine(batch[i] as number[], single)).toBeGreaterThan(0.999);
      }
      await embedder.dispose();
    }, 300_000);
  },
);
