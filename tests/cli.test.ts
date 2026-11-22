import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

const root = resolve(import.meta.dirname, "..");
const temporaryPaths: string[] = [];

function createLoader() {
  const directory = mkdtempSync(join(tmpdir(), "nachos-cli-test-"));
  temporaryPaths.push(directory);
  const loader = join(directory, "loader.mjs");
  const common = `
    import { appendFileSync } from "node:fs";
    const log = (entry) => appendFileSync(process.env.NACHOS_CLI_TEST_LOG, JSON.stringify(entry) + "\\n");
  `;
  const transformers = `${common}
    export class Embedder {
      constructor(options) { this.options = options; log({ provider: "transformers", options }); }
      async init() {}
      getModelId() {
        const { model = "stub-default", dtype } = this.options;
        return dtype ? model + ":" + dtype : model;
      }
      async getDimension() { return 1; }
      getConfig() {
        const { pooling, queryPrefix, documentPrefix, maxLength } = this.options;
        return {
          cacheDir: this.options.cacheDir ?? "/stub/default-cache",
          ...(pooling !== undefined && { pooling }),
          ...(queryPrefix !== undefined && { queryPrefix }),
          ...(documentPrefix !== undefined && { documentPrefix }),
          ...(maxLength !== undefined && { maxLength }),
        };
      }
      async dispose() { log({ disposed: "transformers" }); }
      async embedBatch(lines, ...rest) {
        log({ embedBatch: { argCount: 1 + rest.length, options: rest[0] ?? null } });
        return lines.map((_, index) => [index + 0.25]);
      }
    }
  `;
  const bedrock = `${common}
    export class BedrockProvider {
      constructor(options) { this.options = options; log({ provider: "bedrock", options }); }
      async init() {}
      getModelId() { return this.options.modelId ?? "stub-bedrock-default"; }
      async getDimension() { return 2; }
      async dispose() { log({ disposed: "bedrock" }); }
      async embedBatch(lines) { return lines.map((_, index) => [index + 0.5]); }
    }
  `;
  const reranker = `${common}
    export class Reranker {
      constructor(options) { this.options = options; log({ reranker: { options } }); }
      async init() { log({ rerankerInit: true }); }
      getConfig() { return { cacheDir: this.options.cacheDir ?? "/stub/default-cache" }; }
      getModelId() {
        const { model = "stub-rerank-default", dtype } = this.options;
        return dtype ? model + ":" + dtype : model;
      }
      async rerank(query, passages) {
        log({ rerank: { query, passages } });
        if (query === "boom") throw new Error("rerank exploded");
        return passages.map((p) => p.length / 10);
      }
      async dispose() { log({ disposed: "reranker" }); }
    }
  `;
  writeFileSync(
    loader,
    `export async function resolve(specifier, context, nextResolve) {
      const modules = {
        "../dist/embedder.js": ${JSON.stringify(`data:text/javascript,${encodeURIComponent(transformers)}`)},
        "../dist/reranker.js": ${JSON.stringify(`data:text/javascript,${encodeURIComponent(reranker)}`)},
        "../dist/providers/bedrock/index.js": ${JSON.stringify(`data:text/javascript,${encodeURIComponent(bedrock)}`)},
      };
      if (modules[specifier]) return { url: modules[specifier], shortCircuit: true };
      return nextResolve(specifier, context);
    }
    `,
  );
  return { loader, log: join(directory, "providers.jsonl") };
}

function launchArgs(args: string[]) {
  const { loader, log } = createLoader();
  return {
    log,
    command: [
      "--experimental-loader",
      pathToFileURL(loader).href,
      "bin/cli.js",
      ...args,
    ],
    options: {
      cwd: root,
      env: { ...process.env, NACHOS_CLI_TEST_LOG: log },
    },
  };
}

function readEvents(log: string) {
  if (!existsSync(log)) return [];
  return readFileSync(log, "utf8")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function runCli(args: string[], input = "") {
  const { log, command, options } = launchArgs(args);
  const result = spawnSync(process.execPath, command, {
    ...options,
    encoding: "utf8",
    input,
  });
  const events = readEvents(log);
  const providers = events.filter((event) => event.provider);
  const disposed = events.flatMap((event) => event.disposed ?? []);
  const batches = events.flatMap((event) => event.embedBatch ?? []);
  const rerankers = events.flatMap((event) => event.reranker ?? []);
  const reranks = events.flatMap((event) => event.rerank ?? []);
  return { ...result, providers, disposed, batches, rerankers, reranks };
}

afterEach(() => {
  for (const path of temporaryPaths.splice(0))
    rmSync(path, { recursive: true });
});

describe("nachos-embeddings CLI", () => {
  it("returns usage errors without writing JSONL", () => {
    const result = runCli(["embed", "--stdin"]);

    expect(result.status).toBe(1);
    expect(result.stdout).toBe("");
    expect(result.stderr).toContain(
      "usage: nachos-embeddings embed --stdin --json",
    );
    expect(result.providers).toEqual([]);
  });

  it("forwards transformer model arguments and emits only JSONL vectors", () => {
    const result = runCli(
      ["embed", "--stdin", "--json", "--model", "custom-model"],
      "first\nsecond\n",
    );

    expect(result.status).toBe(0);
    expect(result.stdout).toBe("[0.25]\n[1.25]\n");
    expect(result.providers).toEqual([
      { provider: "transformers", options: { model: "custom-model" } },
    ]);
  });

  it("forwards bedrock model and region arguments", () => {
    const result = runCli(
      [
        "embed",
        "--stdin",
        "--json",
        "--provider",
        "bedrock",
        "--model",
        "amazon.titan-embed-text-v2:0",
        "--region",
        "us-west-2",
      ],
      "text\n",
    );

    expect(result.status).toBe(0);
    expect(result.stdout).toBe("[0.5]\n");
    expect(result.providers).toEqual([
      {
        provider: "bedrock",
        options: {
          modelId: "amazon.titan-embed-text-v2:0",
          region: "us-west-2",
        },
      },
    ]);
  });

  it("serve mode loads once, announces ready and answers JSON lines", () => {
    const input = [
      JSON.stringify({ id: 1, texts: ["a", "b"] }),
      "not json",
      JSON.stringify({ id: 3, texts: ["c"] }),
    ].join("\n");
    const result = runCli(["serve", "--model", "custom-model"], input + "\n");

    expect(result.status).toBe(0);
    const out = result.stdout
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));
    expect(out[0]).toEqual({
      ready: true,
      model: "custom-model",
      dimension: 1,
    });
    expect(out[1]).toEqual({ id: 1, vectors: [[0.25], [1.25]] });
    expect(out[2].id).toBeNull();
    expect(out[2].error).toContain("invalid request");
    expect(out[3]).toEqual({ id: 3, vectors: [[0.25]] });
    expect(result.providers).toHaveLength(1);
  });

  it("embed --type forwards the input type; omitted passes no options", () => {
    const typed = runCli(
      ["embed", "--stdin", "--json", "--type", "query"],
      "a\n",
    );
    expect(typed.batches).toEqual([
      { argCount: 2, options: { type: "query" } },
    ]);
    const plain = runCli(["embed", "--stdin", "--json"], "a\n");
    expect(plain.batches).toEqual([{ argCount: 1, options: null }]);
  });

  it("rejects invalid --type, --pooling and --max-length", () => {
    for (const bad of [
      ["--type", "doc"],
      ["--pooling", "max"],
      ["--max-length", "0"],
      ["--max-length", "abc"],
      ["--max-length", "1.5"],
    ]) {
      const result = runCli(["embed", "--stdin", "--json", ...bad], "a\n");
      expect(result.status).toBe(1);
      expect(result.stdout).toBe("");
      expect(result.providers).toEqual([]);
    }
  });

  it("forwards --pooling, prefixes (including empty) and --max-length", () => {
    const result = runCli(
      [
        "serve",
        "--pooling",
        "cls",
        "--query-prefix",
        "",
        "--document-prefix",
        "doc: ",
        "--max-length",
        "128",
      ],
      "",
    );
    expect(result.providers[0].options).toMatchObject({
      pooling: "cls",
      queryPrefix: "",
      documentPrefix: "doc: ",
      maxLength: 128,
    });
    expect(JSON.parse(result.stdout.trim())).toEqual({
      ready: true,
      model: "stub-default",
      dimension: 1,
      pooling: "cls",
      queryPrefix: "",
      documentPrefix: "doc: ",
      maxLength: 128,
    });
  });

  it("serve honours a per-request type and keeps old requests unchanged", () => {
    const input = [
      JSON.stringify({ id: 1, texts: ["a"] }),
      JSON.stringify({ id: 2, texts: ["a"], type: "query" }),
      JSON.stringify({ id: 3, texts: ["a"], type: "document" }),
      JSON.stringify({ id: 4, texts: ["a"], type: "bogus" }),
    ].join("\n");
    const result = runCli(["serve"], input + "\n");
    const out = result.stdout
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line));
    expect(out[1]).toEqual({ id: 1, vectors: [[0.25]] });
    expect(out[2]).toEqual({ id: 2, vectors: [[0.25]] });
    expect(out[3]).toEqual({ id: 3, vectors: [[0.25]] });
    expect(out[4]).toEqual({
      id: 4,
      error: "invalid request: type must be 'query' or 'document'",
    });
    expect(result.batches).toEqual([
      { argCount: 1, options: null },
      { argCount: 2, options: { type: "query" } },
      { argCount: 2, options: { type: "document" } },
    ]);
  });

  it("info --json reports pooling and prefixes when the provider has them", () => {
    const result = runCli(
      ["info", "--json", "--pooling", "mean", "--document-prefix", ""],
      "",
    );
    expect(JSON.parse(result.stdout.trim())).toMatchObject({
      pooling: "mean",
      documentPrefix: "",
    });
    const text = runCli(["info", "--query-prefix", "q: "], "");
    expect(text.stdout).toContain('queryPrefix: "q: "');
  });

  it("serve reports the provider's model id, not a hardcoded default", () => {
    const result = runCli(["serve"], "");
    expect(JSON.parse(result.stdout.trim())).toEqual({
      ready: true,
      model: "stub-default",
      dimension: 1,
    });
    const bedrock = runCli(["serve", "--provider", "bedrock"], "");
    expect(JSON.parse(bedrock.stdout.trim())).toEqual({
      ready: true,
      model: "stub-bedrock-default",
      dimension: 2,
    });
  });

  it("serve includes the dtype in the reported model id", () => {
    const result = runCli(["serve", "--dtype", "q8"], "");
    expect(JSON.parse(result.stdout.trim()).model).toBe("stub-default:q8");
  });

  it("serve disposes the provider on stdin EOF", () => {
    const result = runCli(["serve"], JSON.stringify({ id: 1, texts: ["a"] }));
    expect(result.status).toBe(0);
    expect(result.disposed).toEqual(["transformers"]);
  });

  it("embed disposes the provider after finishing", () => {
    const result = runCli(["embed", "--stdin", "--json"], "a\n");
    expect(result.disposed).toEqual(["transformers"]);
  });

  it("rejects an unknown provider with a usage error", () => {
    const result = runCli(["embed", "--stdin", "--json", "--provider", "nope"]);
    expect(result.status).toBe(1);
    expect(result.stdout).toBe("");
    expect(result.stderr).toContain("unknown provider 'nope'");
    expect(result.stderr).toContain("usage:");
    expect(result.providers).toEqual([]);
  });

  it.each(["--provider", "--model", "--region", "--cache-dir", "--dtype"])(
    "treats %s without a value as a usage error",
    (flag) => {
      const atEnd = runCli(["serve", flag]);
      expect(atEnd.status).toBe(1);
      expect(atEnd.stderr).toContain(`${flag} requires a value`);

      const beforeFlag = runCli(["embed", flag, "--stdin", "--json"]);
      expect(beforeFlag.status).toBe(1);
      expect(beforeFlag.stderr).toContain(`${flag} requires a value`);
      expect(beforeFlag.providers).toEqual([]);
    },
  );

  it("forwards --cache-dir and --dtype to the transformers provider", () => {
    const result = runCli(
      ["serve", "--cache-dir", "/some/cache", "--dtype", "q8"],
      "",
    );
    expect(result.providers).toEqual([
      {
        provider: "transformers",
        options: { cacheDir: "/some/cache", dtype: "q8" },
      },
    ]);
  });

  it("info --json prints provider, model, dimension and cacheDir", () => {
    const result = runCli([
      "info",
      "--json",
      "--model",
      "m",
      "--dtype",
      "q8",
      "--cache-dir",
      "/c",
    ]);
    expect(result.status).toBe(0);
    expect(JSON.parse(result.stdout)).toEqual({
      provider: "transformers",
      model: "m:q8",
      dimension: 1,
      cacheDir: "/c",
    });
    expect(result.disposed).toEqual(["transformers"]);
  });

  it("info prints human-readable lines without --json", () => {
    const result = runCli(["info"]);
    expect(result.status).toBe(0);
    expect(result.stdout).toBe(
      "provider: transformers\nmodel: stub-default\ndimension: 1\ncacheDir: /stub/default-cache\n",
    );
  });

  it("info for bedrock reports a null cacheDir", () => {
    const result = runCli(["info", "--json", "--provider", "bedrock"]);
    expect(JSON.parse(result.stdout)).toEqual({
      provider: "bedrock",
      model: "stub-bedrock-default",
      dimension: 2,
      cacheDir: null,
    });
  });

  it("serve exits cleanly and disposes when the stdout reader closes early", async () => {
    const { log, command, options } = launchArgs(["serve"]);
    const child = spawn(process.execPath, command, options);
    let stderr = "";
    child.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString()));
    const exited = new Promise<number | null>((done) =>
      child.on("close", (code) => done(code)),
    );

    // Wait for the ready line, then behave like `| head -1` and hang up.
    await new Promise<void>((ready) =>
      child.stdout.once("data", () => ready()),
    );
    child.stdout.destroy();
    // This reply has nowhere to go, so the write fails with EPIPE.
    child.stdin.write(JSON.stringify({ id: 1, texts: ["a"] }) + "\n");

    expect(await exited).toBe(0);
    expect(stderr).not.toContain("EPIPE");
    expect(readEvents(log).flatMap((event) => event.disposed ?? [])).toEqual([
      "transformers",
    ]);
  });

  describe("rerank", () => {
    const serveLines = (stdout: string) =>
      stdout
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line));

    it("one-shot rerank prints one JSON array of scores in input order", () => {
      const result = runCli(
        [
          "rerank",
          "--query",
          "q",
          "--stdin",
          "--json",
          "--model",
          "m",
          "--dtype",
          "q4",
        ],
        "aa\n\nbbbbb\n",
      );
      expect(result.status).toBe(0);
      expect(result.stdout).toBe("[0.2,0,0.5]\n");
      expect(result.rerankers).toEqual([
        { options: { model: "m", dtype: "q4" } },
      ]);
      expect(result.reranks).toEqual([
        { query: "q", passages: ["aa", "", "bbbbb"] },
      ]);
      expect(result.providers).toEqual([]);
      expect(result.disposed).toEqual(["reranker"]);
    });

    it("--rerank-model/--rerank-dtype take precedence; empty stdin gives []", () => {
      const result = runCli(
        [
          "rerank",
          "--query",
          "q",
          "--stdin",
          "--json",
          "--model",
          "m",
          "--rerank-model",
          "r",
        ],
        "x\n",
      );
      expect(result.rerankers).toEqual([{ options: { model: "r" } }]);
      const empty = runCli(["rerank", "--query", "q", "--stdin", "--json"], "");
      expect(empty.stdout).toBe("[]\n");
      expect(empty.rerankers).toEqual([]);
    });

    it("one-shot rerank needs --query, --stdin and --json; failures exit 2", () => {
      const missing = runCli(["rerank", "--stdin", "--json"], "x\n");
      expect(missing.status).toBe(1);
      expect(missing.stdout).toBe("");
      const failed = runCli(
        ["rerank", "--query", "boom", "--stdin", "--json"],
        "x\n",
      );
      expect(failed.status).toBe(2);
      expect(failed.stdout).toBe("");
      expect(failed.stderr).toContain("rerank exploded");
    });

    it("serve without rerank flags never constructs a reranker", () => {
      const result = runCli(
        ["serve", "--model", "custom-model"],
        JSON.stringify({ id: 1, texts: ["a"] }) +
          "\n" +
          JSON.stringify({ id: 2, op: "rerank", query: "q", passages: ["a"] }) +
          "\n",
      );
      const out = serveLines(result.stdout);
      expect(out[0]).toEqual({
        ready: true,
        model: "custom-model",
        dimension: 1,
      });
      expect(out[1]).toEqual({ id: 1, vectors: [[0.25]] });
      expect(out[2].id).toBe(2);
      expect(out[2].error).toContain("rerank is not enabled");
      expect(result.rerankers).toEqual([]);
    });

    it("serve --rerank-model announces support and answers op=rerank", () => {
      const input = [
        { id: 1, texts: ["a"] },
        { id: 2, op: "rerank", query: "q", passages: ["aaaa", "b", ""] },
        { id: 3, op: "rerank", query: "q", passages: [] },
        { id: 4, op: "rerank", query: "q" },
        { id: 5, op: "rerank", query: "q", passages: [1] },
        { id: 6, op: "nope" },
        { id: 7, op: "rerank", query: "boom", passages: ["a"] },
        { id: 8, op: "embed", texts: ["a", "b"] },
      ]
        .map((r) => JSON.stringify(r))
        .join("\n");
      const result = runCli(
        ["serve", "--rerank-model", "r", "--rerank-dtype", "q8"],
        input + "\n",
      );
      expect(result.status).toBe(0);
      const out = serveLines(result.stdout);
      expect(out[0]).toEqual({
        ready: true,
        model: "stub-default",
        dimension: 1,
        rerank: { model: "r:q8" },
      });
      expect(out[1]).toEqual({ id: 1, vectors: [[0.25]] });
      expect(out[2]).toEqual({ id: 2, scores: [0.4, 0.1, 0] });
      expect(out[3]).toEqual({ id: 3, scores: [] });
      expect(out[4].error).toContain("invalid request");
      expect(out[5].error).toContain("invalid request");
      expect(out[6]).toEqual({
        id: 6,
        error: "invalid request: unknown op 'nope'",
      });
      expect(out[7]).toEqual({ id: 7, error: "rerank exploded" });
      expect(out[8]).toEqual({ id: 8, vectors: [[0.25], [1.25]] });
      expect(result.rerankers).toEqual([
        { options: { model: "r", dtype: "q8" } },
      ]);
      expect(result.disposed.sort()).toEqual(["reranker", "transformers"]);
    });

    it("serve --rerank-dtype alone enables the default reranker model", () => {
      const result = runCli(["serve", "--rerank-dtype", "fp32"], "");
      expect(serveLines(result.stdout)[0].rerank).toEqual({
        model: "stub-rerank-default:fp32",
      });
    });
  });

  describe("offline and pull", () => {
    it("--offline is forwarded to the embedder and the reranker", () => {
      const embed = runCli(["embed", "--stdin", "--json", "--offline"], "a\n");
      expect(embed.providers).toEqual([
        { provider: "transformers", options: { offline: true } },
      ]);
      const rerank = runCli(
        ["rerank", "--query", "q", "--stdin", "--json", "--offline"],
        "a\n",
      );
      expect(rerank.rerankers).toEqual([{ options: { offline: true } }]);
    });

    it("without --offline the option is not passed (library env var decides)", () => {
      const embed = runCli(["embed", "--stdin", "--json"], "a\n");
      expect(embed.providers).toEqual([
        { provider: "transformers", options: {} },
      ]);
    });

    it("serve announces offline in the ready line only when enabled", () => {
      const plain = runCli(["serve"], "");
      expect(JSON.parse(plain.stdout.trim())).not.toHaveProperty("offline");
      const flagged = runCli(["serve", "--offline"], "");
      expect(JSON.parse(flagged.stdout.trim()).offline).toBe(true);
    });

    it("NACHOS_EMBEDDINGS_OFFLINE=1 also marks serve offline", () => {
      const { command, options } = launchArgs(["serve"]);
      const result = spawnSync(process.execPath, command, {
        ...options,
        env: { ...options.env, NACHOS_EMBEDDINGS_OFFLINE: "1" },
        encoding: "utf8",
        input: "",
      });
      expect(JSON.parse(result.stdout.trim()).offline).toBe(true);
    });

    it("bedrock refuses to run offline", () => {
      const result = runCli(
        ["embed", "--stdin", "--json", "--provider", "bedrock", "--offline"],
        "a\n",
      );
      expect(result.status).toBe(2);
      expect(result.stdout).toBe("");
      expect(result.stderr).toContain("cannot be used in offline mode");
      expect(result.providers).toEqual([]);
    });

    it("pull downloads an embedding model by default", () => {
      const result = runCli([
        "pull",
        "some/embed-model",
        "--dtype",
        "q8",
        "--cache-dir",
        "/c",
      ]);
      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toEqual({
        pulled: true,
        kind: "embedding",
        model: "some/embed-model:q8",
        cacheDir: "/c",
      });
      expect(result.providers).toEqual([
        {
          provider: "transformers",
          options: { model: "some/embed-model", dtype: "q8", cacheDir: "/c" },
        },
      ]);
      expect(result.disposed).toEqual(["transformers"]);
    });

    it("pull treats reranker ids as rerankers, and --kind overrides", () => {
      const named = runCli(["pull", "jinaai/jina-reranker-v1-tiny-en"]);
      expect(JSON.parse(named.stdout).kind).toBe("rerank");
      expect(named.rerankers).toEqual([
        { options: { model: "jinaai/jina-reranker-v1-tiny-en" } },
      ]);
      expect(named.providers).toEqual([]);

      const forced = runCli(["pull", "some/model", "--kind", "rerank"]);
      expect(JSON.parse(forced.stdout).kind).toBe("rerank");
      const other = runCli([
        "pull",
        "Xenova/ms-marco-MiniLM-L-6-v2",
        "--kind",
        "embedding",
      ]);
      expect(JSON.parse(other.stdout).kind).toBe("embedding");
      expect(runCli(["pull", "m", "--kind", "bogus"]).status).toBe(1);
    });

    it("pull needs a model and refuses to run offline", () => {
      expect(runCli(["pull"]).status).toBe(1);
      const offline = runCli(["pull", "some/model", "--offline"]);
      expect(offline.status).toBe(2);
      expect(offline.stdout).toBe("");
      expect(offline.stderr).toContain("pull needs network access");
      expect(offline.providers).toEqual([]);
    });
  });

  describe("models, info <model> and --normalize", () => {
    const cacheDir = () => {
      const directory = mkdtempSync(join(tmpdir(), "nachos-cli-cache-"));
      temporaryPaths.push(directory);
      return directory;
    };

    it("--no-normalize and --normalize are forwarded to the embedder", () => {
      const off = runCli(
        ["embed", "--stdin", "--json", "--no-normalize"],
        "a\n",
      );
      expect(off.providers[0].options).toEqual({ normalize: false });
      const on = runCli(["embed", "--stdin", "--json", "--normalize"], "a\n");
      expect(on.providers[0].options).toEqual({ normalize: true });
      expect(
        runCli(["embed", "--stdin", "--json"], "a\n").providers[0].options,
      ).toEqual({});
    });

    it("models lists built-in profiles and cached models as JSON", () => {
      const dir = cacheDir();
      mkdirSync(join(dir, "org", "cached-model", "onnx"), { recursive: true });
      writeFileSync(
        join(dir, "org", "cached-model", "config.json"),
        '{"model_type":"bert","hidden_size":16}',
      );
      writeFileSync(
        join(dir, "org", "cached-model", "onnx", "model_quantized.onnx"),
        "x",
      );
      const result = runCli(["models", "--json", "--cache-dir", dir]);
      expect(result.status).toBe(0);
      const report = JSON.parse(result.stdout);
      expect(report.cacheDir).toBe(dir);
      expect(report.builtin.map((m: { id: string }) => m.id)).toContain(
        "Alibaba-NLP/gte-modernbert-base",
      );
      expect(report.cached).toMatchObject([
        {
          id: "org/cached-model",
          kind: "embedding",
          dtypes: ["q8"],
          dimension: 16,
        },
      ]);
      expect(result.providers).toEqual([]);
    });

    it("models prints a readable table by default", () => {
      const result = runCli(["models", "--cache-dir", cacheDir()]);
      expect(result.stdout).toContain("Built-in profiles");
      expect(result.stdout).toContain("gte-modernbert-base");
      expect(result.stdout).toContain("(none)");
    });

    it("info <model> describes a model without loading it", () => {
      const result = runCli([
        "info",
        "Xenova/all-MiniLM-L6-v2",
        "--dtype",
        "q8",
        "--json",
        "--cache-dir",
        cacheDir(),
      ]);
      expect(result.status).toBe(0);
      expect(JSON.parse(result.stdout)).toMatchObject({
        id: "Xenova/all-MiniLM-L6-v2",
        kind: "embedding",
        cached: false,
        dimension: 384,
        profile: { pooling: "mean" },
      });
      expect(result.providers).toEqual([]);
    });

    it("info without a model keeps loading the provider", () => {
      const result = runCli(["info", "--json"]);
      expect(JSON.parse(result.stdout).provider).toBe("transformers");
      expect(result.providers).toHaveLength(1);
    });
  });
});
