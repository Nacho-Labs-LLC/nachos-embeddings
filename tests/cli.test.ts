import {
  existsSync,
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
      getConfig() { return { cacheDir: this.options.cacheDir ?? "/stub/default-cache" }; }
      async dispose() { log({ disposed: "transformers" }); }
      async embedBatch(lines) { return lines.map((_, index) => [index + 0.25]); }
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
  writeFileSync(
    loader,
    `export async function resolve(specifier, context, nextResolve) {
      const modules = {
        "../dist/embedder.js": ${JSON.stringify(`data:text/javascript,${encodeURIComponent(transformers)}`)},
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
  return { ...result, providers, disposed };
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
});
