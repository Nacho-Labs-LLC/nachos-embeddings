import {
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import { afterEach, describe, expect, it } from "vitest";

const root = resolve(import.meta.dirname, "..");
const temporaryPaths: string[] = [];

function createLoader() {
  const directory = mkdtempSync(join(tmpdir(), "nachos-cli-test-"));
  temporaryPaths.push(directory);
  const loader = join(directory, "loader.mjs");
  const transformers = `
    import { appendFileSync } from "node:fs";
    const record = (provider, options) => appendFileSync(process.env.NACHOS_CLI_TEST_LOG, JSON.stringify({ provider, options }) + "\\n");
    export class Embedder {
      constructor(options) { record("transformers", options); }
      async init() {}
      async embedBatch(lines) { return lines.map((_, index) => [index + 0.25]); }
    }
  `;
  const bedrock = `
    import { appendFileSync } from "node:fs";
    const record = (provider, options) => appendFileSync(process.env.NACHOS_CLI_TEST_LOG, JSON.stringify({ provider, options }) + "\\n");
    export class BedrockProvider {
      constructor(options) { record("bedrock", options); }
      async init() {}
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

function runCli(args: string[], input = "") {
  const { loader, log } = createLoader();
  const result = spawnSync(
    process.execPath,
    [
      "--experimental-loader",
      pathToFileURL(loader).href,
      "bin/cli.js",
      ...args,
    ],
    {
      cwd: root,
      encoding: "utf8",
      input,
      env: { ...process.env, NACHOS_CLI_TEST_LOG: log },
    },
  );
  const providers = (existsSync(log) ? readFileSync(log, "utf8") : "")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  return { ...result, providers };
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
});
