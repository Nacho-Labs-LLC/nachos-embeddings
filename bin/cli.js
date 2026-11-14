#!/usr/bin/env node
/**
 * nachos-embeddings CLI — thin stdin/stdout embedding shim.
 *
 * Purpose: let non-JS callers (e.g. the Nachos memory plugin's Python
 * SemanticScorer) get embeddings from this library without importing it.
 * Reads newline-delimited texts on stdin, emits one JSON array of floats
 * per line on stdout — order preserved, one output line per input line.
 *
 * Usage:
 *   nachos-embeddings embed --stdin --json
 *   nachos-embeddings embed --stdin --json --provider bedrock
 *   nachos-embeddings embed --stdin --json --provider transformers --model Xenova/all-MiniLM-L6-v2
 *   nachos-embeddings serve [--provider ...] [--model ...]
 *   nachos-embeddings info [--provider ...] [--model ...] [--json]
 *
 * Flags:
 *   embed                subcommand: one-shot embedding
 *   serve                subcommand: persistent JSON-lines server. Loads the
 *                        provider once, prints
 *                        {"ready":true,"model":"<id>","dimension":<n>},
 *                        then for each stdin line {"id":n,"texts":[...]}
 *                        replies {"id":n,"vectors":[[...]]} or
 *                        {"id":n,"error":"..."}. Exits 0 on stdin EOF.
 *   info                 subcommand: load the provider and print
 *                        {"provider","model","dimension","cacheDir"}
 *                        (key: value lines unless --json)
 *   --stdin              read texts from stdin (required; one text per line)
 *   --json               emit JSON arrays (required; the only format today)
 *   --provider <name>    'transformers' (default, local) | 'bedrock' (Titan v2)
 *   --model <id>         override model (transformers model name or bedrock modelId)
 *   --region <region>    bedrock region (default us-east-1)
 *   --cache-dir <dir>    transformers model cache (default: per-user, see
 *                        $NACHOS_EMBEDDINGS_CACHE / $XDG_CACHE_HOME / ~/.cache)
 *   --dtype <type>       transformers precision: fp32|fp16|q8|q4|... A
 *                        non-default dtype is appended to the reported model
 *                        id (e.g. "Xenova/all-MiniLM-L6-v2:q8").
 *
 * Exit codes: 0 ok, 1 usage error, 2 embedding failure.
 * All diagnostics go to stderr so stdout stays pure JSONL.
 */

import { createInterface } from "node:readline";
import { Embedder } from "../dist/embedder.js";

const PROVIDERS = ["transformers", "bedrock"];

const VALUE_OPTIONS = {
  "--provider": "provider",
  "--model": "model",
  "--region": "region",
  "--cache-dir": "cacheDir",
  "--dtype": "dtype",
};

class UsageError extends Error {}

function parseArgs(argv) {
  const args = {
    _: [],
    provider: "transformers",
    model: undefined,
    region: undefined,
    cacheDir: undefined,
    dtype: undefined,
    stdin: false,
    json: false,
  };
  for (let i = 0; i < argv.length; i++) {
    i += parseArgument(args, argv, i);
  }
  if (!PROVIDERS.includes(args.provider)) {
    throw new UsageError(
      `unknown provider '${args.provider}' (expected ${PROVIDERS.join("|")})`,
    );
  }
  return args;
}

function parseArgument(args, argv, index) {
  const argument = argv[index];
  const valueOption = VALUE_OPTIONS[argument];
  if (valueOption !== undefined) {
    const value = argv[index + 1];
    if (value === undefined || value.startsWith("--")) {
      throw new UsageError(`${argument} requires a value`);
    }
    args[valueOption] = value;
    return 1;
  }
  if (setBooleanFlag(args, argument)) return 0;
  addPositionalArgument(args, argument);
  return 0;
}

function addPositionalArgument(args, argument) {
  if (argument && !argument.startsWith("--")) args._.push(argument);
}

function setBooleanFlag(args, argument) {
  const flag = argument?.slice(2);
  if (flag !== "stdin" && flag !== "json") return false;
  args[flag] = true;
  return true;
}

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
}

// Provider currently in use, so the stdout EPIPE handler can dispose it.
let activeProvider;

async function makeProvider(args) {
  activeProvider =
    args.provider === "bedrock"
      ? await makeBedrockProvider(args)
      : await makeTransformersProvider(args);
  return activeProvider;
}

async function makeBedrockProvider({ model, region }) {
  // Lazy import so the local (transformers) path needs no AWS SDK.
  const { BedrockProvider } =
    await import("../dist/providers/bedrock/index.js");
  const provider = new BedrockProvider({
    ...(model && { modelId: model }),
    ...(region && { region }),
  });
  await provider.init();
  return provider;
}

async function makeTransformersProvider({ model, cacheDir, dtype }) {
  const provider = new Embedder({
    ...(model && { model }),
    ...(cacheDir && { cacheDir }),
    ...(dtype && { dtype }),
  });
  await provider.init();
  return provider;
}

const USAGE =
  "usage: nachos-embeddings embed --stdin --json [options]\n" +
  "       nachos-embeddings serve [options]\n" +
  "       nachos-embeddings info [--json] [options]\n" +
  "options: [--provider transformers|bedrock] [--model <id>] [--region <r>]\n" +
  "         [--cache-dir <dir>] [--dtype <fp32|fp16|q8|q4|...>]\n";

function parseLines(raw) {
  // One text per line; preserve order. Drop only a trailing empty line.
  const lines = raw.split("\n");
  if (lines.length && lines[lines.length - 1] === "") lines.pop();
  return lines;
}

function fail(stage, err) {
  process.stderr.write(`${stage}: ${err?.message ?? err}\n`);
  process.exit(2);
}

async function embedLines(provider, lines) {
  const vectors = await provider.embedBatch(lines);
  process.stdout.write(vectors.map((v) => JSON.stringify(v)).join("\n") + "\n");
}

function writeJson(value) {
  process.stdout.write(JSON.stringify(value) + "\n");
}

function parseRequest(line) {
  try {
    const request = JSON.parse(line);
    return request && typeof request === "object" ? request : {};
  } catch {
    return {};
  }
}

async function answerRequest(provider, line) {
  const request = parseRequest(line);
  const id = request.id ?? null;
  if (!Array.isArray(request.texts)) {
    return { id, error: "invalid request: expected {id, texts: [...]}" };
  }
  try {
    return { id, vectors: await provider.embedBatch(request.texts) };
  } catch (e) {
    return { id, error: String(e?.message ?? e) };
  }
}

function modelId(provider) {
  return provider.getModelId?.() ?? provider.name;
}

async function disposeProvider(provider) {
  try {
    await provider.dispose?.();
  } catch (e) {
    process.stderr.write(`dispose failed: ${e?.message ?? e}\n`);
  }
}

// A reader that closes early (`serve | head -1`) makes stdout writes fail with
// EPIPE. That is a normal way for a consumer to hang up: dispose and exit 0.
let stdoutClosing = false;
function handleStdoutError(err) {
  if (err?.code !== "EPIPE") return fail("stdout", err);
  if (stdoutClosing) return;
  stdoutClosing = true;
  const done = () => process.exit(0);
  if (activeProvider) disposeProvider(activeProvider).then(done, done);
  else done();
}

async function runServe(args) {
  const provider = await makeProvider(args).catch((e) =>
    fail("init failed", e),
  );
  writeJson({
    ready: true,
    model: modelId(provider),
    dimension: await provider.getDimension(),
  });
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  // Sequential: one in-flight request at a time keeps replies in order.
  for await (const line of lines) {
    if (line.trim()) writeJson(await answerRequest(provider, line));
  }
  await disposeProvider(provider);
}

async function runInfo(args) {
  const provider = await makeProvider(args).catch((e) =>
    fail("init failed", e),
  );
  const info = {
    provider: args.provider,
    model: modelId(provider),
    dimension: await provider.getDimension(),
    cacheDir: provider.getConfig?.().cacheDir ?? null,
  };
  if (args.json) {
    writeJson(info);
  } else {
    for (const [key, value] of Object.entries(info)) {
      process.stdout.write(`${key}: ${value ?? "-"}\n`);
    }
  }
  await disposeProvider(provider);
}

function usageError(message) {
  if (message) process.stderr.write(`${message}\n`);
  process.stderr.write(USAGE);
  process.exit(1);
}

async function main() {
  process.stdout.on("error", handleStdoutError);
  let args;
  try {
    args = parseArgs(process.argv.slice(2));
  } catch (e) {
    if (e instanceof UsageError) return usageError(e.message);
    throw e;
  }
  if (args._[0] === "serve") return runServe(args);
  if (args._[0] === "info") return runInfo(args);
  if (!isEmbedCommand(args)) return usageError();

  await runEmbedding(args);
}

function isEmbedCommand(args) {
  return args._[0] === "embed" && args.stdin && args.json;
}

async function runEmbedding(args) {
  const lines = parseLines(await readStdin());
  if (lines.length === 0) process.exit(0);

  const provider = await makeProvider(args).catch((e) =>
    fail("init failed", e),
  );
  await embedLines(provider, lines).catch((e) => fail("embed failed", e));
  await disposeProvider(provider);
}

main().catch((err) => fail("fatal", err));
