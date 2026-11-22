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
 *   --type <t>           embed only: 'query' | 'document' (default). Selects the
 *                        model's query/document prefix. In serve, set per
 *                        request with an optional "type" field.
 *   --query-prefix <s>   override the model's query prefix ("" disables it)
 *   --document-prefix <s> override the model's document prefix
 *   --pooling <mode>     mean | cls (default: per model; bge uses cls)
 *   --max-length <n>     truncate inputs to n tokens (can only lower the
 *                        tokenizer's own limit; default: tokenizer limit)
 *
 *
 *   rerank               subcommand: one-shot cross-encoder rerank. Reads one
 *                        passage per stdin line and prints ONE line: a JSON
 *                        array of scores (sigmoid of the model logit, in
 *                        (0,1), higher = more relevant) in input order. Uses
 *                        --model/--dtype/--max-length/--cache-dir for the
 *                        reranker (default jinaai/jina-reranker-v1-tiny-en,
 *                        q8, 512 tokens); --rerank-model/--rerank-dtype also
 *                        work and take precedence.
 *   --query <q>          rerank only: the query text
 *   --rerank-max-length <n> serve: truncation length of the reranker (its own
 *                        setting; --max-length applies to the embedding model).
 *                        In `rerank` it defaults to --max-length.
 *   --rerank-model <id>  serve: load this reranker at startup and accept
 *   --rerank-dtype <t>   {"id","op":"rerank","query","passages":[...]}
 *                        requests, replying {"id","scores":[...]}. Either flag
 *                        enables it; the ready line then also carries
 *                        "rerank":{"model":"<id>"}. Without them no reranker
 *                        is downloaded or loaded.
 *
 *   --offline            never touch the network: load models from the cache only
 *                        and fail clearly if one is missing (also enabled by
 *                        NACHOS_EMBEDDINGS_OFFLINE=1). Without it, a fully
 *                        cached model still loads with no network request; a
 *                        download happens only when files are missing.
 *   models               subcommand: list the built-in model profiles (dims,
 *                        licence, download size per dtype, pooling/prefixes)
 *                        and the models in the cache. --json for machines.
 *   info <model>         subcommand: describe a model id or a local directory
 *                        without loading or downloading it (profile, cached
 *                        files, architecture, dimension). `info` without an id
 *                        keeps its old meaning (load the default provider).
 *   --normalize          L2-normalize vectors (default) / --no-normalize for
 *                        raw pooled vectors. A non-default value changes the
 *                        reported model id.
 *   <model> may also be a local directory (./dir, /abs/dir, ~/dir) holding
 *                        config.json, tokenizer.json, tokenizer_config.json and
 *                        the ONNX weights: fully offline.
 *   pull <model>         subcommand: download a model into the cache and print
 *                        {"pulled":true,"kind","model","cacheDir"}. Uses
 *                        --dtype/--cache-dir; --kind embedding|rerank overrides
 *                        the guess (rerank when the id names a reranker).
 *                        Refuses to run in offline mode.
 *
 * Exit codes: 0 ok, 1 usage error, 2 embedding failure.
 * All diagnostics go to stderr so stdout stays pure JSONL.
 */

import { createInterface } from "node:readline";
import { Embedder } from "../dist/embedder.js";
import { OFFLINE_ENV_VAR, isOfflineEnv } from "../dist/offline.js";

const PROVIDERS = ["transformers", "bedrock"];

const VALUE_OPTIONS = {
  "--provider": "provider",
  "--model": "model",
  "--region": "region",
  "--cache-dir": "cacheDir",
  "--dtype": "dtype",
  "--type": "type",
  "--query-prefix": "queryPrefix",
  "--document-prefix": "documentPrefix",
  "--pooling": "pooling",
  "--max-length": "maxLength",
  "--rerank-max-length": "rerankMaxLength",
  "--query": "query",
  "--rerank-model": "rerankModel",
  "--rerank-dtype": "rerankDtype",
  "--kind": "kind",
};

const INPUT_TYPES = ["query", "document"];

class UsageError extends Error {}

function parseArgs(argv) {
  const args = {
    _: [],
    provider: "transformers",
    model: undefined,
    region: undefined,
    cacheDir: undefined,
    dtype: undefined,
    type: undefined,
    queryPrefix: undefined,
    documentPrefix: undefined,
    pooling: undefined,
    maxLength: undefined,
    rerankMaxLength: undefined,
    query: undefined,
    rerankModel: undefined,
    rerankDtype: undefined,
    kind: undefined,
    normalize: undefined,
    offline: false,
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
  if (args.type !== undefined && !INPUT_TYPES.includes(args.type)) {
    throw new UsageError(
      `unknown type '${args.type}' (expected ${INPUT_TYPES.join("|")})`,
    );
  }
  if (args.pooling !== undefined && !["mean", "cls"].includes(args.pooling)) {
    throw new UsageError(
      `unknown pooling '${args.pooling}' (expected mean|cls)`,
    );
  }
  if (args.kind !== undefined && !["embedding", "rerank"].includes(args.kind)) {
    throw new UsageError(
      `unknown kind '${args.kind}' (expected embedding|rerank)`,
    );
  }
  for (const [key, flag] of [
    ["maxLength", "--max-length"],
    ["rerankMaxLength", "--rerank-max-length"],
  ]) {
    if (args[key] === undefined) continue;
    if (!/^[1-9]\d*$/.test(args[key])) {
      throw new UsageError(`${flag} must be a positive integer`);
    }
    args[key] = Number(args[key]);
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
  if (flag === "normalize" || flag === "no-normalize") {
    args.normalize = flag === "normalize";
    return true;
  }
  if (flag !== "stdin" && flag !== "json" && flag !== "offline") return false;
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
let activeReranker;

async function makeProvider(args) {
  activeProvider =
    args.provider === "bedrock"
      ? await makeBedrockProvider(args)
      : await makeTransformersProvider(args);
  return activeProvider;
}

async function makeBedrockProvider(args) {
  const { model, region } = args;
  // Lazy import so the local (transformers) path needs no AWS SDK.
  if (isOffline(args)) {
    throw new Error(
      `bedrock is a network service and cannot be used in offline mode (--offline or ${OFFLINE_ENV_VAR}=1)`,
    );
  }
  const { BedrockProvider } =
    await import("../dist/providers/bedrock/index.js");
  const provider = new BedrockProvider({
    ...(model && { modelId: model }),
    ...(region && { region }),
  });
  await provider.init();
  return provider;
}

async function makeTransformersProvider(args) {
  const { model, cacheDir, dtype, pooling, queryPrefix, documentPrefix } = args;
  const provider = new Embedder({
    ...(model && { model }),
    ...(cacheDir && { cacheDir }),
    ...(dtype && { dtype }),
    ...(pooling && { pooling }),
    // An empty prefix is meaningful (it disables the model's built-in one).
    ...(queryPrefix !== undefined && { queryPrefix }),
    ...(documentPrefix !== undefined && { documentPrefix }),
    ...(args.maxLength !== undefined && { maxLength: args.maxLength }),
    ...(args.normalize !== undefined && { normalize: args.normalize }),
    ...(args.offline && { offline: true }),
  });
  await provider.init();
  return provider;
}

const USAGE =
  "usage: nachos-embeddings embed --stdin --json [options]\n" +
  "       nachos-embeddings rerank --query <q> --stdin --json [options]\n" +
  "       nachos-embeddings pull <model> [--dtype <type>] [--kind embedding|rerank]\n" +
  "       nachos-embeddings models [--json]\n" +
  "       nachos-embeddings info <model> [--dtype <type>] [--json]\n" +
  "       nachos-embeddings serve [options]\n" +
  "       nachos-embeddings info [--json] [options]\n" +
  "options: [--provider transformers|bedrock] [--model <id>] [--region <r>]\n" +
  "         [--cache-dir <dir>] [--dtype <fp32|fp16|q8|q4|...>]\n" +
  "         [--type query|document] [--pooling mean|cls] [--max-length <tokens>]\n" +
  "         [--query-prefix <text>] [--document-prefix <text>] [--normalize|--no-normalize]\n" +
  "         [--offline]   never use the network (also NACHOS_EMBEDDINGS_OFFLINE=1)\n" +
  "serve:   [--rerank-model <id>] [--rerank-dtype <type>] enable op=rerank\n" +
  "         [--rerank-max-length <tokens>]   reranker truncation (--max-length stays with the embedder)\n";

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

// Only pass options when a type was requested, so the call is identical to
// the pre-0.6 one otherwise (and providers without options never see them).
function embedBatchWithType(provider, texts, type) {
  return type === undefined
    ? provider.embedBatch(texts)
    : provider.embedBatch(texts, { type });
}

async function embedLines(provider, lines, type) {
  const vectors = await embedBatchWithType(provider, lines, type);
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

async function answerRequest(provider, line, reranker) {
  const request = parseRequest(line);
  const id = request.id ?? null;
  if (request.op !== undefined && request.op !== "embed") {
    return answerOp(id, request, reranker);
  }
  if (!Array.isArray(request.texts)) {
    return { id, error: "invalid request: expected {id, texts: [...]}" };
  }
  if (request.type !== undefined && !INPUT_TYPES.includes(request.type)) {
    return { id, error: "invalid request: type must be 'query' or 'document'" };
  }
  try {
    return {
      id,
      vectors: await embedBatchWithType(provider, request.texts, request.type),
    };
  } catch (e) {
    return { id, error: String(e?.message ?? e) };
  }
}

async function answerOp(id, request, reranker) {
  if (request.op !== "rerank") {
    return { id, error: `invalid request: unknown op '${request.op}'` };
  }
  if (!reranker) {
    return {
      id,
      error: "rerank is not enabled (start serve with --rerank-model)",
    };
  }
  const { query, passages } = request;
  if (
    typeof query !== "string" ||
    !Array.isArray(passages) ||
    passages.some((passage) => typeof passage !== "string")
  ) {
    return {
      id,
      error:
        "invalid request: expected {id, op: 'rerank', query: string, passages: [string, ...]}",
    };
  }
  try {
    return { id, scores: await reranker.rerank(query, passages) };
  } catch (e) {
    return { id, error: String(e?.message ?? e) };
  }
}

// Active embedding settings, when the provider reports them (transformers).
// Keys are omitted otherwise, so other providers' output is unchanged.
function embeddingSettings(provider) {
  const config = provider.getConfig?.() ?? {};
  const settings = {};
  for (const key of ["pooling", "queryPrefix", "documentPrefix", "maxLength"]) {
    if (config[key] !== undefined) settings[key] = config[key];
  }
  // Only reported when it differs from the default (L2-normalized vectors).
  if (config.normalize === false) settings.normalize = false;
  return settings;
}

// --offline, or NACHOS_EMBEDDINGS_OFFLINE=1 in the environment.
function isOffline(args) {
  return args.offline || isOfflineEnv();
}

function modelId(provider) {
  return provider.getModelId?.() ?? provider.name;
}

async function disposeProvider(provider) {
  try {
    await provider?.dispose?.();
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
  Promise.all(
    [activeReranker, activeProvider]
      .filter(Boolean)
      .map((resource) => disposeProvider(resource)),
  ).then(done, done);
}

// The reranker is a separate model from the embedding provider. It is only
// imported and loaded when asked for, so plain serve never touches it.
async function makeReranker(args, { model, dtype, maxLength }) {
  const { Reranker } = await import("../dist/reranker.js");
  const reranker = new Reranker({
    ...(model && { model }),
    ...(dtype && { dtype }),
    ...(args.cacheDir && { cacheDir: args.cacheDir }),
    ...(maxLength !== undefined && { maxLength }),
    ...(args.offline && { offline: true }),
  });
  await reranker.init();
  activeReranker = reranker;
  return reranker;
}

async function runServe(args) {
  const provider = await makeProvider(args).catch((e) =>
    fail("init failed", e),
  );
  const rerankEnabled =
    args.rerankModel !== undefined || args.rerankDtype !== undefined;
  const reranker = rerankEnabled
    ? await makeReranker(args, {
        model: args.rerankModel,
        dtype: args.rerankDtype,
        // --max-length belongs to the embedding model in serve.
        maxLength: args.rerankMaxLength,
      }).catch((e) => fail("rerank init failed", e))
    : undefined;
  writeJson({
    ready: true,
    model: modelId(provider),
    dimension: await provider.getDimension(),
    ...embeddingSettings(provider),
    ...(reranker && { rerank: { model: reranker.getModelId() } }),
    ...(isOffline(args) && { offline: true }),
  });
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  // Sequential: one in-flight request at a time keeps replies in order.
  for await (const line of lines) {
    if (line.trim()) writeJson(await answerRequest(provider, line, reranker));
  }
  await disposeProvider(reranker);
  await disposeProvider(provider);
}

async function resolveCacheDir(args) {
  const { resolveDefaultCacheDir } =
    await import("../dist/providers/transformers-provider.js");
  return args.cacheDir ?? resolveDefaultCacheDir();
}

// `models`: built-in profiles plus what is cached. Reads the disk only.
async function runModels(args) {
  const { buildModelsReport, formatModelsText } =
    await import("../dist/model-report.js");
  const report = await buildModelsReport(await resolveCacheDir(args));
  if (args.json) writeJson(report);
  else process.stdout.write(formatModelsText(report));
}

// `info <model>`: what a model id is, without loading or downloading it.
async function runModelInfo(args) {
  const { inspectModel } = await import("../dist/model-cache.js");
  const { formatInfoText } = await import("../dist/model-report.js");
  const info = await inspectModel(args._[1], {
    cacheDir: await resolveCacheDir(args),
    ...(args.dtype && { dtype: args.dtype }),
    ...(args.kind && { kind: args.kind === "rerank" ? "rerank" : "embedding" }),
  });
  if (args.json) writeJson(info);
  else process.stdout.write(formatInfoText(info));
}

async function runInfo(args) {
  if (args._[1] !== undefined) return runModelInfo(args);
  const provider = await makeProvider(args).catch((e) =>
    fail("init failed", e),
  );
  const info = {
    provider: args.provider,
    model: modelId(provider),
    dimension: await provider.getDimension(),
    cacheDir: provider.getConfig?.().cacheDir ?? null,
    ...embeddingSettings(provider),
  };
  if (args.json) {
    writeJson(info);
  } else {
    for (const [key, value] of Object.entries(info)) {
      const shown =
        typeof value === "string" && key.endsWith("Prefix")
          ? JSON.stringify(value)
          : (value ?? "-");
      process.stdout.write(`${key}: ${shown}\n`);
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
  if (args._[0] === "rerank") return runRerank(args);
  if (args._[0] === "pull") return runPull(args);
  if (args._[0] === "models") return runModels(args);
  if (!isEmbedCommand(args)) return usageError();

  await runEmbedding(args);
}

async function runRerank(args) {
  if (args.query === undefined || !args.stdin || !args.json) {
    return usageError("rerank requires --query <q> --stdin --json");
  }
  const passages = parseLines(await readStdin());
  if (passages.length === 0) {
    writeJson([]);
    return;
  }
  const reranker = await makeReranker(args, {
    model: args.rerankModel ?? args.model,
    dtype: args.rerankDtype ?? args.dtype,
    maxLength: args.rerankMaxLength ?? args.maxLength,
  }).catch((e) => fail("init failed", e));
  const scores = await reranker
    .rerank(args.query, passages)
    .catch((e) => fail("rerank failed", e));
  writeJson(scores);
  await disposeProvider(reranker);
}

// Download a model into the cache so later runs (or air-gapped machines that
// copy the cache directory) need no network. This is the one command that
// reaches the network on purpose, so it refuses to run in offline mode.
async function runPull(args) {
  const model = args._[1];
  if (!model) return usageError("pull requires a model id");
  if (isOffline(args)) {
    return fail(
      "pull failed",
      new Error(
        `pull needs network access but offline mode is on (--offline or ${OFFLINE_ENV_VAR}=1); unset it to download`,
      ),
    );
  }
  const { looksLikeReranker } =
    await import("../dist/providers/reranker-profiles.js");
  const kind = args.kind ?? (looksLikeReranker(model) ? "rerank" : "embedding");
  const options = {
    model,
    ...(args.dtype && { dtype: args.dtype }),
    ...(args.cacheDir && { cacheDir: args.cacheDir }),
  };
  let loaded;
  if (kind === "rerank") {
    loaded = await makeReranker(args, {
      model,
      dtype: args.dtype,
      maxLength: args.rerankMaxLength ?? args.maxLength,
    }).catch((e) => fail("pull failed", e));
  } else {
    loaded = new Embedder(options);
    await loaded.init().catch((e) => fail("pull failed", e));
  }
  writeJson({
    pulled: true,
    kind,
    model: loaded.getModelId(),
    cacheDir: loaded.getConfig().cacheDir,
  });
  await disposeProvider(loaded);
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
  await embedLines(provider, lines, args.type).catch((e) =>
    fail("embed failed", e),
  );
  await disposeProvider(provider);
}

main().catch((err) => fail("fatal", err));
