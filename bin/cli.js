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
 *
 * Flags:
 *   embed                subcommand (required)
 *   --stdin              read texts from stdin (required; one text per line)
 *   --json               emit JSON arrays (required; the only format today)
 *   --provider <name>    'transformers' (default, local) | 'bedrock' (Titan v2)
 *   --model <id>         override model (transformers model name or bedrock modelId)
 *   --region <region>    bedrock region (default us-east-1)
 *
 * Exit codes: 0 ok, 1 usage error, 2 embedding failure.
 * All diagnostics go to stderr so stdout stays pure JSONL.
 */

import { Embedder } from '../dist/embedder.js';

function parseArgs(argv) {
  const args = { _: [], provider: 'transformers', model: undefined, region: undefined,
                 stdin: false, json: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--stdin') args.stdin = true;
    else if (a === '--json') args.json = true;
    else if (a === '--provider') args.provider = argv[++i];
    else if (a === '--model') args.model = argv[++i];
    else if (a === '--region') args.region = argv[++i];
    else if (!a.startsWith('--')) args._.push(a);
  }
  return args;
}

async function readStdin() {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString('utf8');
}

async function makeProvider(args) {
  if (args.provider === 'bedrock') {
    // Lazy import so the local (transformers) path needs no AWS SDK.
    const { BedrockProvider } = await import('../dist/providers/bedrock/index.js');
    const cfg = {};
    if (args.model) cfg.modelId = args.model;
    if (args.region) cfg.region = args.region;
    const p = new BedrockProvider(cfg);
    await p.init();
    return p;
  }
  const cfg = {};
  if (args.model) cfg.model = args.model;
  const e = new Embedder(cfg);
  await e.init();
  return e;
}

const USAGE =
  "usage: nachos-embeddings embed --stdin --json " +
  "[--provider transformers|bedrock] [--model <id>] [--region <r>]\n";

function parseLines(raw) {
  // One text per line; preserve order. Drop only a trailing empty line.
  const lines = raw.split('\n');
  if (lines.length && lines[lines.length - 1] === '') lines.pop();
  return lines;
}

function fail(stage, err) {
  process.stderr.write(`${stage}: ${err?.message ?? err}\n`);
  process.exit(2);
}

async function embedLines(provider, lines) {
  const vectors = await provider.embedBatch(lines);
  process.stdout.write(vectors.map((v) => JSON.stringify(v)).join('\n') + '\n');
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args._[0] !== 'embed' || !args.stdin || !args.json) {
    process.stderr.write(USAGE);
    process.exit(1);
  }

  const lines = parseLines(await readStdin());
  if (lines.length === 0) process.exit(0);

  const provider = await makeProvider(args).catch((e) => fail('init failed', e));
  await embedLines(provider, lines).catch((e) => fail('embed failed', e));
}

main().catch((err) => fail('fatal', err));
