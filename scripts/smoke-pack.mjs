#!/usr/bin/env node
/**
 * Smoke test of the packed npm tarball.
 *
 * Packs the package, extracts it, checks the file list and the executable
 * bit, then runs the extracted CLI. Model-backed checks (info, embed, serve, rerank)
 * only run when SMOKE_MODEL=1; point NACHOS_EMBEDDINGS_CACHE at a persistent
 * directory to reuse a downloaded model.
 *
 * The extracted package borrows the repo's node_modules via symlink, so this
 * verifies the tarball's own files, not its dependency resolution.
 */
import { spawn, spawnSync } from "node:child_process";
import {
  mkdtempSync,
  rmSync,
  symlinkSync,
  mkdirSync,
  existsSync,
  readFileSync,
  cpSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const repoRoot = resolve(import.meta.dirname, "..");
const withModel = process.env.SMOKE_MODEL === "1";
const MODEL_TIMEOUT_MS = 10 * 60_000;

const REQUIRED = [
  "dist/index.js",
  "dist/index.d.ts",
  "bin/cli.js",
  "LICENSE",
  "README.md",
];
const FORBIDDEN = [
  { test: (f) => f.startsWith("src/"), why: "source directory" },
  { test: (f) => f.startsWith("tests/"), why: "tests directory" },
  {
    test: (f) => f.endsWith(".map"),
    why: "source map (sources are not shipped)",
  },
];

const failures = [];
function check(ok, message) {
  console.log(`${ok ? "ok  " : "FAIL"} ${message}`);
  if (!ok) failures.push(message);
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { encoding: "utf8", ...options });
  if (result.error) throw result.error;
  return result;
}

function packTarball(workDir) {
  const packed = run(
    "npm",
    ["pack", "--json", "--pack-destination", workDir, "--ignore-scripts"],
    { cwd: repoRoot },
  );
  if (packed.status !== 0)
    throw new Error(`npm pack failed:\n${packed.stderr}`);
  const [entry] = JSON.parse(packed.stdout);
  return entry;
}

function extract(workDir, filename) {
  const target = join(workDir, "extracted");
  mkdirSync(target);
  const result = run("tar", ["-xzf", join(workDir, filename), "-C", target]);
  if (result.status !== 0) throw new Error(`tar failed:\n${result.stderr}`);
  const pkg = join(target, "package");
  symlinkSync(join(repoRoot, "node_modules"), join(pkg, "node_modules"), "dir");
  return pkg;
}

function checkFiles(entry) {
  const names = entry.files.map((f) => f.path);
  for (const required of REQUIRED) {
    check(names.includes(required), `tarball contains ${required}`);
  }
  for (const { test, why } of FORBIDDEN) {
    const hits = names.filter(test);
    check(
      hits.length === 0,
      `tarball has no ${why}${hits.length ? `: ${hits.slice(0, 3)}` : ""}`,
    );
  }
  const cli = entry.files.find((f) => f.path === "bin/cli.js");
  check(
    cli !== undefined && (cli.mode & 0o111) === 0o111,
    `bin/cli.js is executable in the tarball (mode ${cli ? cli.mode.toString(8) : "missing"})`,
  );
}

function checkUsage(cli, env) {
  const result = run(process.execPath, [cli], { env, input: "" });
  check(result.status === 1, `no-args exits 1 (got ${result.status})`);
  check(result.stderr.includes("usage:"), "no-args prints usage on stderr");
  check(result.stdout === "", "no-args writes nothing to stdout");
}

function checkInfo(cli, env, cacheDir) {
  const result = run(process.execPath, [cli, "info", "--json"], {
    env,
    timeout: MODEL_TIMEOUT_MS,
  });
  check(
    result.status === 0,
    `info --json exits 0 (got ${result.status}) ${result.stderr.slice(0, 300)}`,
  );
  const info = JSON.parse(result.stdout.trim().split("\n").pop() ?? "{}");
  check(
    info.dimension > 0 && typeof info.model === "string",
    `info reports model ${info.model} dim ${info.dimension}`,
  );
  check(
    info.cacheDir === cacheDir,
    `info uses NACHOS_EMBEDDINGS_CACHE (${info.cacheDir})`,
  );
  return info;
}

function checkEmbed(cli, env, dimension) {
  const result = run(process.execPath, [cli, "embed", "--stdin", "--json"], {
    env,
    input: "hello world\nsecond line\n",
    timeout: MODEL_TIMEOUT_MS,
  });
  check(
    result.status === 0,
    `embed exits 0 (got ${result.status}) ${result.stderr.slice(0, 300)}`,
  );
  const rows = result.stdout
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));
  check(
    rows.length === 2 && rows.every((r) => r.length === dimension),
    "embed returns one vector per line with the reported dimension",
  );
}

function checkServe(cli, env, dimension) {
  return new Promise((done) => {
    const child = spawn(process.execPath, [cli, "serve"], { env });
    const lines = [];
    let buffer = "";
    const timer = setTimeout(() => child.kill(), MODEL_TIMEOUT_MS);
    child.stdout.on("data", (chunk) => {
      buffer += chunk;
      const parts = buffer.split("\n");
      buffer = parts.pop() ?? "";
      for (const part of parts) {
        lines.push(JSON.parse(part));
        if (lines.length === 1)
          child.stdin.write('{"id":7,"texts":["a","b"]}\n');
        if (lines.length === 2) child.stdin.end();
      }
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      const [ready, reply] = lines;
      check(code === 0, `serve exits 0 on stdin EOF (got ${code})`);
      check(
        ready?.ready === true && ready.dimension === dimension,
        "serve announces ready",
      );
      check(
        reply?.id === 7 &&
          reply.vectors?.length === 2 &&
          reply.vectors[0].length === dimension,
        "serve answers a request",
      );
      done();
    });
  });
}

function checkRerank(cli, env) {
  const passages = [
    "New York City is famous for the Metropolitan Museum of Art.",
    "Berlin has a population of 3,520,031 registered inhabitants.",
    "",
  ];
  const result = run(
    process.execPath,
    [
      cli,
      "rerank",
      "--query",
      "How many people live in Berlin?",
      "--stdin",
    ].concat("--json"),
    { env, input: passages.join("\n") + "\n", timeout: MODEL_TIMEOUT_MS },
  );
  check(
    result.status === 0,
    `rerank exits 0 (got ${result.status}) ${result.stderr.slice(0, 300)}`,
  );
  const scores = JSON.parse(result.stdout.trim() || "[]");
  check(
    scores.length === 3 && scores[1] > scores[0] && scores[1] > scores[2],
    `rerank scores the relevant passage highest (${scores.map((x) => x.toFixed?.(3))})`,
  );
}

/**
 * The model files are cached by now (the checks above downloaded them), so
 * further runs must make no network request at all. Proof: each command runs
 * with scripts/block-network.mjs preloaded, which makes every socket connect
 * and DNS lookup throw and logs it. Any attempt fails the command and shows
 * up in the log. Runs both without --offline (the default cache-first path)
 * and with it.
 */
function checkNoNetwork(cli, env, workDir) {
  const log = join(workDir, "network-attempts.log");
  const blocker = join(repoRoot, "scripts", "block-network.mjs");
  const blockedEnv = { ...env, BLOCK_NETWORK_LOG: log };
  const commands = [
    [["embed", "--stdin", "--json"], "hello\nworld\n"],
    [["rerank", "--query", "q", "--stdin", "--json"], "a\nb\n"],
    [["embed", "--stdin", "--json", "--offline"], "hello\n"],
    [["rerank", "--query", "q", "--stdin", "--json", "--offline"], "a\n"],
  ];
  for (const [args, input] of commands) {
    const result = run(process.execPath, ["--import", blocker, cli, ...args], {
      env: blockedEnv,
      input,
      timeout: MODEL_TIMEOUT_MS,
    });
    const attempts = existsSync(log) ? readFileSync(log, "utf8").trim() : "";
    check(
      result.status === 0 && attempts === "",
      `${args.join(" ")}: cached run makes no network requests (exit ${result.status}${attempts ? `, attempts: ${attempts}` : ""})`,
    );
  }
  const missing = run(
    process.execPath,
    [
      cli,
      "embed",
      "--stdin",
      "--json",
      "--offline",
      "--cache-dir",
      join(workDir, "empty-cache"),
    ],
    { env: blockedEnv, input: "hi\n", timeout: MODEL_TIMEOUT_MS },
  );
  check(
    missing.status === 2 &&
      missing.stderr.includes("not in the cache") &&
      missing.stderr.includes("nachos-embeddings pull"),
    "--offline with an empty cache fails clearly and makes no request",
  );
}

/**
 * A local model directory works with no network and gives the same vectors as
 * the hub copy it was made from; models that cannot work fail with a clear
 * message (checked against the real hub).
 */
function checkLocalAndErrors(cli, env, workDir, cacheDir) {
  const local = join(workDir, "local-model");
  cpSync(join(cacheDir, "Xenova", "all-MiniLM-L6-v2"), local, {
    recursive: true,
  });
  const log = join(workDir, "local-network.log");
  const blocker = join(repoRoot, "scripts", "block-network.mjs");
  const embed = (model) =>
    run(
      process.execPath,
      [
        "--import",
        blocker,
        cli,
        "embed",
        "--stdin",
        "--json",
        "--model",
        model,
        "--offline",
      ],
      {
        env: { ...env, BLOCK_NETWORK_LOG: log },
        input: "hello world\n",
        timeout: MODEL_TIMEOUT_MS,
      },
    );
  const fromDir = embed(local);
  const fromHub = embed("Xenova/all-MiniLM-L6-v2");
  const attempts = existsSync(log) ? readFileSync(log, "utf8").trim() : "";
  check(
    fromDir.status === 0 &&
      fromDir.stdout === fromHub.stdout &&
      attempts === "",
    `local model directory matches the hub copy with no network (exit ${fromDir.status} ${fromDir.stderr.slice(0, 200)}${attempts ? `, attempts: ${attempts}` : ""})`,
  );
  const pull = (model) =>
    run(process.execPath, [cli, "pull", model, "--dtype", "q8"], {
      env,
      timeout: MODEL_TIMEOUT_MS,
    });
  const noOnnx = pull("intfloat/e5-small-v2");
  check(
    noOnnx.status === 2 && noOnnx.stderr.includes("no ONNX weights"),
    `pull of a repo without ONNX weights fails clearly (exit ${noOnnx.status}: ${noOnnx.stderr.slice(0, 160)})`,
  );
  const missing = pull("nobody-at-all/does-not-exist-xyz");
  check(
    missing.status === 2 &&
      missing.stderr.includes("not found on the Hugging Face hub"),
    `pull of an unknown model fails clearly (exit ${missing.status}: ${missing.stderr.slice(0, 160)})`,
  );
  const decoder = pull("openai-community/gpt2");
  check(
    decoder.status === 2 &&
      decoder.stderr.includes("cannot produce sentence embeddings"),
    `pull of a generative model fails clearly (exit ${decoder.status}: ${decoder.stderr.slice(0, 160)})`,
  );
}

async function main() {
  const workDir = mkdtempSync(join(tmpdir(), "nachos-smoke-"));
  try {
    const entry = packTarball(workDir);
    checkFiles(entry);
    const pkg = extract(workDir, entry.filename);
    const cli = join(pkg, "bin", "cli.js");
    const cacheDir =
      process.env.NACHOS_EMBEDDINGS_CACHE ?? join(workDir, "model-cache");
    const env = { ...process.env, NACHOS_EMBEDDINGS_CACHE: cacheDir };

    checkUsage(cli, env);
    if (withModel) {
      const info = checkInfo(cli, env, cacheDir);
      checkEmbed(cli, env, info.dimension);
      await checkServe(cli, env, info.dimension);
      checkRerank(cli, env);
      checkNoNetwork(cli, env, workDir);
      checkLocalAndErrors(cli, env, workDir, cacheDir);
    } else {
      console.log(
        "skip model checks (set SMOKE_MODEL=1 to run info/embed/serve/rerank)",
      );
    }
  } finally {
    rmSync(workDir, { recursive: true, force: true });
  }
  if (failures.length) {
    console.error(`\nsmoke-pack: ${failures.length} check(s) failed`);
    process.exit(1);
  }
  console.log("\nsmoke-pack: all checks passed");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
