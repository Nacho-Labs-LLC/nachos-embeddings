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
import { mkdtempSync, rmSync, symlinkSync, mkdirSync } from "node:fs";
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
