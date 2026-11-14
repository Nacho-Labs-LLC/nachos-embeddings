#!/usr/bin/env node
// `prepare` hook: install git hooks only in a developer git checkout.
// No-op in CI, outside a git checkout, or when simple-git-hooks is absent.
import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";

if (process.env.CI || !existsSync(".git")) process.exit(0);

const result = spawnSync("npx", ["--no-install", "simple-git-hooks"], {
  stdio: "inherit",
  shell: process.platform === "win32",
});
if (result.status !== 0) {
  console.warn("install-hooks: skipped (simple-git-hooks unavailable)");
}
process.exit(0);
