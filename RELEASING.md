# Releasing @nacho-labs/nachos-embeddings

Releases are published by GitHub Actions (`.github/workflows/npm-publish.yml`)
with npm provenance via OIDC. Never run `npm publish` from a laptop: it skips
provenance and the checks.

## 1. Release PR

1. Branch from `master`, e.g. `release/x.y.z`.
2. `npm version x.y.z --no-git-tag-version` (the tag is created on `master`
   after merge, so a squash merge cannot strand it).
3. Add the `x.y.z` entry to `CHANGELOG.md`.
4. `npm run check` (lint, format, typecheck, build, tests with coverage,
   packed-tarball smoke test). Optionally `SMOKE_MODEL=1 npm run smoke:pack`
   for the model-backed CLI checks.
5. Open a PR to `master`. CI runs `npm run check` on Node 20 and 24 plus the
   real-model smoke test. Merge when green.

## 2. Tag and publish

```sh
git checkout master && git pull
git tag vX.Y.Z && git push origin vX.Y.Z
gh release create vX.Y.Z --title vX.Y.Z --notes-file <notes>
```

Publishing the GitHub release triggers `npm-publish.yml`, which runs in the
`npm-publish` environment, fails unless the tag equals `v` + the `package.json`
version, runs `npm run check`, then `npm publish --provenance --access public`.
To retry an existing tag, re-run it with `workflow_dispatch` (input
`release_tag`).

## 3. Verify

```sh
npm view @nacho-labs/nachos-embeddings version dist.tarball
```

The npmjs.com package page should show the provenance badge linking to the
workflow run.
