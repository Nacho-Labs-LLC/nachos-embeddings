/**
 * Safe on-disk persistence helpers: async fs only, atomic writes, and strict
 * parsing so a corrupt store is reported rather than silently overwritten.
 *
 * The on-disk format is a bare JSON array of
 * `{ id, text, vector, metadata? }` (unchanged from earlier releases).
 */

import { mkdir, open, readFile, rename, unlink } from "node:fs/promises";
import { dirname } from "node:path";

export interface StoredDocument<T = unknown> {
  id: string;
  text: string;
  vector: number[];
  metadata?: T | undefined;
}

/** Thrown when a store file exists but cannot be parsed or is malformed. */
export class StoreCorruptError extends Error {
  readonly path: string;

  constructor(path: string, reason: string, cause?: unknown) {
    super(`Semantic store "${path}" is corrupt: ${reason}`, { cause });
    this.name = "StoreCorruptError";
    this.path = path;
  }
}

function isStoredDocument(item: unknown): item is StoredDocument {
  if (typeof item !== "object" || item === null) return false;
  const rec = item as Record<string, unknown>;
  return (
    typeof rec["id"] === "string" &&
    typeof rec["text"] === "string" &&
    Array.isArray(rec["vector"]) &&
    rec["vector"].every((n) => typeof n === "number")
  );
}

/**
 * Reads a store file. Returns undefined when the file does not exist.
 * Throws StoreCorruptError for unparseable or malformed content and rethrows
 * any other I/O error.
 */
export async function readStoreFile<T>(
  path: string,
): Promise<StoredDocument<T>[] | undefined> {
  let raw: string;
  try {
    raw = await readFile(path, "utf-8");
  } catch (err) {
    if ((err as { code?: string }).code === "ENOENT") return undefined;
    throw err;
  }

  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch (err) {
    throw new StoreCorruptError(path, "invalid JSON", err);
  }
  if (!Array.isArray(data)) {
    throw new StoreCorruptError(path, "expected a JSON array of documents");
  }
  const bad = data.findIndex((item) => !isStoredDocument(item));
  if (bad !== -1) {
    throw new StoreCorruptError(path, `entry ${bad} is malformed`);
  }
  return data as StoredDocument<T>[];
}

let tmpCounter = 0;

/**
 * Writes JSON atomically: temp file in the same directory, fsync, then
 * rename over the target. A failure leaves the previous file untouched.
 */
export async function writeStoreFileAtomic(
  path: string,
  data: unknown,
): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const tmp = `${path}.${process.pid}.${Date.now()}.${tmpCounter++}.tmp`;
  try {
    const handle = await open(tmp, "w");
    try {
      await handle.writeFile(JSON.stringify(data, null, 2), "utf-8");
      await handle.sync();
    } finally {
      await handle.close();
    }
    await rename(tmp, path);
  } catch (err) {
    await unlink(tmp).catch(() => undefined);
    throw err;
  }
}

/** Renames a corrupt store aside and returns the new path. */
export async function quarantineStoreFile(path: string): Promise<string> {
  const target = `${path}.corrupt-${Date.now()}`;
  await rename(path, target);
  return target;
}
