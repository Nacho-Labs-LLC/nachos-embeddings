/** Environment variable that forces offline mode (`1`, `true`, `yes` or `on`). */
export const OFFLINE_ENV_VAR = "NACHOS_EMBEDDINGS_OFFLINE";

/** True when `NACHOS_EMBEDDINGS_OFFLINE` asks for offline mode. */
export function isOfflineEnv(
  env: Record<string, string | undefined> = process.env,
): boolean {
  return /^(1|true|yes|on)$/i.test((env[OFFLINE_ENV_VAR] ?? "").trim());
}
