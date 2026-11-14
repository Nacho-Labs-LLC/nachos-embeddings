export function isModuleNotFound(error: unknown, specifier: string): boolean {
  if (!(error instanceof Error)) return false;
  const code = (error as Error & { code?: unknown }).code;
  const notFoundCode =
    code === "ERR_MODULE_NOT_FOUND" || code === "MODULE_NOT_FOUND";
  return notFoundCode && error.message.includes(specifier);
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/**
 * Wrap a failed optional-dependency import: a genuinely missing package gets
 * the install hint; any other failure keeps its own message. The original
 * error is always attached as `cause`.
 */
export function dependencyError(
  error: unknown,
  specifier: string,
  missingHint: string,
): Error {
  if (isModuleNotFound(error, specifier)) {
    return new Error(missingHint, { cause: error });
  }
  return new Error(`Failed to load ${specifier}: ${describeError(error)}`, {
    cause: error,
  });
}
