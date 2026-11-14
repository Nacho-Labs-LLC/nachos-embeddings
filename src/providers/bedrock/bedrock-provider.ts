import type { EmbeddingProvider, BaseProviderConfig } from "../types.js";
import type { BedrockModelAdapter } from "./models/types.js";
import { resolveModelAdapter } from "./models/index.js";
import { dependencyError } from "./dependency-error.js";

/** Default Bedrock embedding model. */
export const BEDROCK_DEFAULT_MODEL_ID = "amazon.titan-embed-text-v2:0";

/** Default AWS region. */
export const BEDROCK_DEFAULT_REGION = "us-east-1";

/** Minimal structural view of the parts of the AWS SDK we use. */
interface BedrockClientLike {
  send(command: unknown): Promise<{ body: Uint8Array }>;
  destroy(): void;
}
type BedrockClientCtor = new (
  config: Record<string, unknown>,
) => BedrockClientLike;
type InvokeModelCommandCtor = new (input: {
  modelId: string;
  body: string;
  contentType: string;
  accept: string;
}) => unknown;

interface BedrockSession {
  client: BedrockClientLike;
  Command: InvokeModelCommandCtor;
}

export type CredentialStrategy = "default" | "profile" | "explicit" | "role";

export interface BedrockCredentials {
  strategy?: CredentialStrategy;
  profile?: string;
  accessKeyId?: string;
  secretAccessKey?: string;
  sessionToken?: string;
  roleArn?: string;
  roleSessionName?: string;
  externalId?: string;
}

export interface BedrockRetryConfig {
  maxAttempts?: number;
  backoffMs?: number;
}

export interface BedrockProviderConfig extends BaseProviderConfig {
  region?: string;
  modelId?: string;
  credentials?: BedrockCredentials;
  endpoint?: string;
  batchSize?: number;
  maxConcurrency?: number;
  timeout?: number;
  retry?: BedrockRetryConfig;
  modelOptions?: Record<string, unknown>;
  modelAdapter?: BedrockModelAdapter;
}

interface ResolvedConfig {
  region: string;
  modelId: string;
  credentials: Required<Pick<BedrockCredentials, "strategy">> &
    Omit<BedrockCredentials, "strategy">;
  endpoint: string | undefined;
  batchSize: number;
  maxConcurrency: number;
  timeout: number;
  retry: Required<BedrockRetryConfig>;
  modelOptions: Record<string, unknown> | undefined;
  progressLogging: boolean;
}

function resolveConfig(config: BedrockProviderConfig): {
  adapter: BedrockModelAdapter;
  resolvedConfig: ResolvedConfig;
} {
  const modelId = config.modelId ?? BEDROCK_DEFAULT_MODEL_ID;
  const adapter = config.modelAdapter ?? resolveModelAdapter(modelId);

  if (config.modelOptions && adapter.validateOptions) {
    adapter.validateOptions(config.modelOptions);
  }

  return {
    adapter,
    resolvedConfig: resolveRuntimeConfig(config, modelId),
  };
}

function resolveRuntimeConfig(
  config: BedrockProviderConfig,
  modelId: string,
): ResolvedConfig {
  return {
    region: config.region ?? BEDROCK_DEFAULT_REGION,
    modelId,
    credentials: resolveCredentialsConfig(config.credentials),
    endpoint: config.endpoint,
    batchSize: config.batchSize ?? 25,
    maxConcurrency: config.maxConcurrency ?? 5,
    timeout: config.timeout ?? 30000,
    retry: resolveRetryConfig(config.retry),
    modelOptions: config.modelOptions,
    progressLogging: config.progressLogging ?? false,
  };
}

function resolveCredentialsConfig(
  credentials: BedrockCredentials | undefined,
): ResolvedConfig["credentials"] {
  return { strategy: credentials?.strategy ?? "default", ...credentials };
}

function resolveRetryConfig(
  retry: BedrockRetryConfig | undefined,
): Required<BedrockRetryConfig> {
  return {
    maxAttempts: retry?.maxAttempts ?? 3,
    backoffMs: retry?.backoffMs ?? 200,
  };
}

export class BedrockProvider implements EmbeddingProvider {
  readonly name = "bedrock";

  private session: BedrockSession | null = null;
  private adapter: BedrockModelAdapter;
  private resolvedConfig: ResolvedConfig;
  private initialized = false;
  private initPromise: Promise<void> | null = null;
  private dimension: number | null = null;

  constructor(config: BedrockProviderConfig = {}) {
    const { adapter, resolvedConfig } = resolveConfig(config);
    this.adapter = adapter;
    this.resolvedConfig = resolvedConfig;
  }

  init(): Promise<void> {
    if (this.initialized) return Promise.resolve();
    // Memoize the in-flight init; clear it afterwards so a failure can be retried.
    this.initPromise ??= this.load().finally(() => {
      this.initPromise = null;
    });
    return this.initPromise;
  }

  private async load(): Promise<void> {
    const { BedrockRuntimeClient, InvokeModelCommand } = await this.loadSdk();
    const clientConfig = await this.createClientConfig();
    const session: BedrockSession = {
      client: new BedrockRuntimeClient(clientConfig),
      Command: InvokeModelCommand,
    };

    if (this.resolvedConfig.progressLogging) {
      // eslint-disable-next-line no-console -- opt-in `progressLogging` is documented to log to the console
      console.log(
        `[BedrockProvider] Initialized with model: ${this.resolvedConfig.modelId}, region: ${this.resolvedConfig.region}`,
      );
    }

    // Probe dimension; commit state only once it succeeds.
    let dimension: number;
    try {
      dimension = (await this.embedWith(session, "dimension probe")).length;
    } catch (error) {
      session.client.destroy();
      throw error;
    }
    this.session = session;
    this.dimension = dimension;
    this.initialized = true;

    if (this.resolvedConfig.progressLogging) {
      // eslint-disable-next-line no-console -- opt-in `progressLogging` is documented to log to the console
      console.log(
        `[BedrockProvider] Detected dimension: ${String(this.dimension)}`,
      );
    }
  }

  async embed(text: string): Promise<number[]> {
    if (!this.initialized) {
      throw new Error("BedrockProvider not initialized. Call init() first.");
    }
    return this.embedSingle(text);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (!this.initialized) {
      throw new Error("BedrockProvider not initialized. Call init() first.");
    }

    const results: number[][] = [];
    const { batchSize, maxConcurrency } = this.resolvedConfig;

    for (let i = 0; i < texts.length; i += batchSize) {
      const chunk = texts.slice(i, i + batchSize);
      const chunkResults = await this.processWithConcurrency(
        chunk,
        maxConcurrency,
      );
      results.push(...chunkResults);
    }

    return results;
  }

  async getDimension(): Promise<number | null> {
    return this.dimension;
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  getModelId(): string {
    return this.resolvedConfig.modelId;
  }

  /** Destroy the underlying AWS client. The provider can be init()ed again. */
  async dispose(): Promise<void> {
    if (this.initPromise) {
      await this.initPromise.catch(() => undefined);
    }
    const session = this.session;
    this.session = null;
    this.initialized = false;
    this.dimension = null;
    session?.client.destroy();
  }

  getConfig(): Readonly<ResolvedConfig> {
    return { ...this.resolvedConfig };
  }

  private async loadSdk(): Promise<{
    BedrockRuntimeClient: BedrockClientCtor;
    InvokeModelCommand: InvokeModelCommandCtor;
  }> {
    const specifier = "@aws-sdk/client-bedrock-runtime";
    try {
      const sdk = await import("@aws-sdk/client-bedrock-runtime");
      return sdk as unknown as {
        BedrockRuntimeClient: BedrockClientCtor;
        InvokeModelCommand: InvokeModelCommandCtor;
      };
    } catch (error) {
      throw dependencyError(
        error,
        specifier,
        `${specifier} is required for BedrockProvider. ` +
          `Install it: npm install ${specifier}`,
      );
    }
  }

  private async createClientConfig(): Promise<Record<string, unknown>> {
    const config: Record<string, unknown> = {
      region: this.resolvedConfig.region,
      ...(this.resolvedConfig.endpoint && {
        endpoint: this.resolvedConfig.endpoint,
      }),
      ...(this.resolvedConfig.timeout && {
        requestTimeout: this.resolvedConfig.timeout,
      }),
    };
    const credentials = await this.resolveCredentials();
    if (credentials) {
      config["credentials"] = credentials;
    }
    return config;
  }

  private async resolveCredentials(): Promise<unknown> {
    switch (this.resolvedConfig.credentials.strategy) {
      case "profile":
        return this.profileCredentials();
      case "explicit":
        return this.explicitCredentials();
      case "role":
        return this.roleCredentials();
      default:
        return undefined;
    }
  }

  private async profileCredentials(): Promise<unknown> {
    try {
      const { fromIni } = await import("@aws-sdk/credential-provider-ini");
      const { profile } = this.resolvedConfig.credentials;
      return fromIni(profile ? { profile } : {});
    } catch (error) {
      throw dependencyError(
        error,
        "@aws-sdk/credential-provider-ini",
        "@aws-sdk/credential-provider-ini is required for profile credential strategy. " +
          "It should be available with @aws-sdk/client-bedrock-runtime.",
      );
    }
  }

  private explicitCredentials(): Record<string, string> {
    const { accessKeyId, secretAccessKey, sessionToken } =
      this.resolvedConfig.credentials;
    if (!accessKeyId || !secretAccessKey) {
      throw new Error(
        "Explicit credential strategy requires 'accessKeyId' and 'secretAccessKey'.",
      );
    }
    return {
      accessKeyId,
      secretAccessKey,
      ...(sessionToken !== undefined ? { sessionToken } : {}),
    };
  }

  private async roleCredentials(): Promise<unknown> {
    const { roleArn, roleSessionName, externalId } =
      this.resolvedConfig.credentials;
    if (!roleArn) {
      throw new Error("Role credential strategy requires 'roleArn'.");
    }
    try {
      const { fromTemporaryCredentials } =
        await import("@aws-sdk/credential-providers");
      return fromTemporaryCredentials({
        params: {
          RoleArn: roleArn,
          RoleSessionName: roleSessionName ?? "nachos-embeddings",
          ...(externalId !== undefined ? { ExternalId: externalId } : {}),
        },
      });
    } catch (error) {
      throw dependencyError(
        error,
        "@aws-sdk/credential-providers",
        "@aws-sdk/credential-providers is required for role credential strategy. " +
          "Install it: npm install @aws-sdk/credential-providers",
      );
    }
  }

  private async embedSingle(text: string): Promise<number[]> {
    if (!this.session) {
      throw new Error("BedrockProvider not initialized. Call init() first.");
    }
    return this.embedWith(this.session, text);
  }

  private async embedWith(
    session: BedrockSession,
    text: string,
  ): Promise<number[]> {
    const body = this.adapter.formatRequest(
      text,
      this.resolvedConfig.modelOptions,
    );
    const responseBody = await this.invokeModel(session, body);
    return this.adapter.parseResponse(responseBody);
  }

  private async processWithConcurrency(
    texts: string[],
    maxConcurrency: number,
  ): Promise<number[][]> {
    const results: number[][] = new Array(texts.length);
    let nextIndex = 0;

    async function runWorker(
      embedFn: (text: string) => Promise<number[]>,
    ): Promise<void> {
      while (nextIndex < texts.length) {
        const idx = nextIndex++;
        const text = texts[idx];
        if (text !== undefined) {
          results[idx] = await embedFn(text);
        }
      }
    }

    const workers: Promise<void>[] = [];
    const workerCount = Math.min(maxConcurrency, texts.length);
    const boundEmbed = this.embedSingle.bind(this);

    for (let w = 0; w < workerCount; w++) {
      workers.push(runWorker(boundEmbed));
    }

    await Promise.all(workers);

    return results as number[][];
  }

  private async invokeModel(
    session: BedrockSession,
    body: string,
  ): Promise<string> {
    return this.invokeWithRetry(async () => {
      const command = new session.Command({
        modelId: this.resolvedConfig.modelId,
        body,
        contentType: "application/json",
        accept: "application/json",
      });

      const response = await session.client.send(command);
      return new TextDecoder().decode(response.body);
    });
  }

  private async invokeWithRetry(fn: () => Promise<string>): Promise<string> {
    const { maxAttempts, backoffMs } = this.resolvedConfig.retry;
    let lastError: unknown;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        return await fn();
      } catch (error: unknown) {
        lastError = error;

        if (!this.isRetryableError(error)) {
          throw error;
        }

        if (attempt < maxAttempts - 1) {
          const delay = backoffMs * Math.pow(2, attempt);
          await this.sleep(delay);
        }
      }
    }

    throw lastError;
  }

  private isRetryableError(error: unknown): boolean {
    if (error instanceof Error) {
      const name = error.name;
      if (
        name === "ThrottlingException" ||
        name === "ServiceUnavailableException" ||
        name === "TooManyRequestsException"
      ) {
        return true;
      }

      const message = error.message;
      if (
        message.includes("ECONNRESET") ||
        message.includes("ETIMEDOUT") ||
        message.includes("ECONNREFUSED") ||
        message.includes("socket hang up")
      ) {
        return true;
      }
    }

    return false;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
