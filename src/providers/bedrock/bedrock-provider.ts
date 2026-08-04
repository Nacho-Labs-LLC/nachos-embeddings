import type { EmbeddingProvider, BaseProviderConfig } from "../types.js";
import type { BedrockModelAdapter } from "./models/types.js";
import { resolveModelAdapter } from "./models/index.js";

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
  const modelId = config.modelId ?? "amazon.titan-embed-text-v2:0";
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
    region: config.region ?? "us-east-1",
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

  private client: any = null;
  private InvokeModelCommandCtor: any = null;
  private adapter: BedrockModelAdapter;
  private resolvedConfig: ResolvedConfig;
  private initialized = false;
  private dimension: number | null = null;

  constructor(config: BedrockProviderConfig = {}) {
    const { adapter, resolvedConfig } = resolveConfig(config);
    this.adapter = adapter;
    this.resolvedConfig = resolvedConfig;
  }

  async init(): Promise<void> {
    if (this.initialized) {
      return;
    }

    const { BedrockRuntimeClient, InvokeModelCommand } = await this.loadSdk();
    this.InvokeModelCommandCtor = InvokeModelCommand;
    const clientConfig = await this.createClientConfig();

    this.client = new BedrockRuntimeClient(clientConfig);

    if (this.resolvedConfig.progressLogging) {
      console.log(
        `[BedrockProvider] Initialized with model: ${this.resolvedConfig.modelId}, region: ${this.resolvedConfig.region}`,
      );
    }

    // Probe dimension
    const probeVector = await this.embedSingle("dimension probe");
    this.dimension = probeVector.length;

    this.initialized = true;

    if (this.resolvedConfig.progressLogging) {
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

  getConfig(): Readonly<ResolvedConfig> {
    return { ...this.resolvedConfig };
  }

  private async loadSdk(): Promise<{
    BedrockRuntimeClient: any;
    InvokeModelCommand: any;
  }> {
    try {
      return await import("@aws-sdk/client-bedrock-runtime");
    } catch {
      throw new Error(
        "@aws-sdk/client-bedrock-runtime is required for BedrockProvider. " +
          "Install it: npm install @aws-sdk/client-bedrock-runtime",
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
    } catch {
      throw new Error(
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
      } as any);
    } catch {
      throw new Error(
        "@aws-sdk/credential-providers is required for role credential strategy. " +
          "Install it: npm install @aws-sdk/credential-providers",
      );
    }
  }

  private async embedSingle(text: string): Promise<number[]> {
    const body = this.adapter.formatRequest(
      text,
      this.resolvedConfig.modelOptions,
    );
    const responseBody = await this.invokeModel(body);
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

  private async invokeModel(body: string): Promise<string> {
    return this.invokeWithRetry(async () => {
      const command = new this.InvokeModelCommandCtor({
        modelId: this.resolvedConfig.modelId,
        body,
        contentType: "application/json",
        accept: "application/json",
      });

      const response = await this.client.send(command);
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
