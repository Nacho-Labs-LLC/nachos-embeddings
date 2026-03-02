import type { EmbeddingProvider, BaseProviderConfig } from '../types.js';
import type { BedrockModelAdapter } from './models/types.js';
import { resolveModelAdapter } from './models/index.js';

export type CredentialStrategy = 'default' | 'profile' | 'explicit' | 'role';

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
  credentials: Required<Pick<BedrockCredentials, 'strategy'>> & Omit<BedrockCredentials, 'strategy'>;
  endpoint: string | undefined;
  batchSize: number;
  maxConcurrency: number;
  timeout: number;
  retry: Required<BedrockRetryConfig>;
  modelOptions: Record<string, unknown> | undefined;
  progressLogging: boolean;
}

export class BedrockProvider implements EmbeddingProvider {
  readonly name = 'bedrock';

  private client: any = null;
  private InvokeModelCommandCtor: any = null;
  private adapter: BedrockModelAdapter;
  private resolvedConfig: ResolvedConfig;
  private initialized = false;
  private dimension: number | null = null;

  constructor(config: BedrockProviderConfig = {}) {
    const modelId = config.modelId ?? 'amazon.titan-embed-text-v2:0';

    this.adapter = config.modelAdapter ?? resolveModelAdapter(modelId);

    if (config.modelOptions && this.adapter.validateOptions) {
      this.adapter.validateOptions(config.modelOptions);
    }

    this.resolvedConfig = {
      region: config.region ?? 'us-east-1',
      modelId,
      credentials: {
        strategy: config.credentials?.strategy ?? 'default',
        ...config.credentials,
      },
      endpoint: config.endpoint,
      batchSize: config.batchSize ?? 25,
      maxConcurrency: config.maxConcurrency ?? 5,
      timeout: config.timeout ?? 30000,
      retry: {
        maxAttempts: config.retry?.maxAttempts ?? 3,
        backoffMs: config.retry?.backoffMs ?? 200,
      },
      modelOptions: config.modelOptions,
      progressLogging: config.progressLogging ?? false,
    };
  }

  async init(): Promise<void> {
    if (this.initialized) {
      return;
    }

    let BedrockRuntimeClient: any;
    let InvokeModelCommand: any;

    try {
      const sdk = await import('@aws-sdk/client-bedrock-runtime');
      BedrockRuntimeClient = sdk.BedrockRuntimeClient;
      InvokeModelCommand = sdk.InvokeModelCommand;
    } catch {
      throw new Error(
        '@aws-sdk/client-bedrock-runtime is required for BedrockProvider. ' +
          'Install it: npm install @aws-sdk/client-bedrock-runtime',
      );
    }

    this.InvokeModelCommandCtor = InvokeModelCommand;

    const clientConfig: Record<string, unknown> = {
      region: this.resolvedConfig.region,
    };

    const { strategy } = this.resolvedConfig.credentials;

    if (strategy === 'profile') {
      try {
        const { fromIni } = await import('@aws-sdk/credential-provider-ini');
        const profile = this.resolvedConfig.credentials.profile;
        clientConfig['credentials'] = fromIni(profile ? { profile } : {});
      } catch {
        throw new Error(
          '@aws-sdk/credential-provider-ini is required for profile credential strategy. ' +
            'It should be available with @aws-sdk/client-bedrock-runtime.',
        );
      }
    } else if (strategy === 'explicit') {
      const { accessKeyId, secretAccessKey, sessionToken } = this.resolvedConfig.credentials;
      if (!accessKeyId || !secretAccessKey) {
        throw new Error(
          "Explicit credential strategy requires 'accessKeyId' and 'secretAccessKey'.",
        );
      }
      clientConfig['credentials'] = {
        accessKeyId,
        secretAccessKey,
        ...(sessionToken !== undefined ? { sessionToken } : {}),
      };
    } else if (strategy === 'role') {
      const { roleArn } = this.resolvedConfig.credentials;
      if (!roleArn) {
        throw new Error("Role credential strategy requires 'roleArn'.");
      }
      try {
        const { fromTemporaryCredentials } = await import('@aws-sdk/credential-providers');
        const params: Record<string, unknown> = {
          params: {
            RoleArn: roleArn,
            RoleSessionName:
              this.resolvedConfig.credentials.roleSessionName ?? 'nachos-embeddings',
            ...(this.resolvedConfig.credentials.externalId !== undefined
              ? { ExternalId: this.resolvedConfig.credentials.externalId }
              : {}),
          },
        };
        clientConfig['credentials'] = fromTemporaryCredentials(params as any);
      } catch {
        throw new Error(
          '@aws-sdk/credential-providers is required for role credential strategy. ' +
            'Install it: npm install @aws-sdk/credential-providers',
        );
      }
    }
    // 'default' strategy: no explicit credentials — SDK auto-resolves

    if (this.resolvedConfig.endpoint) {
      clientConfig['endpoint'] = this.resolvedConfig.endpoint;
    }

    this.client = new BedrockRuntimeClient(clientConfig);

    if (this.resolvedConfig.progressLogging) {
      console.log(
        `[BedrockProvider] Initialized with model: ${this.resolvedConfig.modelId}, region: ${this.resolvedConfig.region}`,
      );
    }

    // Probe dimension
    const probeVector = await this.embedSingle('dimension probe');
    this.dimension = probeVector.length;

    this.initialized = true;

    if (this.resolvedConfig.progressLogging) {
      console.log(`[BedrockProvider] Detected dimension: ${String(this.dimension)}`);
    }
  }

  async embed(text: string): Promise<number[]> {
    if (!this.initialized) {
      throw new Error('BedrockProvider not initialized. Call init() first.');
    }
    return this.embedSingle(text);
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    if (!this.initialized) {
      throw new Error('BedrockProvider not initialized. Call init() first.');
    }

    const results: number[][] = [];
    const { batchSize, maxConcurrency } = this.resolvedConfig;

    for (let i = 0; i < texts.length; i += batchSize) {
      const chunk = texts.slice(i, i + batchSize);
      const chunkResults = await this.processWithConcurrency(chunk, maxConcurrency);
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

  private async embedSingle(text: string): Promise<number[]> {
    const body = this.adapter.formatRequest(text, this.resolvedConfig.modelOptions);
    const responseBody = await this.invokeModel(body);
    return this.adapter.parseResponse(responseBody);
  }

  private async processWithConcurrency(
    texts: string[],
    maxConcurrency: number,
  ): Promise<number[][]> {
    const results: number[][] = new Array(texts.length);
    let nextIndex = 0;

    async function runWorker(embedFn: (text: string) => Promise<number[]>): Promise<void> {
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
        contentType: 'application/json',
        accept: 'application/json',
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
        name === 'ThrottlingException' ||
        name === 'ServiceUnavailableException' ||
        name === 'TooManyRequestsException'
      ) {
        return true;
      }

      const message = error.message;
      if (
        message.includes('ECONNRESET') ||
        message.includes('ETIMEDOUT') ||
        message.includes('ECONNREFUSED') ||
        message.includes('socket hang up')
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
