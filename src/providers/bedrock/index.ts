export {
  BedrockProvider,
  type BedrockProviderConfig,
  type BedrockCredentials,
  type CredentialStrategy,
  type BedrockRetryConfig,
} from './bedrock-provider.js';

export {
  resolveModelAdapter,
  TitanV2Adapter,
  CohereEmbedAdapter,
} from './models/index.js';

export type { BedrockModelAdapter } from './models/types.js';
