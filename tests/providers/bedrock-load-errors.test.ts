import { dependencyError } from "../../src/providers/bedrock/dependency-error.js";
import { describe, it, expect, vi, beforeEach } from "vitest";

type Factory = () => unknown;
const factories = vi.hoisted(() => ({
  sdk: undefined as unknown as { current: Factory },
  ini: undefined as unknown as { current: Factory },
  providers: undefined as unknown as { current: Factory },
}));
factories.sdk = { current: () => ({}) };
factories.ini = { current: () => ({}) };
factories.providers = { current: () => ({}) };

vi.mock("@aws-sdk/client-bedrock-runtime", () => factories.sdk.current());
vi.mock("@aws-sdk/credential-provider-ini", () => factories.ini.current());
vi.mock("@aws-sdk/credential-providers", () => factories.providers.current());

function notFound(specifier: string): Error {
  return Object.assign(new Error(`Cannot find package '${specifier}'`), {
    code: "ERR_MODULE_NOT_FOUND",
  });
}

async function freshProvider(
  config: ConstructorParameters<
    typeof import("../../src/providers/bedrock/bedrock-provider.js").BedrockProvider
  >[0],
) {
  const { BedrockProvider } =
    await import("../../src/providers/bedrock/bedrock-provider.js");
  return new BedrockProvider(config);
}

describe("BedrockProvider dependency loading errors", () => {
  beforeEach(() => {
    vi.resetModules();
  });

  it("does not claim the package is missing for other SDK load failures", async () => {
    const cause = new SyntaxError("Unexpected token in sdk");
    vi.doMock("@aws-sdk/client-bedrock-runtime", () => {
      throw cause;
    });
    const error = await (await freshProvider({})).init().catch((e) => e);

    expect(error.message).toContain("Failed to load");
    expect(error.message).not.toContain("npm install");
    expect(error.cause).toBeInstanceOf(Error);
  });

  it("preserves the cause for profile credential load failures", async () => {
    vi.doMock("@aws-sdk/client-bedrock-runtime", () => ({
      BedrockRuntimeClient: vi.fn(),
      InvokeModelCommand: vi.fn(),
    }));
    const broken = new Error("broken ini");
    vi.doMock("@aws-sdk/client-bedrock-runtime", () => ({
      BedrockRuntimeClient: vi.fn(),
      InvokeModelCommand: vi.fn(),
    }));
    vi.doMock("@aws-sdk/credential-provider-ini", () => {
      throw broken;
    });
    const other = await (
      await freshProvider({ credentials: { strategy: "profile" } })
    )
      .init()
      .catch((e) => e);
    expect(other.message).toContain("Failed to load");
    expect(other.message).not.toContain("is required");
    expect(other.cause).toBeInstanceOf(Error);
  });

  it("preserves the cause for role credential load failures", async () => {
    vi.doMock("@aws-sdk/client-bedrock-runtime", () => ({
      BedrockRuntimeClient: vi.fn(),
      InvokeModelCommand: vi.fn(),
    }));
    vi.doMock("@aws-sdk/credential-providers", () => {
      throw new Error("sts exploded");
    });
    const error = await (
      await freshProvider({
        credentials: { strategy: "role", roleArn: "arn:aws:iam::1:role/r" },
      })
    )
      .init()
      .catch((e) => e);
    expect(error.message).toContain("Failed to load");
    expect(error.message).not.toContain("is required");
    expect(error.cause).toBeInstanceOf(Error);
  });
});

describe("dependencyError", () => {
  it("gives the install hint for a genuinely missing package", () => {
    const cause = notFound("@aws-sdk/x");
    const error = dependencyError(cause, "@aws-sdk/x", "install it");
    expect(error.message).toBe("install it");
    expect(error.cause).toBe(cause);
  });

  it("accepts CommonJS MODULE_NOT_FOUND", () => {
    const cause = Object.assign(new Error("Cannot find module '@aws-sdk/x'"), {
      code: "MODULE_NOT_FOUND",
    });
    expect(dependencyError(cause, "@aws-sdk/x", "install it").message).toBe(
      "install it",
    );
  });

  it("keeps the real message for other failures and for other missing modules", () => {
    for (const cause of [
      new SyntaxError("bad syntax"),
      notFound("transitive-dep"),
      "string failure",
    ]) {
      const error = dependencyError(cause, "@aws-sdk/x", "install it");
      expect(error.message).toContain("Failed to load @aws-sdk/x");
      expect(error.message).not.toBe("install it");
      expect(error.cause).toBe(cause);
    }
    expect(
      dependencyError("string failure", "@aws-sdk/x", "h").message,
    ).toContain("string failure");
  });
});
