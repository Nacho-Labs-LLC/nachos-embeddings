import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    include: ["tests/**/*.test.ts"],
    testTimeout: 30_000, // Model loading can be slow on first run
    coverage: {
      provider: "v8",
      include: ["src/**"],
      // Pure type declarations and re-export barrels have no runtime logic.
      exclude: [
        "src/**/*.d.ts",
        "src/providers/types.ts",
        "src/providers/bedrock/models/types.ts",
      ],
      reporter: ["text", "lcov"],
    },
  },
});
