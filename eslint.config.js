import js from "@eslint/js";
import eslintConfigPrettier from "eslint-config-prettier";
import tseslint from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";
import globals from "globals";

const qualityRules = {
  complexity: ["error", 12],
  "max-depth": ["error", 4],
  "max-lines-per-function": [
    "error",
    { max: 60, skipBlankLines: true, skipComments: true },
  ],
  "max-params": ["error", 5],
  eqeqeq: ["error", "always"],
  "prefer-const": "error",
};

export default [
  {
    ignores: ["dist/**", "node_modules/**", "coverage/**"],
  },
  js.configs.recommended,
  {
    files: ["src/**/*.ts", "tests/**/*.ts", "vitest.config.ts"],
    languageOptions: {
      globals: globals.node,
      parser: tsParser,
      parserOptions: {
        sourceType: "module",
        // Type-aware rules need tests and config files, which tsconfig.json
        // excludes from the build; lint against tsconfig.eslint.json instead.
        project: "./tsconfig.eslint.json",
        tsconfigRootDir: import.meta.dirname,
      },
    },
    plugins: {
      "@typescript-eslint": tseslint,
    },
    rules: {
      ...tseslint.configs.recommended.rules,
      ...qualityRules,
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-floating-promises": "error",
      "@typescript-eslint/no-misused-promises": "error",
    },
  },
  {
    files: ["src/**/*.ts"],
    rules: { "no-console": "error" },
  },
  {
    // The CLI is plain JS and is not part of any tsconfig, so it gets the
    // non-type-aware rules only. It writes via process.stdout/stderr, but
    // console is allowed for diagnostics.
    files: ["bin/**/*.js", "scripts/**/*.mjs"],
    languageOptions: {
      globals: globals.node,
      parserOptions: { sourceType: "module" },
    },
    rules: qualityRules,
  },
  {
    files: ["tests/**/*.ts"],
    rules: {
      "max-lines-per-function": "off",
    },
  },
  eslintConfigPrettier,
];
