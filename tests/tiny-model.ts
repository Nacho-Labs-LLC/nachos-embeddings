import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

/** A hand-made WordPiece tokenizer: enough for Transformers.js to load. */
export const TOKENIZER_JSON = JSON.stringify({
  version: "1.0",
  truncation: null,
  padding: null,
  added_tokens: ["[PAD]", "[UNK]", "[CLS]", "[SEP]"].map((content, id) => ({
    id,
    content,
    special: true,
    single_word: false,
    lstrip: false,
    rstrip: false,
    normalized: false,
  })),
  normalizer: { type: "BertNormalizer", lowercase: true },
  pre_tokenizer: { type: "BertPreTokenizer" },
  post_processor: null,
  decoder: null,
  model: {
    type: "WordPiece",
    unk_token: "[UNK]",
    continuing_subword_prefix: "##",
    max_input_chars_per_word: 100,
    vocab: { "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, hello: 4 },
  },
});

export const TOKENIZER_CONFIG = JSON.stringify({
  tokenizer_class: "BertTokenizer",
  model_max_length: 512,
  unk_token: "[UNK]",
  pad_token: "[PAD]",
  cls_token: "[CLS]",
  sep_token: "[SEP]",
});

export interface TinyModelOptions {
  /** config.json contents (default: a BERT encoder, hidden size 8). */
  config?: Record<string, unknown>;
  /** ONNX weights path relative to the directory (default `onnx/model_quantized.onnx`). */
  weights?: string | null;
}

/** Write a minimal model directory (the weights are a placeholder file). */
export function writeTinyModel(
  directory: string,
  { config, weights = "onnx/model_quantized.onnx" }: TinyModelOptions = {},
): void {
  const files: Record<string, string> = {
    "config.json": JSON.stringify(
      config ?? {
        model_type: "bert",
        architectures: ["BertModel"],
        hidden_size: 8,
        max_position_embeddings: 512,
      },
    ),
    "tokenizer.json": TOKENIZER_JSON,
    "tokenizer_config.json": TOKENIZER_CONFIG,
    ...(weights !== null && { [weights]: "not a real model" }),
  };
  for (const [name, content] of Object.entries(files)) {
    const path = join(directory, name);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, content);
  }
}
