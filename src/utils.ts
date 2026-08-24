export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

export function chunkText(
  text: string,
  options?: {
    maxTokens?: number;
    overlapTokens?: number;
  },
): string[] {
  const maxTokens = options?.maxTokens ?? 500;
  const overlapTokens = options?.overlapTokens ?? 50;

  const sentences = text.split(/(?<=[.!?])\s+/);

  const chunks: string[] = [];
  let currentChunk: string[] = [];
  let currentTokens = 0;

  const saveChunk = () => {
    if (currentChunk.length > 0) {
      chunks.push(currentChunk.join(" "));
    }
  };

  for (const sentence of sentences) {
    const sentenceTokens = estimateTokens(sentence);

    if (currentTokens + sentenceTokens > maxTokens && currentChunk.length > 0) {
      saveChunk();

      const overlapSentences = Math.max(
        1,
        Math.floor((overlapTokens / maxTokens) * currentChunk.length),
      );
      currentChunk = currentChunk.slice(-overlapSentences);
      currentTokens = currentChunk.reduce(
        (sum, s) => sum + estimateTokens(s),
        0,
      );
    }

    currentChunk.push(sentence);
    currentTokens += sentenceTokens;
  }

  saveChunk();

  return chunks.length > 0 ? chunks : [text];
}

export function normalizeText(text: string): string {
  return text.toLowerCase().trim().replace(/\s+/g, " ");
}
