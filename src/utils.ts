export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

export function chunkText(
  text: string,
  options?: {
    maxTokens?: number;
    overlapTokens?: number;
  }
): string[] {
  const maxTokens = options?.maxTokens ?? 500;
  const overlapTokens = options?.overlapTokens ?? 50;

  const sentences = text.split(/(?<=[.!?])\s+/);
  
  if (sentences.length === 0) {
    return [text];
  }

  const chunks: string[] = [];
  let currentChunk: string[] = [];
  let currentTokens = 0;

  for (const sentence of sentences) {
    const sentenceTokens = estimateTokens(sentence);

    if (currentTokens + sentenceTokens > maxTokens && currentChunk.length > 0) {
      chunks.push(currentChunk.join(' '));

      const overlapSentences = Math.max(
        1,
        Math.floor((overlapTokens / maxTokens) * currentChunk.length)
      );
      currentChunk = currentChunk.slice(-overlapSentences);
      currentTokens = currentChunk.reduce((sum, s) => sum + estimateTokens(s), 0);
    }

    currentChunk.push(sentence);
    currentTokens += sentenceTokens;
  }

  if (currentChunk.length > 0) {
    chunks.push(currentChunk.join(' '));
  }

  return chunks.length > 0 ? chunks : [text];
}

export function textSimilarity(a: string, b: string): number {
  const wordsA = new Set(a.toLowerCase().split(/\s+/));
  const wordsB = new Set(b.toLowerCase().split(/\s+/));

  const intersection = new Set([...wordsA].filter(w => wordsB.has(w)));
  const union = new Set([...wordsA, ...wordsB]);

  return union.size === 0 ? 0 : intersection.size / union.size;
}

export function normalizeText(text: string): string {
  return text.toLowerCase().trim().replace(/\s+/g, ' ');
}
