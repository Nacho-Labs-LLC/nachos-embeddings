import { describe, it, expect, vi } from "vitest";
import { EnhancedSemanticSearch } from "../src/semantic-search-enhanced.js";
import * as fsPromises from "node:fs/promises";

vi.mock("node:fs/promises", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs/promises")>();
  return {
    ...actual,
    writeFile: vi.fn(),
  };
});

describe("EnhancedSemanticSearch", () => {
  it("logs an error when saving fails", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    // Make writeFile throw an error
    vi.mocked(fsPromises.writeFile).mockRejectedValueOnce(
      new Error("Disk full"),
    );

    const search = new EnhancedSemanticSearch({
      autoSave: true,
      storePath: "dummy.json",
    });

    await search.forceSave();

    expect(errorSpy).toHaveBeenCalledWith(
      "[EnhancedSemanticSearch] Save failed:",
      expect.any(Error),
    );
    expect(errorSpy.mock.calls[0][1].message).toBe("Disk full");

    errorSpy.mockRestore();
    vi.restoreAllMocks();
  });
});
