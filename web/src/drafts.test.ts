import { test } from "node:test";
import assert from "node:assert/strict";
import { persistDraft, restoreDraft, draftKey } from "./drafts.ts";
const storage = () => {
  const m = new Map<string, string>();
  return {
    getItem: (k: string) => m.get(k) ?? null,
    setItem: (k: string, v: string) => {
      m.set(k, v);
    },
    removeItem: (k: string) => {
      m.delete(k);
    },
    values: m,
  };
};
test("undo to saved state clears current revision draft", () => {
  const s = storage();
  persistDraft(s, "f", 0, [1], []);
  persistDraft(s, "f", 0, [], []);
  assert.equal(s.getItem(draftKey("f")), null);
});
test("conflicting draft is archived before subsequent edits", () => {
  const s = storage();
  persistDraft(s, "f", 0, ["old"], []);
  const r = restoreDraft(s, "f", 1, ["server"]);
  assert.equal(r.archived, true);
  persistDraft(s, "f", 1, ["new"], ["server"]);
  assert.ok(
    [...s.values.entries()].some(
      ([k, v]) =>
        k.includes(":archive:") && JSON.parse(v).annotations[0] === "old",
    ),
  );
});
