import { test } from "node:test";
import assert from "node:assert/strict";
import { saveBeforeLeaving, nextFrameId } from "./navigation.ts";
test("navigation waits for dirty edits to be saved", async () => {
  const events: string[] = [];
  let release!: () => void;
  const pending = new Promise<void>((r) => (release = r));
  const task = saveBeforeLeaving(
    true,
    async () => {
      events.push("saving");
      await pending;
      events.push("saved");
    },
    async () => {
      events.push("navigate");
    },
  );
  assert.deepEqual(events, ["saving"]);
  release();
  await task;
  assert.deepEqual(events, ["saving", "saved", "navigate"]);
});
test("failed save blocks navigation", async () => {
  let navigated = false;
  await assert.rejects(
    saveBeforeLeaving(
      true,
      async () => {
        throw Error("Conflict");
      },
      async () => {
        navigated = true;
      },
    ),
    /Conflict/,
  );
  assert.equal(navigated, false);
});
test("clean navigation does not reopen a confirmed frame", async () => {
  let saved = false;
  await saveBeforeLeaving(
    false,
    async () => {
      saved = true;
    },
    async () => {},
  );
  assert.equal(saved, false);
});
test("progression follows queue order and stops at last frame", () => {
  const f = [{ frame_id: "a" }, { frame_id: "b" }];
  assert.equal(nextFrameId(f, "a"), "b");
  assert.equal(nextFrameId(f, "b"), null);
  assert.equal(nextFrameId(f, "missing"), null);
});
