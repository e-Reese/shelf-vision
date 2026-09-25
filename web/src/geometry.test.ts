import { test } from "node:test";
import assert from "node:assert/strict";
import { drawBox, moveBox, pointInBox, fitCamera } from "./geometry.ts";
test("reverse drawing is normalized and clipped to the review region", () => {
  assert.deepEqual(
    drawBox([80, 70], [0, 0], [10, 20, 100, 100]),
    [10, 20, 80, 70],
  );
});
test("moving against a boundary preserves size", () => {
  assert.deepEqual(
    moveBox([20, 30, 60, 70], [-100, 80], [10, 20, 100, 100]),
    [10, 60, 50, 100],
  );
});
test("review region boundary is inclusive", () => {
  assert.equal(pointInBox([10, 20], [10, 20, 100, 100]), true);
  assert.equal(pointInBox([9, 20], [10, 20, 100, 100]), false);
});
test("fit accommodates tall review regions without changing image geometry", () => {
  const c = fitCamera([0, 0, 100, 400], 800, 400);
  assert.equal(c.x, 50);
  assert.equal(c.y, 200);
  assert.ok(c.width >= 800);
});
