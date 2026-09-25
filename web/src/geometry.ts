import type { Box } from "./types.ts";
export type Point = [number, number];
export type Camera = { x: number; y: number; width: number };
const clamp = (n: number, min: number, max: number) =>
  Math.max(min, Math.min(max, n));
export function pointInBox([x, y]: Point, b: Box) {
  return x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3];
}
export function drawBox(a: Point, b: Point, roi: Box): Box {
  return [
    clamp(Math.min(a[0], b[0]), roi[0], roi[2]),
    clamp(Math.min(a[1], b[1]), roi[1], roi[3]),
    clamp(Math.max(a[0], b[0]), roi[0], roi[2]),
    clamp(Math.max(a[1], b[1]), roi[1], roi[3]),
  ];
}
export function moveBox(box: Box, [dx, dy]: Point, roi: Box): Box {
  dx = clamp(dx, roi[0] - box[0], roi[2] - box[2]);
  dy = clamp(dy, roi[1] - box[1], roi[3] - box[3]);
  return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy];
}
export function fitCamera(box: Box, width: number, height: number): Camera {
  return {
    x: (box[0] + box[2]) / 2,
    y: (box[1] + box[3]) / 2,
    width:
      Math.max(box[2] - box[0], ((box[3] - box[1]) * width) / height) * 1.12,
  };
}
