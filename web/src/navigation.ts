export async function saveBeforeLeaving(
  dirty: boolean,
  save: () => Promise<unknown>,
  navigate: () => Promise<unknown>,
) {
  if (dirty) await save();
  await navigate();
}
export function nextFrameId(
  frames: { frame_id: string }[],
  current: string,
): string | null {
  const index = frames.findIndex((f) => f.frame_id === current);
  return index >= 0 && index + 1 < frames.length
    ? frames[index + 1].frame_id
    : null;
}
