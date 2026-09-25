type DraftStorage = Pick<Storage, "getItem" | "setItem" | "removeItem">;
export const draftKey = (id: string) => "shelf-vision-draft:" + id;
export function persistDraft<T>(
  storage: DraftStorage,
  id: string,
  revision: number,
  annotations: T[],
  baseline: T[],
) {
  const key = draftKey(id);
  if (JSON.stringify(annotations) !== JSON.stringify(baseline))
    storage.setItem(key, JSON.stringify({ revision, annotations }));
  else {
    const raw = storage.getItem(key);
    if (raw && JSON.parse(raw).revision === revision) storage.removeItem(key);
  }
}
export function restoreDraft<T>(
  storage: DraftStorage,
  id: string,
  revision: number,
  baseline: T[],
): { annotations: T[]; recovered: boolean; archived: boolean } {
  const key = draftKey(id),
    raw = storage.getItem(key);
  if (raw) {
    const draft = JSON.parse(raw);
    if (draft.revision === revision && Array.isArray(draft.annotations))
      return {
        annotations: draft.annotations,
        recovered:
          JSON.stringify(draft.annotations) !== JSON.stringify(baseline),
        archived: false,
      };
    // Archive successfully before removing or allowing a new revision to replace it.
    storage.setItem(key + ":archive:" + crypto.randomUUID(), raw);
    storage.removeItem(key);
    return { annotations: baseline, recovered: false, archived: true };
  }
  return { annotations: baseline, recovered: false, archived: false };
}
