import { useEffect, useState } from "react";
import {
  ArrowRight,
  Check,
  CheckCheck,
  Download,
  FolderPlus,
  LoaderCircle,
  RotateCcw,
  RotateCw,
  Save,
  ScanLine,
  Search,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import { nextFrameId, saveBeforeLeaving } from "./navigation";
import { Canvas } from "./Canvas";
import { draftKey, persistDraft, restoreDraft } from "./drafts";
import type {
  Area,
  Document,
  Frame,
  Job,
  Product,
  Proposal,
  ModelOption,
} from "./types";
import { drawBox, pointInBox } from "./geometry";
async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch("/api" + path, init);
  if (!r.ok) {
    const e = await r.json().catch(() => ({ detail: r.statusText }));
    throw Error(
      typeof e.detail === "string" ? e.detail : JSON.stringify(e.detail),
    );
  }
  return r.json();
}
const json = (body: unknown) => ({
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
const equal = (a: unknown, b: unknown) =>
  JSON.stringify(a) === JSON.stringify(b);
export default function App() {
  const [frames, setFrames] = useState<Frame[]>([]),
    [catalog, setCatalog] = useState<Product[]>([]),
    [doc, setDoc] = useState<Document | null>(null);
  const [areas, setAreas] = useState<Area[]>([]),
    [selected, setSelected] = useState<string | null>(null),
    [past, setPast] = useState<Area[][]>([]),
    [future, setFuture] = useState<Area[][]>([]);
  const [tool, setTool] = useState<"select" | "draw" | "pan">("select"),
    [showProposals, setShowProposals] = useState(false),
    [query, setQuery] = useState(""),
    [filter, setFilter] = useState("all");
  const [message, setMessage] = useState(""),
    [busy, setBusy] = useState(false),
    [job, setJob] = useState<Job | null>(null),
    [importing, setImporting] = useState(false),
    [tab, setTab] = useState<"catalog" | "proposals">("catalog");
  const dirty = !!doc && !equal(areas, doc.annotations),
    area = areas.find((a) => a.region_id === selected),
    working = job?.status === "queued" || job?.status === "running";
  const refresh = async () => {
    const f = await api<Frame[]>("/frames");
    setFrames(f);
    return f;
  };
  const [hasRecovery, setHasRecovery] = useState(false);
  const [modelOptions, setModelOptions] = useState<ModelOption[]>([
    {
      id: "baseline",
      label: "Original baseline + SKU matches",
      available: true,
    },
  ]);
  const [modelSource, setModelSource] = useState<"baseline" | "trained">(
    "baseline",
  );
  useEffect(() => {
    api<ModelOption[]>("/models")
      .then(setModelOptions)
      .catch((e) => setMessage(String(e)));
  }, []);
  function backupDrafts() {
    try {
      const records = Object.fromEntries(
        Object.keys(localStorage)
          .filter((k) => k.startsWith("shelf-vision-draft:"))
          .map((k) => [k, JSON.parse(localStorage.getItem(k)!)]),
      );
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(records, null, 2)], {
          type: "application/json",
        }),
      );
      const a = document.createElement("a");
      a.href = url;
      a.download = "shelf-vision-drafts.json";
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      setMessage(String(e));
    }
  }
  function install(d: Document) {
    let next = d.annotations;
    let note = "";
    try {
      const draft = restoreDraft(
        localStorage,
        d.meta.frame_id,
        d.revision,
        d.annotations,
      );
      next = draft.annotations;
      if (draft.recovered) note = "Recovered your unsaved local draft.";
      if (draft.archived)
        note =
          "A conflicting draft was archived. Download Draft backup to recover those corrections; the saved version is open.";
      setHasRecovery(
        Object.keys(localStorage).some(
          (k) => k.startsWith("shelf-vision-draft:") && k.includes(":archive:"),
        ),
      );
    } catch {
      note =
        "Draft storage unavailable. Download a backup and save before leaving.";
      setHasRecovery(true);
    }
    setDoc(d);
    setAreas(next);
    setSelected(null);
    setPast([]);
    setFuture([]);
    setMessage(note);
    setTool("select");
  }

  async function persistCurrent(review = false) {
    if (!doc) return null;
    const d = await api<Document>(
      "/frames/" + doc.meta.frame_id + (review ? "/review" : ""),
      {
        method: review ? "POST" : "PUT",
        ...json({ revision: doc.revision, annotations: areas }),
      },
    );
    setDoc(d);
    setAreas(d.annotations);
    setPast([]);
    setFuture([]);
    try {
      localStorage.removeItem(draftKey(d.meta.frame_id));
    } catch {}
    setFrames((previous) =>
      previous.map((f) =>
        f.frame_id === d.meta.frame_id
          ? {
              ...f,
              status: d.status,
              revision: d.revision,
              area_count: d.annotations.length,
            }
          : f,
      ),
    );
    return d;
  }
  async function open(id: string) {
    if (busy || id === doc?.meta.frame_id) return;
    setBusy(true);
    try {
      await saveBeforeLeaving(
        dirty,
        () => persistCurrent(),
        async () => {
          install(await api<Document>("/frames/" + id));
        },
      );
    } catch (e) {
      setMessage(
        String(e) + " Stayed on this frame. Your edits are still here.",
      );
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    Promise.all([api<Product[]>("/catalog"), api<Frame[]>("/frames")])
      .then(async ([c, f]) => {
        setCatalog(c);
        setFrames(f);
        if (f.length) install(await api<Document>("/frames/" + f[0].frame_id));
      })
      .catch((e) => setMessage(String(e)));
  }, []);
  useEffect(() => {
    if (!doc) return;
    try {
      persistDraft(
        localStorage,
        doc.meta.frame_id,
        doc.revision,
        areas,
        doc.annotations,
      );
    } catch {
      setMessage("Draft storage unavailable. Use Save to keep your changes.");
    }
  }, [areas, doc, dirty]);
  useEffect(() => {
    const before = (e: BeforeUnloadEvent) => {
      if (dirty) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", before);
    return () => window.removeEventListener("beforeunload", before);
  }, [dirty]);
  function change(next: Area[]) {
    if (busy) return;
    setPast((p) => [...p.slice(-99), areas]);
    setFuture([]);
    setAreas(next);
  }
  function undo() {
    if (!past.length || busy) return;
    setFuture((f) => [areas, ...f]);
    setAreas(past[past.length - 1]);
    setPast((p) => p.slice(0, -1));
  }
  function redo() {
    if (!future.length || busy) return;
    setPast((p) => [...p, areas]);
    setAreas(future[0]);
    setFuture((f) => f.slice(1));
  }
  function patch(update: Partial<Area>) {
    if (area)
      change(
        areas.map((a) =>
          a.region_id === area.region_id ? { ...a, ...update } : a,
        ),
      );
  }
  function remove() {
    if (area) {
      change(areas.filter((a) => a.region_id !== area.region_id));
      setSelected(null);
    }
  }
  async function save(review = false) {
    if (!doc || busy) return;
    setBusy(true);
    try {
      // A clean manual save must not reopen an already confirmed frame.
      const d = review || dirty ? await persistCurrent(review) : doc;
      if (!d) return;
      if (review) {
        const next = nextFrameId(frames, d.meta.frame_id);
        if (next) {
          install(await api<Document>("/frames/" + next));
          setMessage("Frame confirmed and saved. Continue with this frame.");
        } else {
          const remaining = frames.filter(
            (f) => f.frame_id !== d.meta.frame_id && f.status !== "reviewed",
          ).length;
          setMessage(
            remaining
              ? `Frame confirmed and saved. End of queue; ${remaining} earlier frame(s) still need confirmation.`
              : "All frames confirmed and saved. Your dataset is ready to export.",
          );
        }
      } else
        setMessage(
          "Changes saved. Confirm frame when you have checked every review region.",
        );
    } catch (e) {
      setMessage(
        String(e) + " Your edits are still here. No frame was skipped.",
      );
    } finally {
      setBusy(false);
    }
  }
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (
        (e.target as HTMLElement).closest(
          "input,textarea,select,[contenteditable]",
        )
      )
        return;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
        e.preventDefault();
        void save();
      } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        e.shiftKey ? redo() : undo();
      } else if (!e.metaKey && !e.ctrlKey) {
        if (e.key === "Delete" || e.key === "Backspace") {
          e.preventDefault();
          remove();
        }
        if (e.key.toLowerCase() === "b") setTool("draw");
        if (e.key.toLowerCase() === "v") setTool("select");
        if (e.key.toLowerCase() === "h") setTool("pan");
        if (e.key === "Escape") setSelected(null);
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  });
  async function predict() {
    if (!doc || working) return;
    try {
      setJob(
        await api<Job>(
          "/frames/" + doc.meta.frame_id + "/predict?model=" + modelSource,
          {
            method: "POST",
          },
        ),
      );
      setMessage("Running locally. You can keep editing.");
    } catch (e) {
      setMessage(String(e));
    }
  }
  useEffect(() => {
    if (!job || !working) return;
    let stopped = false;
    const timer = setInterval(async () => {
      try {
        const result = await api<Job>("/jobs/" + job.id);
        if (stopped) return;
        if (result.status === "completed") {
          const d = await api<Document>("/frames/" + result.frame_id);
          if (stopped) return;
          setDoc((old) =>
            old?.meta.frame_id === result.frame_id
              ? {
                  ...old,
                  proposals: d.proposals,
                  prediction_provenance: d.prediction_provenance,
                }
              : old,
          );
          setMessage("New proposals ready. Corrections preserved.");
          setShowProposals(true);
          setTab("proposals");
        } else if (result.status === "failed")
          setMessage("Model run failed: " + result.error);
        setJob(result);
      } catch (e) {
        if (!stopped) {
          setMessage(String(e));
          setJob(null);
        }
      }
    }, 1500);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, [job?.id, working]);
  function addProposal(p: Proposal) {
    if (!doc) return;
    const b = p.bbox_xyxy;
    const roi = p.review_roi_id
      ? doc.meta.rois.find((r) => r.roi_id === p.review_roi_id)
      : doc.meta.rois.find((r) =>
          pointInBox([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], r.bbox_xyxy),
        );
    if (!roi) {
      setMessage("Proposal is outside the review regions.");
      return;
    }
    const box = drawBox([b[0], b[1]], [b[2], b[3]], roi.bbox_xyxy);
    if (box[2] - box[0] < 4 || box[3] - box[1] < 4) return;
    const id = crypto.randomUUID();
    change([
      ...areas,
      {
        region_id: id,
        frame_id: doc.meta.frame_id,
        review_roi_id: roi.roi_id,
        bbox_xyxy: box,
        class_name: "display_area",
        occupancy: "unclear",
        identity_state: "unresolved",
        observed_sku_id: null,
        intended_sku_id: null,
        truncated: false,
      },
    ]);
    setSelected(id);
    setTab("catalog");
    setMessage(
      "Proposal added as unresolved. Check bounds, occupancy, and identity.",
    );
  }
  async function exportZip() {
    try {
      const r = await fetch("/api/export");
      if (!r.ok) throw Error((await r.json()).detail);
      const url = URL.createObjectURL(await r.blob());
      const a = document.createElement("a");
      a.href = url;
      a.download = "shelf-vision-reviewed.zip";
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setMessage(
        "Downloaded reviewed, non-test regions and correction history.",
      );
    } catch (e) {
      setMessage(String(e));
    }
  }
  async function importFile(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (busy) return;
    const upload = new FormData(e.currentTarget);
    setBusy(true);
    try {
      if (dirty) await persistCurrent();
      const d = await api<Document>("/import", {
        method: "POST",
        body: upload,
      });
      await refresh();
      install(d);
      setImporting(false);
      setMessage(
        "Image imported. Review scope is the entire image. Draw areas or run the models.",
      );
    } catch (e) {
      setMessage(String(e));
    } finally {
      setBusy(false);
    }
  }
  const reviewed = frames.filter((f) => f.status === "reviewed").length,
    products = catalog.filter((p) =>
      (p.name + " " + p.sku_id + " " + (p.size || ""))
        .toLowerCase()
        .includes(query.toLowerCase()),
    );
  return (
    <div className="app">
      <header>
        <div className="brand">
          <span>
            <ScanLine size={24} />
          </span>
          <div>
            Shelf Vision<small>REVIEW WORKSPACE</small>
          </div>
        </div>
        <span className="local-pill">● Local on your Mac</span>
        <div className="header-actions">
          {hasRecovery && <button onClick={backupDrafts}>Draft backup</button>}
          <button onClick={() => setImporting(true)} disabled={busy}>
            <FolderPlus size={16} />
            Import image
          </button>
          <button onClick={exportZip}>
            <Download size={16} />
            Export reviewed
          </button>
        </div>
      </header>
      <div className="workspace">
        <aside className="queue">
          <div className="queue-title">
            <h2>Review queue</h2>
            <span>{frames.length}</span>
          </div>
          <div className="progress">
            <div
              style={{
                width:
                  (frames.length ? (100 * reviewed) / frames.length : 0) + "%",
              }}
            />
          </div>
          <p className="hint">
            {reviewed} of {frames.length} frames confirmed
          </p>
          <select
            aria-label="Filter frames"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          >
            <option value="all">All frames</option>
            <option value="pending">Needs confirmation</option>
            <option value="reviewed">Confirmed</option>
          </select>
          <div className="frame-list">
            {frames
              .filter(
                (f) =>
                  filter === "all" ||
                  (filter === "reviewed"
                    ? f.status === "reviewed"
                    : f.status !== "reviewed"),
              )
              .map((f, i) => (
                <button
                  key={f.frame_id}
                  disabled={busy}
                  className={
                    "frame-card " +
                    (doc?.meta.frame_id === f.frame_id ? "active" : "")
                  }
                  onClick={() => open(f.frame_id)}
                >
                  <div className="frame-thumb">
                    <img
                      src={"/api/frames/" + f.frame_id + "/image"}
                      loading="lazy"
                      alt=""
                    />
                    <span>{String(i + 1).padStart(2, "0")}</span>
                    {f.status === "reviewed" && <CheckCheck size={16} />}
                  </div>
                  <strong>{f.name}</strong>
                  <small>
                    {f.area_count} areas · {f.split}
                  </small>
                  <small>
                    {f.status === "reviewed"
                      ? "✓ Confirmed"
                      : f.revision
                        ? "Saved · not confirmed"
                        : "Needs confirmation"}
                  </small>
                </button>
              ))}
          </div>
          <div className="queue-foot">
            Two documented shelves.
            <br />
            Other products may be unknown.
          </div>
        </aside>
        <main>
          <div className="frame-heading">
            <div>
              <div className="eyebrow">
                {doc?.meta.capture_group || "LOCAL DATASET"} /{" "}
                {doc?.meta.split || "REVIEW"}
              </div>
              <h1>{doc?.meta.name || "Opening workspace…"}</h1>
              <p>
                {doc
                  ? `${areas.length} display areas · ${doc.meta.rois.length} review regions`
                  : "Loading frames and catalog"}
              </p>
            </div>
            <span className={"save-state " + (dirty ? "dirty" : "")}>
              {busy ? (
                <LoaderCircle size={14} className="spin" />
              ) : (
                <Check size={14} />
              )}{" "}
              {busy
                ? "Working…"
                : dirty
                  ? "Changes save on navigation"
                  : doc?.status === "reviewed"
                    ? "Frame confirmed"
                    : "Saved locally"}
            </span>
          </div>
          {message && (
            <div className="notice" role="status">
              <span>{message}</span>
              <button
                aria-label="Dismiss message"
                onClick={() => setMessage("")}
              >
                <X size={15} />
              </button>
            </div>
          )}
          {doc ? (
            <div className={"canvas-wrap " + (busy ? "busy" : "")}>
              <Canvas
                doc={doc}
                areas={areas}
                selected={selected}
                onSelect={setSelected}
                onChange={change}
                catalog={catalog}
                tool={tool}
                setTool={setTool}
                notice={setMessage}
                showProposals={showProposals}
                setShowProposals={setShowProposals}
              />
            </div>
          ) : (
            <div className="empty-state">
              <LoaderCircle className="spin" />
              Loading workspace
            </div>
          )}
          <div className="save-bar">
            <div className="history">
              <button
                onClick={undo}
                disabled={!past.length || busy}
                aria-label="Undo"
                title="Undo (⌘Z)"
              >
                <RotateCcw size={17} />
              </button>
              <button
                onClick={redo}
                disabled={!future.length || busy}
                aria-label="Redo"
                title="Redo (⇧⌘Z)"
              >
                <RotateCw size={17} />
              </button>
              <small>Revision {doc?.revision || 0}</small>
            </div>
            <div className="save-actions">
              <button onClick={() => save()} disabled={!doc || busy || !dirty}>
                <Save size={15} />
                Save
              </button>
              <button
                onClick={() => {
                  const id = doc && nextFrameId(frames, doc.meta.frame_id);
                  if (id) void open(id);
                }}
                disabled={
                  !doc || busy || !nextFrameId(frames, doc.meta.frame_id)
                }
              >
                Next frame
                <ArrowRight size={15} />
              </button>
              <button
                className="primary"
                onClick={() => save(true)}
                disabled={!doc || busy}
              >
                <CheckCheck size={16} />
                Confirm frame
              </button>
            </div>
          </div>
          <p className="scope-note">
            Changes save automatically when you switch frames. Confirm frame
            saves, confirms all cyan review regions, and advances.
          </p>
        </main>
        <aside className="inspector">
          <section className="selection">
            <div className="section-title">
              <h2>
                {area
                  ? "Display area " + (areas.indexOf(area) + 1)
                  : "Area details"}
              </h2>
              {area && (
                <button
                  onClick={remove}
                  disabled={busy}
                  aria-label="Delete selected area"
                >
                  <Trash2 size={15} />
                </button>
              )}
            </div>
            {area ? (
              <>
                <label>
                  Occupancy
                  <select
                    disabled={busy}
                    value={area.occupancy}
                    onChange={(e) => {
                      const occupancy = e.target.value as Area["occupancy"];
                      patch({
                        occupancy,
                        identity_state:
                          occupancy === "empty" || occupancy === "mixed"
                            ? "not_applicable"
                            : "unresolved",
                        observed_sku_id: null,
                      });
                    }}
                  >
                    <option value="occupied">Occupied</option>
                    <option value="empty">Empty</option>
                    <option value="mixed">Mixed products</option>
                    <option value="unclear">Unclear</option>
                  </select>
                </label>
                <label>
                  Identity
                  <select
                    disabled={busy || area.occupancy !== "occupied"}
                    value={area.identity_state}
                    onChange={(e) =>
                      patch({
                        identity_state: e.target
                          .value as Area["identity_state"],
                        observed_sku_id: null,
                      })
                    }
                  >
                    {area.identity_state === "known" && (
                      <option value="known">Catalog product assigned</option>
                    )}
                    <option value="unresolved">Unresolved</option>
                    <option value="unknown">Unknown product</option>
                    {area.identity_state === "not_applicable" && (
                      <option value="not_applicable">Not applicable</option>
                    )}
                  </select>
                </label>
                {area.observed_sku_id && (
                  <div className="assigned">
                    <img
                      src={"/api/catalog/" + area.observed_sku_id + "/image"}
                      alt=""
                    />
                    <div>
                      <small>ASSIGNED PRODUCT</small>
                      <strong>
                        {catalog.find((p) => p.sku_id === area.observed_sku_id)
                          ?.name || area.observed_sku_id}
                      </strong>
                    </div>
                  </div>
                )}
                <label className="checkbox">
                  <input
                    type="checkbox"
                    disabled={busy}
                    checked={area.truncated}
                    onChange={(e) => patch({ truncated: e.target.checked })}
                  />
                  Partially outside the image
                </label>
                <p className="coordinates">
                  {area.bbox_xyxy.map(Math.round).join(" / ")} px
                </p>
              </>
            ) : (
              <div className="selection-empty">
                <ScanLine size={28} />
                <p>
                  Select an area to inspect it.
                  <br />
                  Press <kbd>B</kbd> to draw a new one.
                </p>
              </div>
            )}
          </section>
          <div className="tabs">
            <button
              className={tab === "catalog" ? "active" : ""}
              onClick={() => setTab("catalog")}
            >
              Catalog <small>{catalog.length}</small>
            </button>
            <button
              className={tab === "proposals" ? "active" : ""}
              onClick={() => setTab("proposals")}
            >
              Proposals <small>{doc?.proposals.length || 0}</small>
            </button>
          </div>
          {tab === "catalog" ? (
            <>
              <div className="catalog-search">
                <Search size={15} />
                <input
                  placeholder="Find a product…"
                  aria-label="Search catalog"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
              </div>
              <p className="hint">
                {area
                  ? "Click a product to assign it."
                  : "Select or draw an area to assign a product."}
              </p>
              <div className="catalog-list">
                {products.map((p) => (
                  <button
                    key={p.sku_id}
                    disabled={!area || busy}
                    className={
                      "product " +
                      (area?.observed_sku_id === p.sku_id ? "chosen" : "")
                    }
                    onClick={() =>
                      patch({
                        occupancy: "occupied",
                        identity_state: "known",
                        observed_sku_id: p.sku_id,
                      })
                    }
                  >
                    <img src={p.thumbnail} alt="" loading="lazy" />
                    <span>
                      <strong>{p.name}</strong>
                      <small>{p.size || p.sku_id}</small>
                    </span>
                    {area?.observed_sku_id === p.sku_id && <Check size={15} />}
                  </button>
                ))}
                {!products.length && (
                  <p className="hint">No matching products.</p>
                )}
              </div>
            </>
          ) : (
            <>
              <div className="model-actions">
                <label>
                  Proposal source
                  <select
                    aria-label="Proposal source"
                    value={modelSource}
                    disabled={working}
                    onChange={(e) =>
                      setModelSource(e.target.value as "baseline" | "trained")
                    }
                  >
                    {modelOptions.map((m) => (
                      <option key={m.id} value={m.id} disabled={!m.available}>
                        {m.label}
                        {!m.available ? " (unavailable)" : ""}
                      </option>
                    ))}
                  </select>
                </label>
                <button onClick={predict} disabled={!doc || working}>
                  {working ? (
                    <LoaderCircle size={16} className="spin" />
                  ) : (
                    <Sparkles size={16} />
                  )}{" "}
                  {working ? "Running locally…" : "Run local models"}
                </button>
                <p>
                  {modelSource === "trained"
                    ? "Trained on your reviewed display areas. Detector only; assign product identities manually. " +
                      (modelOptions.find((m) => m.id === "trained")?.calibration
                        ? `Development-selected threshold ${modelOptions.find((m) => m.id === "trained")!.calibration!.threshold}. Experimental; independent-visit evaluation pending.`
                        : modelOptions.find((m) => m.id === "trained")
                              ?.promotion_passed
                          ? "Passed the pilot validation rule. New-visit testing is still needed."
                          : "Experimental: did not pass the pilot validation rule.")
                    : "Grounding DINO + DINOv2. Match scores are similarity, not calibrated probabilities."}
                </p>
                <p>
                  Showing:{" "}
                  {doc?.prediction_provenance?.model === "trained"
                    ? "trained detector"
                    : "original baseline"}
                  . Run the selected source to refresh proposals.
                </p>
              </div>
              <div className="proposal-list">
                {doc?.proposals.map((p, i) => (
                  <div className="proposal" key={i}>
                    <span>
                      Proposal {i + 1}
                      <small>Detection {p.confidence.toFixed(2)}</small>
                    </span>
                    <p>
                      {p.matches
                        ?.slice(0, 2)
                        .map(
                          (m) =>
                            `${catalog.find((c) => c.sku_id === m.sku_id)?.name || m.sku_id} (${m.score.toFixed(2)})`,
                        )
                        .join(" · ") ||
                        (doc?.prediction_provenance?.model === "trained"
                          ? "Detector only. Assign identity manually."
                          : "No catalog match")}
                    </p>
                    <button disabled={busy} onClick={() => addProposal(p)}>
                      Add for review
                    </button>
                  </div>
                ))}
                {!doc?.proposals.length && (
                  <p className="hint">Run the models or draw areas manually.</p>
                )}
              </div>
            </>
          )}
        </aside>
      </div>
      {importing && (
        <div className="modal-backdrop">
          <form className="modal" onSubmit={importFile}>
            <div className="section-title">
              <h2>Import a shelf image</h2>
              <button
                type="button"
                disabled={busy}
                onClick={() => setImporting(false)}
                aria-label="Close import"
              >
                <X size={18} />
              </button>
            </div>
            <p>
              JPEG, PNG, or HEIC, up to 25 MB. The original is preserved
              locally. The full image becomes the review region.
            </p>
            <label>
              Image
              <input
                name="file"
                type="file"
                accept=".jpg,.jpeg,.png,.heic,.heif"
                required
              />
            </label>
            <label>
              Capture session
              <input
                name="capture_group"
                placeholder="e.g. visit-002"
                required
                maxLength={100}
              />
            </label>
            <p>
              Use the same name for images from the same visit. A session
              belongs to one split.
            </p>
            <label>
              Dataset split
              <select name="split">
                <option value="development">Development</option>
                <option value="validation">Validation</option>
                <option value="test">
                  Final test (excluded from training export)
                </option>
              </select>
            </label>
            <button className="primary" disabled={busy} type="submit">
              {busy ? "Importing…" : "Import image"}
            </button>
            {message && <p role="alert">{message}</p>}
          </form>
        </div>
      )}
    </div>
  );
}
