import { useEffect, useRef, useState } from "react";
import {
  Maximize2,
  Minus,
  Plus,
  MousePointer2,
  Hand,
  SquareDashed,
  Layers,
  ScanLine,
} from "lucide-react";
import type { Area, Document, Box, Product } from "./types";
import {
  drawBox,
  fitCamera,
  moveBox,
  pointInBox,
  type Camera,
  type Point,
} from "./geometry";

type Tool = "select" | "draw" | "pan";
type Gesture = {
  type: "pan" | "draw" | "move" | "resize";
  start: Point;
  client: Point;
  camera: Camera;
  roi: Box;
  roiId: string;
  area?: Area;
  anchor?: Point;
  box?: Box;
};
export function Canvas({
  doc,
  areas,
  selected,
  onSelect,
  onChange,
  catalog,
  tool,
  setTool,
  notice,
  showProposals,
  setShowProposals,
}: {
  doc: Document;
  areas: Area[];
  selected: string | null;
  onSelect: (id: string | null) => void;
  onChange: (a: Area[]) => void;
  catalog: Product[];
  tool: Tool;
  setTool: (t: Tool) => void;
  notice: (s: string) => void;
  showProposals: boolean;
  setShowProposals: (s: boolean) => void;
}) {
  const host = useRef<HTMLDivElement>(null),
    svg = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState({ width: 800, height: 700 }),
    [camera, setCamera] = useState<Camera>({
      x: doc.meta.width / 2,
      y: doc.meta.height / 2,
      width: doc.meta.width * 1.1,
    });
  const [gesture, setGesture] = useState<Gesture | null>(null),
    [preview, setPreview] = useState<Box | null>(null);
  const space = useRef(false),
    cameraRef = useRef(camera);
  cameraRef.current = camera;
  const unit = camera.width / size.width,
    height = (camera.width * size.height) / size.width;
  const fit = (full = false) => {
    const r = doc.meta.rois.map((r) => r.bbox_xyxy);
    const b: Box = full
      ? [0, 0, doc.meta.width, doc.meta.height]
      : [
          Math.min(...r.map((b) => b[0])),
          Math.min(...r.map((b) => b[1])),
          Math.max(...r.map((b) => b[2])),
          Math.max(...r.map((b) => b[3])),
        ];
    setCamera(fitCamera(b, size.width, size.height));
  };
  useEffect(() => {
    if (!host.current) return;
    const o = new ResizeObserver((entries) => {
      const r = entries[0].contentRect;
      setSize({ width: r.width || 800, height: r.height || 700 });
    });
    o.observe(host.current);
    return () => o.disconnect();
  }, []);
  useEffect(() => {
    fit();
    setGesture(null);
    setPreview(null);
  }, [doc.meta.frame_id, size.width, size.height]);
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (
        e.code === "Space" &&
        !(
          e.target instanceof HTMLInputElement ||
          e.target instanceof HTMLTextAreaElement ||
          e.target instanceof HTMLSelectElement
        )
      ) {
        space.current = true;
        e.preventDefault();
      }
    };
    const up = () => {
      space.current = false;
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", up);
    };
  }, []);
  useEffect(() => {
    const el = svg.current;
    if (!el) return;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const rx = (e.clientX - rect.left) / rect.width - 0.5,
        ry = (e.clientY - rect.top) / rect.height - 0.5;
      setCamera((c) => {
        const next = Math.max(
          doc.meta.width * 0.08,
          Math.min(doc.meta.width * 6, c.width * Math.exp(e.deltaY * 0.0015)),
        );
        return {
          x: c.x + rx * (c.width - next),
          y: c.y + (ry * (c.width - next) * rect.height) / rect.width,
          width: next,
        };
      });
    };
    el.addEventListener("wheel", wheel, { passive: false });
    return () => el.removeEventListener("wheel", wheel);
  }, [doc.meta.width]);
  const point = (e: React.PointerEvent): Point => {
    const rect = svg.current!.getBoundingClientRect();
    return [
      camera.x -
        camera.width / 2 +
        ((e.clientX - rect.left) / rect.width) * camera.width,
      camera.y - height / 2 + ((e.clientY - rect.top) / rect.height) * height,
    ];
  };
  const begin = (e: React.PointerEvent, area?: Area, corner?: number) => {
    if (e.button !== 0 && e.button !== 1) return;
    e.preventDefault();
    e.stopPropagation();
    svg.current?.setPointerCapture(e.pointerId);
    const p = point(e),
      base = {
        start: p,
        client: [e.clientX, e.clientY] as Point,
        camera,
        roi: [0, 0, doc.meta.width, doc.meta.height] as Box,
        roiId: "",
      };
    if (tool === "pan" || space.current || e.button === 1) {
      setGesture({ ...base, type: "pan" });
      return;
    }
    if (tool === "draw" && corner === undefined) {
      const roi = doc.meta.rois.find((r) => pointInBox(p, r.bbox_xyxy));
      if (!roi) {
        notice("Start inside a cyan review region.");
        return;
      }
      setGesture({
        ...base,
        type: "draw",
        roi: roi.bbox_xyxy,
        roiId: roi.roi_id,
      });
      setPreview([p[0], p[1], p[0], p[1]]);
      return;
    }
    if (area) {
      onSelect(area.region_id);
      const roi = doc.meta.rois.find((r) => r.roi_id === area.review_roi_id)!;
      const b = area.bbox_xyxy;
      const anchors: Point[] = [
        [b[2], b[3]],
        [b[0], b[3]],
        [b[0], b[1]],
        [b[2], b[1]],
      ];
      setGesture({
        ...base,
        type: corner === undefined ? "move" : "resize",
        area,
        roi: roi.bbox_xyxy,
        roiId: roi.roi_id,
        anchor: corner === undefined ? undefined : anchors[corner],
      });
      setPreview(b);
    } else onSelect(null);
  };
  const move = (e: React.PointerEvent) => {
    if (!gesture) return;
    if (gesture.type === "pan") {
      const dx = e.clientX - gesture.client[0],
        dy = e.clientY - gesture.client[1];
      setCamera({
        ...gesture.camera,
        x: gesture.camera.x - (dx * gesture.camera.width) / size.width,
        y: gesture.camera.y - (dy * gesture.camera.width) / size.width,
      });
      return;
    }
    const p = point(e);
    setPreview(
      gesture.type === "draw"
        ? drawBox(gesture.start, p, gesture.roi)
        : gesture.type === "resize"
          ? drawBox(gesture.anchor!, p, gesture.roi)
          : moveBox(
              gesture.area!.bbox_xyxy,
              [p[0] - gesture.start[0], p[1] - gesture.start[1]],
              gesture.roi,
            ),
    );
  };
  const end = (e: React.PointerEvent) => {
    if (svg.current?.hasPointerCapture(e.pointerId))
      svg.current.releasePointerCapture(e.pointerId);
    if (
      gesture &&
      preview &&
      gesture.type !== "pan" &&
      preview[2] - preview[0] >= 4 &&
      preview[3] - preview[1] >= 4
    ) {
      const b = preview.map((v) => Math.round(v * 100) / 100) as Box;
      if (gesture.type === "draw") {
        const area: Area = {
          region_id: crypto.randomUUID(),
          frame_id: doc.meta.frame_id,
          review_roi_id: gesture.roiId,
          bbox_xyxy: b,
          class_name: "display_area",
          occupancy: "occupied",
          identity_state: "unknown",
          observed_sku_id: null,
          intended_sku_id: null,
          truncated: false,
        };
        onChange([...areas, area]);
        onSelect(area.region_id);
      } else if (JSON.stringify(b) !== JSON.stringify(gesture.area!.bbox_xyxy))
        onChange(
          areas.map((a) =>
            a.region_id === gesture.area!.region_id
              ? { ...a, bbox_xyxy: b }
              : a,
          ),
        );
    }
    setGesture(null);
    setPreview(null);
  };
  const zoom = (factor: number) =>
    setCamera((c) => ({
      ...c,
      width: Math.max(
        doc.meta.width * 0.08,
        Math.min(doc.meta.width * 6, c.width * factor),
      ),
    }));
  return (
    <section className="canvas-section">
      <div className="canvas-toolbar">
        <div className="segmented">
          <button
            title="Select and move (V)"
            aria-label="Select tool"
            className={tool === "select" ? "active" : ""}
            onClick={() => setTool("select")}
          >
            <MousePointer2 size={17} />
          </button>
          <button
            title="Draw display area (B)"
            aria-label="Draw area tool"
            className={tool === "draw" ? "active" : ""}
            onClick={() => setTool("draw")}
          >
            <SquareDashed size={18} />
            <span>Draw area</span>
          </button>
          <button
            title="Pan (H or hold Space)"
            aria-label="Pan tool"
            className={tool === "pan" ? "active" : ""}
            onClick={() => setTool("pan")}
          >
            <Hand size={17} />
          </button>
        </div>
        <div className="canvas-options">
          <button
            className={showProposals ? "toggle on" : "toggle"}
            onClick={() => setShowProposals(!showProposals)}
          >
            <Layers size={15} /> Proposals
          </button>
          <button
            title="Fit review regions"
            aria-label="Fit review regions"
            onClick={() => fit()}
          >
            <ScanLine size={17} />
          </button>
          <button
            title="Fit entire photo"
            aria-label="Fit entire photo"
            onClick={() => fit(true)}
          >
            <Maximize2 size={17} />
          </button>
        </div>
      </div>
      <div className={`stage ${tool}`} ref={host}>
        <svg
          ref={svg}
          role="img"
          aria-label="Shelf annotation canvas"
          width="100%"
          height="100%"
          viewBox={`${camera.x - camera.width / 2} ${camera.y - height / 2} ${camera.width} ${height}`}
          onPointerDown={(e) => begin(e)}
          onPointerMove={move}
          onPointerUp={end}
          onPointerCancel={() => {
            setGesture(null);
            setPreview(null);
          }}
        >
          <image
            href={`/api/frames/${doc.meta.frame_id}/image`}
            width={doc.meta.width}
            height={doc.meta.height}
          />
          {doc.meta.rois.map((r, i) => (
            <g key={r.roi_id} pointerEvents="none">
              <rect
                x={r.bbox_xyxy[0]}
                y={r.bbox_xyxy[1]}
                width={r.bbox_xyxy[2] - r.bbox_xyxy[0]}
                height={r.bbox_xyxy[3] - r.bbox_xyxy[1]}
                fill="none"
                stroke="#75d6df"
                strokeWidth={unit * 1.5}
                strokeDasharray={`${unit * 7} ${unit * 5}`}
              />
              <text
                x={r.bbox_xyxy[0] + unit * 6}
                y={r.bbox_xyxy[1] - unit * 7}
                fill="#a5f1f0"
                stroke="#17312c"
                strokeWidth={unit * 2}
                paintOrder="stroke"
                fontSize={unit * 11}
              >
                REVIEW REGION {i + 1}
              </text>
            </g>
          ))}
          {showProposals &&
            doc.proposals.map((p, i) => (
              <rect
                key={i}
                x={p.bbox_xyxy[0]}
                y={p.bbox_xyxy[1]}
                width={p.bbox_xyxy[2] - p.bbox_xyxy[0]}
                height={p.bbox_xyxy[3] - p.bbox_xyxy[1]}
                fill="none"
                stroke="#d6b0ff"
                strokeWidth={unit * 1.4}
                strokeDasharray={`${unit * 4} ${unit * 4}`}
                pointerEvents="none"
              />
            ))}
          {areas.map((a, i) => {
            const b =
                gesture?.area?.region_id === a.region_id && preview
                  ? preview
                  : a.bbox_xyxy,
              isSelected = selected === a.region_id;
            const color = isSelected
              ? "#ecff85"
              : a.identity_state === "known"
                ? "#72e0ab"
                : a.occupancy === "empty"
                  ? "#d7e0df"
                  : "#ffc37a";
            const name =
              catalog.find((p) => p.sku_id === a.observed_sku_id)?.name ||
              {
                unknown: "Unknown product",
                not_applicable:
                  a.occupancy === "empty" ? "Empty area" : "Mixed products",
                unresolved: "Needs identification",
                known: "Product",
              }[a.identity_state];
            return (
              <g key={a.region_id}>
                <rect
                  data-area-id={a.region_id}
                  x={b[0]}
                  y={b[1]}
                  width={b[2] - b[0]}
                  height={b[3] - b[1]}
                  fill={isSelected ? "#ecff851c" : "#00000008"}
                  stroke={color}
                  strokeWidth={unit * (isSelected ? 2.5 : 1.5)}
                  onPointerDown={(e) => begin(e, a)}
                  style={{ cursor: tool === "select" ? "move" : undefined }}
                />
                <g pointerEvents="none">
                  <rect
                    x={b[0]}
                    y={b[1]}
                    width={Math.min(
                      b[2] - b[0],
                      unit * (isSelected ? 210 : 32),
                    )}
                    height={unit * 21}
                    fill={color}
                  />
                  <text
                    x={b[0] + unit * 5}
                    y={b[1] + unit * 15}
                    fill="#10251d"
                    fontSize={unit * 11}
                    fontWeight="600"
                  >
                    {isSelected
                      ? `${String(i + 1).padStart(2, "0")} · ${name.slice(0, 28)}`
                      : String(i + 1).padStart(2, "0")}
                  </text>
                </g>
                {(isSelected || tool === "draw") &&
                  [
                    [b[0], b[1]],
                    [b[2], b[1]],
                    [b[2], b[3]],
                    [b[0], b[3]],
                  ].map(([x, y], c) => (
                    <rect
                      key={c}
                      data-handle={c}
                      x={x - unit * 5}
                      y={y - unit * 5}
                      width={unit * 10}
                      height={unit * 10}
                      rx={unit * 1.5}
                      fill="#ecff85"
                      stroke="#163b2e"
                      strokeWidth={unit}
                      onPointerDown={(e) => begin(e, a, c)}
                      style={{
                        cursor: c % 2 === 0 ? "nwse-resize" : "nesw-resize",
                      }}
                    />
                  ))}
              </g>
            );
          })}
          {gesture?.type === "draw" && preview && (
            <rect
              x={preview[0]}
              y={preview[1]}
              width={preview[2] - preview[0]}
              height={preview[3] - preview[1]}
              fill="#ecff8522"
              stroke="#ecff85"
              strokeWidth={unit * 2}
              pointerEvents="none"
            />
          )}
        </svg>
        <div className="stage-caption">
          <span className="cyan-dot" /> Review regions only{" "}
          <span className="caption-sep">/</span> {doc.meta.width} ×{" "}
          {doc.meta.height}
        </div>
        <div className="zoom-controls">
          <button
            title="Zoom out"
            aria-label="Zoom out"
            onClick={() => zoom(1.25)}
          >
            <Minus size={15} />
          </button>
          <span>{Math.round((doc.meta.width / camera.width) * 100)}%</span>
          <button
            title="Zoom in"
            aria-label="Zoom in"
            onClick={() => zoom(0.8)}
          >
            <Plus size={15} />
          </button>
        </div>
      </div>
      <div className="canvas-footer">
        <span>
          <i className="legend-green" /> Identified
        </span>
        <span>
          <i className="legend-amber" /> Needs review
        </span>
        <span className="footer-help">
          Scroll to zoom · Space + drag to pan
        </span>
      </div>
    </section>
  );
}
