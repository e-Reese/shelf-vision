export type Box = [number, number, number, number];
export type ROI = { roi_id: string; bbox_xyxy: Box };
export type Area = {
  region_id: string;
  frame_id: string;
  review_roi_id: string;
  bbox_xyxy: Box;
  class_name: "display_area";
  occupancy: "occupied" | "empty" | "mixed" | "unclear";
  identity_state: "known" | "unknown" | "unresolved" | "not_applicable";
  observed_sku_id: string | null;
  intended_sku_id: string | null;
  truncated: boolean;
};
export type Frame = {
  frame_id: string;
  name: string;
  width: number;
  height: number;
  capture_group: string;
  split: string;
  rois: ROI[];
  status: string;
  revision: number;
  area_count: number;
};
export type Product = {
  sku_id: string;
  name: string;
  size: string | null;
  status: string;
  thumbnail: string;
};
export type ModelOption = {
  id: "baseline" | "trained";
  label: string;
  available: boolean;
  promotion_passed?: boolean;
  calibration?: { threshold: number; independent_test_pending: boolean } | null;
};
export type Proposal = {
  review_roi_id?: string;
  bbox_xyxy: Box;
  confidence: number;
  label?: string;
  matches?: { sku_id: string; score: number }[];
};
export type Document = {
  meta: Frame;
  annotations: Area[];
  proposals: Proposal[];
  original_proposals: Proposal[];
  prediction_provenance?: { model?: string; source?: string };
  revision: number;
  status: string;
  updated: string;
};
export type Job = {
  id: string;
  frame_id: string;
  status: "queued" | "running" | "completed" | "failed";
  error: string | null;
};
