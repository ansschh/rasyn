export interface Constraint {
  id: string;
  label: string;
  type: "avoid_element" | "avoid_condition" | "prefer" | "limit";
  active: boolean;
  value?: string;
}

export interface ActionLogEntry {
  id: string;
  message: string;
  status: "pending" | "running" | "done" | "warning";
  detail?: string;
  timestamp: number;
  duration?: number;
}

export type AppState = "idle" | "planning" | "results";
export type NoveltyMode = "conservative" | "balanced" | "exploratory";
export type ProjectTab = "discover" | "plan" | "source" | "execute" | "analyze" | "learn" | "admin";
