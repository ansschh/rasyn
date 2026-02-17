/**
 * Rasyn API v2 client — job-based retrosynthesis with SSE streaming.
 *
 * When API_BASE is not set or the API is unreachable, falls back to
 * demo mode (mock data from mock-data.ts).
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export interface PlanStartResponse {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
}

export interface PlanRequest {
  smiles: string;
  top_k?: number;
  models?: string[];
  constraints?: Record<string, unknown> | null;
}

export interface PlanResult {
  job_id: string;
  smiles: string;
  status: "queued" | "running" | "completed" | "failed";
  routes: ApiRoute[];
  safety: SafetyResult | null;
  evidence: EvidenceHit[];
  green_chem: GreenChemResult | null;
  sourcing: SourcingResult | null;
  compute_time_ms: number | null;
  error: string | null;
  created_at: string | null;
}

export interface ApiRoute {
  route_id: string;
  rank: number;
  steps: ApiStep[];
  overall_score: number;
  num_steps: number;
  starting_materials: string[];
  all_purchasable: boolean;
}

export interface ApiStep {
  product: string;
  reactants: string[];
  model: string;
  score: number;
  rxn_class: string | null;
  conditions: Record<string, unknown> | null;
}

export interface SafetyResult {
  alerts: { name: string; severity: string; description: string | null }[];
  druglikeness: {
    mw: number;
    logp: number;
    hbd: number;
    hba: number;
    passes_lipinski: boolean;
    violations: string[];
  } | null;
  tox_flags: string[];
}

export interface EvidenceHit {
  rxn_smiles: string;
  similarity: number;
  source: string;
  year: number | null;
  title: string | null;
  doi: string | null;
}

export interface GreenChemResult {
  atom_economy: number | null;
  e_factor: number | null;
  solvent_score: number | null;
  details: Record<string, unknown> | null;
}

export interface SourcingResult {
  items: {
    smiles: string;
    vendor: string | null;
    price_per_gram: number | null;
    in_stock: boolean;
  }[];
  total_estimated_cost: number | null;
}

/**
 * Submit a retrosynthesis planning job.
 * Returns the job_id to track via SSE or polling.
 */
export async function startPlan(req: PlanRequest): Promise<PlanStartResponse> {
  const res = await fetch(`${API_BASE}/api/v2/plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error: ${res.status}`);
  }
  return res.json();
}

/**
 * Fetch the current job status and result.
 */
export async function fetchResult(jobId: string): Promise<PlanResult> {
  const res = await fetch(`${API_BASE}/api/v2/jobs/${jobId}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch job: ${res.status}`);
  }
  return res.json();
}

/**
 * Get the SSE stream URL for a job.
 */
export function getStreamUrl(jobId: string): string {
  return `${API_BASE}/api/v2/jobs/${jobId}/stream`;
}

/**
 * Check if the API is reachable.
 */
export async function checkApiHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/v1/health`, {
      signal: AbortSignal.timeout(3000),
    });
    return res.ok;
  } catch {
    return false;
  }
}
