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
  discovery: DiscoveryResult | null;
  compute_time_ms: number | null;
  error: string | null;
  created_at: string | null;
}

export interface ScoreBreakdown {
  roundtrip_confidence: number | null;
  step_efficiency: number | null;
  availability: number | null;
  safety: number | null;
  green_chemistry: number | null;
  precedent: number | null;
}

export interface ApiRoute {
  route_id: string;
  rank: number;
  steps: ApiStep[];
  overall_score: number;
  score_breakdown: ScoreBreakdown | null;
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
    catalog_id: string | null;
    price_per_gram: number | null;
    lead_time_days: number | null;
    in_stock: boolean;
    url: string | null;
  }[];
  total_estimated_cost: number | null;
  summary: {
    total_compounds: number;
    available: number;
    not_available: number;
    in_stock_offers: number;
  } | null;
}

export interface DiscoveryPaper {
  title: string;
  authors: string | null;
  year: number | null;
  doi: string | null;
  citation_count: number;
  source: string;
  journal: string | null;
  abstract: string | null;
  url: string | null;
}

export interface DiscoveryResult {
  papers: DiscoveryPaper[];
  compound_info: Record<string, unknown>;
  sources_queried: string[];
  total_results: number;
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

/**
 * Get sourcing quotes for a list of compounds.
 */
export async function getSourcingQuotes(smilesList: string[]): Promise<SourcingResult> {
  const res = await fetch(`${API_BASE}/api/v2/source/quote`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles_list: smilesList }),
  });
  if (!res.ok) throw new Error(`Sourcing error: ${res.status}`);
  return res.json();
}

/**
 * Start a literature discovery search.
 */
export async function startDiscovery(
  query: string,
  smiles?: string | null
): Promise<{ job_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/api/v2/discover/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, smiles, max_results: 20 }),
  });
  if (!res.ok) throw new Error(`Discovery error: ${res.status}`);
  return res.json();
}

/**
 * Quick synchronous discovery search (no job queue).
 */
export async function quickDiscovery(
  query: string,
  smiles?: string | null,
  maxResults: number = 10
): Promise<DiscoveryResult> {
  const params = new URLSearchParams({ query, max_results: String(maxResults) });
  if (smiles) params.set("smiles", smiles);
  const res = await fetch(`${API_BASE}/api/v2/discover/quick?${params}`);
  if (!res.ok) throw new Error(`Discovery error: ${res.status}`);
  return res.json();
}

/**
 * Find alternate building blocks for an unavailable compound.
 */
export async function findAlternates(
  smiles: string,
  topK: number = 5
): Promise<{ query_smiles: string; alternates: { smiles: string; similarity: number; source: string }[] }> {
  const res = await fetch(`${API_BASE}/api/v2/source/alternates`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles, top_k: topK }),
  });
  if (!res.ok) throw new Error(`Alternates error: ${res.status}`);
  return res.json();
}
