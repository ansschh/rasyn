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


// ---------------------------------------------------------------------------
// Execute Module (Slice 7)
// ---------------------------------------------------------------------------

export interface ExperimentResult {
  id: string;
  stepNumber: number;
  reactionName: string;
  product_smiles: string;
  reactant_smiles: string[];
  protocol: string[];
  reagents: { name: string; role: string; equivalents: number; amount: string; mw: number }[];
  workupChecklist: string[];
  samples: { id: string; label: string; type: string; plannedAnalysis: string[]; status: string }[];
  elnExportReady: boolean;
  safety_notes: string[];
  estimated_time: string;
  tlc_checkpoints: string[];
  scale: string;
  route_id: string;
  created_at: string;
}

/**
 * Generate a lab-ready protocol from a route step.
 */
export async function generateProtocol(
  route: ApiRoute,
  stepIndex: number = 0,
  scale: string = "0.5 mmol",
): Promise<ExperimentResult> {
  const res = await fetch(`${API_BASE}/api/v2/execute/generate-protocol`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ route, step_index: stepIndex, scale }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Protocol error: ${res.status}`);
  }
  return res.json();
}

/**
 * Generate protocol from a completed job ID.
 */
export async function generateProtocolFromJob(
  jobId: string,
  stepIndex: number = 0,
  scale: string = "0.5 mmol",
): Promise<ExperimentResult> {
  const params = new URLSearchParams({ step_index: String(stepIndex), scale });
  const res = await fetch(`${API_BASE}/api/v2/execute/generate-from-job/${jobId}?${params}`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Protocol error: ${res.status}`);
  }
  return res.json();
}

/**
 * Export protocol as PDF.
 */
export async function exportProtocolPdf(
  route: ApiRoute,
  stepIndex: number = 0,
  scale: string = "0.5 mmol",
): Promise<Blob> {
  const res = await fetch(`${API_BASE}/api/v2/execute/export-pdf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ route, step_index: stepIndex, scale }),
  });
  if (!res.ok) throw new Error(`PDF export error: ${res.status}`);
  return res.blob();
}


// ---------------------------------------------------------------------------
// Analyze Module (Slice 8)
// ---------------------------------------------------------------------------

export interface AnalysisInterpretation {
  conversion: number;
  purity: number;
  majorProductConfirmed: boolean;
  impurities: { identity: string; percentage: number; flag?: string }[];
  anomalies: string[];
  summary: string;
}

export interface AnalysisFileResult {
  id: string;
  filename: string;
  instrument: string;
  sampleId: string;
  timestamp: string;
  fileSize: string;
  status: "pending" | "interpreted" | "anomaly";
  interpretation: AnalysisInterpretation | null;
}

export interface AnalysisBatchResult {
  files: AnalysisFileResult[];
  summary: { total: number; interpreted: number; anomalies: number; pending: number };
}

/**
 * Upload and analyze instrument files.
 */
export async function uploadAndAnalyze(
  files: File[],
  expectedProductSmiles?: string,
  expectedMw?: number,
  sampleId?: string,
): Promise<AnalysisBatchResult> {
  const form = new FormData();
  for (const f of files) {
    form.append("files", f);
  }
  if (expectedProductSmiles) form.append("expected_product_smiles", expectedProductSmiles);
  if (expectedMw != null) form.append("expected_mw", String(expectedMw));
  if (sampleId) form.append("sample_id", sampleId);

  const res = await fetch(`${API_BASE}/api/v2/analyze/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Analysis error: ${res.status}`);
  }
  return res.json();
}

/**
 * Get analysis results for a specific file.
 */
export async function getAnalysis(fileId: string): Promise<AnalysisFileResult> {
  const res = await fetch(`${API_BASE}/api/v2/analyze/${fileId}`);
  if (!res.ok) throw new Error(`Analysis fetch error: ${res.status}`);
  return res.json();
}

/**
 * Get all analyses for a sample.
 */
export async function getSampleAnalyses(sampleId: string): Promise<{
  sample_id: string;
  files: AnalysisFileResult[];
  total: number;
}> {
  const res = await fetch(`${API_BASE}/api/v2/analyze/sample/${sampleId}`);
  if (!res.ok) throw new Error(`Sample analysis error: ${res.status}`);
  return res.json();
}


// ---------------------------------------------------------------------------
// Learn Module (Slice 9)
// ---------------------------------------------------------------------------

export interface Insight {
  id: string;
  type: string;
  rule: string;
  source: string;
  confidence: number;
  timesApplied: number;
}

export interface InsightsResponse {
  insights: Insight[];
  total_experiments: number;
  total_applications: number;
}

export interface RankingFactor {
  factor: string;
  value: string;
  impact: string;
  detail: string;
}

export interface RankingExplanation {
  question: string;
  factors: RankingFactor[];
}

export interface PastExperimentEntry {
  id: string;
  date: string;
  target: string;
  reaction: string;
  conditions: string;
  outcome: string;
  yield: string | null;
  notes: string;
  scaffold: string;
  impactOnPlanning: string;
}

export interface LearnStatsResponse {
  total_experiments: number;
  total_reactions: number;
  success_rate: number;
  avg_yield: number | null;
  total_insights: number;
  past_experiments: PastExperimentEntry[];
}

export interface OutcomeRequest {
  reaction_id?: number | null;
  experiment_id?: string | null;
  reaction_smiles?: string | null;
  outcome: string;
  actual_yield?: number | null;
  failure_reason?: string | null;
  conditions?: Record<string, unknown> | null;
  notes?: string | null;
}

export interface OutcomeResponse {
  reaction_id: number;
  outcome: string;
  insights_generated: number;
  message: string;
}

/**
 * Record outcome of an experiment.
 */
export async function recordOutcome(req: OutcomeRequest): Promise<OutcomeResponse> {
  const res = await fetch(`${API_BASE}/api/v2/learn/record-outcome`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Record outcome error: ${res.status}`);
  return res.json();
}

/**
 * Get insights from institutional memory.
 */
export async function getInsights(
  targetSmiles?: string | null
): Promise<InsightsResponse> {
  const params = new URLSearchParams();
  if (targetSmiles) params.set("target_smiles", targetSmiles);
  const res = await fetch(`${API_BASE}/api/v2/learn/insights?${params}`);
  if (!res.ok) throw new Error(`Insights error: ${res.status}`);
  return res.json();
}

/**
 * Get ranking explanation for a route.
 */
export async function getRankingExplanation(
  jobId: string,
  routeIndex: number = 0
): Promise<RankingExplanation> {
  const params = new URLSearchParams({
    job_id: jobId,
    route_index: String(routeIndex),
  });
  const res = await fetch(`${API_BASE}/api/v2/learn/explain-ranking?${params}`);
  if (!res.ok) throw new Error(`Ranking explanation error: ${res.status}`);
  return res.json();
}

/**
 * Get past experiments with outcomes.
 */
export async function getPastExperiments(
  limit: number = 50
): Promise<LearnStatsResponse> {
  const res = await fetch(`${API_BASE}/api/v2/learn/experiments?limit=${limit}`);
  if (!res.ok) throw new Error(`Past experiments error: ${res.status}`);
  return res.json();
}


// ---------------------------------------------------------------------------
// Admin Module (Slice 10)
// ---------------------------------------------------------------------------

export interface AuditLogEntry {
  id: number;
  timestamp: string;
  user: string;
  action: string;
  resource: string;
  details: string | null;
  ip_address: string | null;
  status_code: number | null;
}

export interface AuditLogResponse {
  entries: AuditLogEntry[];
  total: number;
}

export interface GuardrailAlert {
  category: string;
  name: string;
  severity: string;
  description: string;
}

export interface GuardrailCheckResponse {
  smiles: string;
  blocked: boolean;
  alerts: GuardrailAlert[];
  requires_review: boolean;
}

/**
 * Get audit log entries.
 */
export async function getAuditLog(
  limit: number = 100,
  offset: number = 0
): Promise<AuditLogResponse> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  const res = await fetch(`${API_BASE}/api/v2/admin/audit-log?${params}`);
  if (!res.ok) throw new Error(`Audit log error: ${res.status}`);
  return res.json();
}

/**
 * Check guardrails for a molecule.
 */
export async function checkGuardrails(smiles: string): Promise<GuardrailCheckResponse> {
  const res = await fetch(`${API_BASE}/api/v2/admin/guardrails-check`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles }),
  });
  if (!res.ok) throw new Error(`Guardrails check error: ${res.status}`);
  return res.json();
}

/**
 * Get roles and permissions.
 */
export async function getRoles(): Promise<{
  roles: { role: string; permission_count: number; permissions: string[] }[];
}> {
  const res = await fetch(`${API_BASE}/api/v2/admin/roles`);
  if (!res.ok) throw new Error(`Roles error: ${res.status}`);
  return res.json();
}
