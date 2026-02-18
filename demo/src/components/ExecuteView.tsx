"use client";

import { useState, useEffect } from "react";
import {
  Play, FileText, FlaskConical, Check, CheckCircle2,
  Download, Beaker, Tag, ExternalLink, ListChecks, Loader2, AlertCircle, Upload, ChevronDown
} from "lucide-react";
import type { ExperimentResult } from "../lib/api";

interface Props {
  jobId?: string | null;
  route?: Record<string, unknown> | null;
  liveExperiment?: ExperimentResult | null;
}

const EXPORT_FORMATS = [
  { id: "pdf" as const, label: "PDF", ext: ".pdf", mime: "application/pdf" },
  { id: "json" as const, label: "JSON", ext: ".json", mime: "application/json" },
  { id: "csv" as const, label: "CSV", ext: ".csv", mime: "text/csv" },
  { id: "sdf" as const, label: "SDF", ext: ".sdf", mime: "chemical/x-mdl-sdfile" },
];

export default function ExecuteView({ jobId, route, liveExperiment }: Props) {
  const [hasGenerated, setHasGenerated] = useState(!!liveExperiment);
  const [isGenerating, setIsGenerating] = useState(false);
  const [liveData, setLiveData] = useState<ExperimentResult | null>(liveExperiment ?? null);
  const [checkedItems, setCheckedItems] = useState<Set<number>>(new Set());
  const [toast, setToast] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<"protocol" | "reagents" | "samples" | "workup">("protocol");
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [elnAvailable, setElnAvailable] = useState(false);
  const [exportingFormat, setExportingFormat] = useState<string | null>(null);

  // Check if ELN webhook is configured
  useEffect(() => {
    (async () => {
      try {
        const { getIntegrationStatus } = await import("../lib/api");
        const status = await getIntegrationStatus();
        const eln = status.integrations.find(i => i.name === "ELN Webhook");
        setElnAvailable(eln?.status === "connected");
      } catch {
        // ignore
      }
    })();
  }, []);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  };

  const handleGenerate = async () => {
    if (!route && !jobId) {
      setError("No route or job available. Run a synthesis plan first.");
      return;
    }

    setIsGenerating(true);
    setError(null);
    try {
      if (jobId) {
        const { generateProtocolFromJob } = await import("../lib/api");
        const result = await generateProtocolFromJob(jobId, 0, "0.5 mmol");
        setLiveData(result);
        setHasGenerated(true);
      } else if (route) {
        const { generateProtocol } = await import("../lib/api");
        const result = await generateProtocol(route as any, 0, "0.5 mmol");
        setLiveData(result);
        setHasGenerated(true);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Protocol generation failed");
    } finally {
      setIsGenerating(false);
    }
  };

  const handleExport = async (format: "pdf" | "json" | "csv" | "sdf") => {
    if (!liveData) {
      setError("Generate a protocol first");
      return;
    }
    setExportingFormat(format);
    setShowExportMenu(false);
    try {
      const { exportExperiment } = await import("../lib/api");
      const blob = await exportExperiment(liveData, format);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const ext = EXPORT_FORMATS.find(f => f.id === format)?.ext || ".dat";
      a.download = `${liveData.id}${ext}`;
      a.click();
      URL.revokeObjectURL(url);
      showToast(`${format.toUpperCase()} downloaded`);
    } catch (e) {
      setError(e instanceof Error ? e.message : `${format.toUpperCase()} export failed`);
    } finally {
      setExportingFormat(null);
    }
  };

  const handlePushToEln = async () => {
    if (!liveData) return;
    try {
      const { exportToWebhook } = await import("../lib/api");
      const result = await exportToWebhook(liveData);
      if (result.status === "success") {
        showToast("Pushed to ELN successfully");
      } else {
        setError(result.message || "ELN push failed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "ELN push failed");
    }
  };

  // No route/job available
  if (!route && !jobId) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <div className="text-center space-y-3">
          <FlaskConical className="w-10 h-10 text-zinc-600 mx-auto" />
          <p className="text-sm text-zinc-400">Run a synthesis plan first to generate protocols</p>
        </div>
      </div>
    );
  }

  const exp = liveData;

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">Experiment Orchestrator</h2>
            <p className="text-xs text-zinc-500">Auto-generate protocols, reagent tables, sample IDs, and ELN entries</p>
          </div>
          {!hasGenerated && !isGenerating && (
            <button onClick={handleGenerate} className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium transition-colors">
              <Play className="w-4 h-4" /> Generate Protocol
            </button>
          )}
          {hasGenerated && exp && (
            <div className="flex items-center gap-2">
              {/* Multi-format export dropdown */}
              <div className="relative">
                <button
                  onClick={() => setShowExportMenu(!showExportMenu)}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs transition-colors"
                >
                  {exportingFormat ? (
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Download className="w-3.5 h-3.5" />
                  )}
                  Export
                  <ChevronDown className="w-3 h-3" />
                </button>
                {showExportMenu && (
                  <div className="absolute right-0 top-full mt-1 bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl z-50 py-1 min-w-[120px]">
                    {EXPORT_FORMATS.map(fmt => (
                      <button
                        key={fmt.id}
                        onClick={() => handleExport(fmt.id)}
                        className="w-full text-left px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-800 hover:text-white transition-colors"
                      >
                        {fmt.label} ({fmt.ext})
                      </button>
                    ))}
                  </div>
                )}
              </div>
              {/* ELN push button */}
              {elnAvailable && (
                <button
                  onClick={handlePushToEln}
                  className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs transition-colors"
                >
                  <Upload className="w-3.5 h-3.5" /> Push to ELN
                </button>
              )}
              {!elnAvailable && (
                <span className="text-[10px] text-zinc-500">No ELN configured</span>
              )}
            </div>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-xs text-red-400">
            <AlertCircle className="w-4 h-4 shrink-0" /> {error}
          </div>
        )}

        {/* Loading spinner */}
        {isGenerating && (
          <div className="flex items-center justify-center py-12 gap-3 text-zinc-400">
            <Loader2 className="w-5 h-5 animate-spin" />
            <span className="text-sm">Generating protocol from AI model...</span>
          </div>
        )}

        {hasGenerated && exp && (
          <>
            {/* Experiment header */}
            <div className="flex items-center gap-4 p-4 rounded-xl bg-zinc-900 border border-zinc-800">
              <div className="w-12 h-12 rounded-xl bg-emerald-500/15 flex items-center justify-center">
                <FlaskConical className="w-6 h-6 text-emerald-400" />
              </div>
              <div className="flex-1">
                <div className="text-sm font-semibold">{exp.id}</div>
                <div className="text-xs text-zinc-400">Step {exp.stepNumber}: {exp.reactionName}</div>
              </div>
              <div className="flex items-center gap-1 text-xs text-emerald-400">
                <CheckCircle2 className="w-4 h-4" /> ELN-ready
              </div>
            </div>

            {/* Section tabs */}
            <div className="flex border-b border-zinc-800">
              {([
                { id: "protocol" as const, icon: FileText, label: "Protocol" },
                { id: "reagents" as const, icon: Beaker, label: "Reagents" },
                { id: "samples" as const, icon: Tag, label: "Samples" },
                { id: "workup" as const, icon: ListChecks, label: "Workup" },
              ]).map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveSection(tab.id)}
                  className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors ${
                    activeSection === tab.id
                      ? "border-emerald-500 text-emerald-400"
                      : "border-transparent text-zinc-500 hover:text-zinc-300"
                  }`}
                >
                  <tab.icon className="w-3.5 h-3.5" /> {tab.label}
                </button>
              ))}
            </div>

            {/* Protocol */}
            {activeSection === "protocol" && (
              <div className="bg-zinc-950 rounded-xl p-5 border border-zinc-800 space-y-3">
                <div className="text-[10px] text-zinc-500 uppercase tracking-wider">Procedure</div>
                <ol className="space-y-3">
                  {exp.protocol.map((step: string, i: number) => (
                    <li key={i} className="flex items-start gap-3 text-xs">
                      <span className="w-6 h-6 rounded-full bg-zinc-800 flex items-center justify-center text-zinc-400 font-medium shrink-0 mt-0.5">{i + 1}</span>
                      <span className="text-zinc-300 leading-relaxed">{step}</span>
                    </li>
                  ))}
                </ol>
              </div>
            )}

            {/* Reagent table */}
            {activeSection === "reagents" && (
              <div className="bg-zinc-950 rounded-xl p-5 border border-zinc-800">
                <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-3">Reagent &amp; Stoichiometry Table</div>
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-zinc-500 border-b border-zinc-800">
                      <th className="text-left pb-2 font-medium">Reagent</th>
                      <th className="text-left pb-2 font-medium">Role</th>
                      <th className="text-right pb-2 font-medium">Equiv.</th>
                      <th className="text-right pb-2 font-medium">Amount</th>
                      <th className="text-right pb-2 font-medium">MW</th>
                    </tr>
                  </thead>
                  <tbody>
                    {exp.reagents.map((r, i) => (
                      <tr key={i} className="border-b border-zinc-800/50">
                        <td className="py-2.5 text-zinc-200 font-medium">{r.name}</td>
                        <td className="py-2.5 text-zinc-400">{r.role}</td>
                        <td className="py-2.5 text-right text-zinc-300">{r.equivalents > 0 ? r.equivalents.toFixed(1) : "\u2014"}</td>
                        <td className="py-2.5 text-right text-zinc-300">{r.amount}</td>
                        <td className="py-2.5 text-right text-zinc-400">{r.mw > 0 ? r.mw.toFixed(1) : "\u2014"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Sample IDs */}
            {activeSection === "samples" && (
              <div className="space-y-3">
                <div className="text-[10px] text-zinc-500 uppercase tracking-wider">Sample Tracking</div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {exp.samples.map((sample) => (
                    <div key={sample.id} className="p-4 rounded-xl bg-zinc-950 border border-zinc-800 space-y-3">
                      <div className="flex items-center gap-2">
                        <Tag className="w-4 h-4 text-blue-400" />
                        <span className="text-xs font-mono font-semibold text-zinc-200">{sample.id}</span>
                      </div>
                      <div className="text-xs text-zinc-400">{sample.label}</div>
                      <div className="text-[10px] text-zinc-500 capitalize">Type: {(sample.type || "").replace(/_/g, " ")}</div>
                      <div className="space-y-1">
                        <div className="text-[10px] text-zinc-500">Planned analytics:</div>
                        <div className="flex flex-wrap gap-1">
                          {(sample.plannedAnalysis || []).map((a: string) => (
                            <span key={a} className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-400 border border-blue-500/30">{a}</span>
                          ))}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Workup checklist */}
            {activeSection === "workup" && (
              <div className="bg-zinc-950 rounded-xl p-5 border border-zinc-800 space-y-3">
                <div className="text-[10px] text-zinc-500 uppercase tracking-wider">Workup &amp; Purification Checklist</div>
                <div className="space-y-2">
                  {exp.workupChecklist.map((item: string, i: number) => (
                    <label key={i} className="flex items-center gap-3 p-2 rounded-lg hover:bg-zinc-900 cursor-pointer transition-colors">
                      <input
                        type="checkbox"
                        checked={checkedItems.has(i)}
                        onChange={() => setCheckedItems(prev => {
                          const next = new Set(prev);
                          next.has(i) ? next.delete(i) : next.add(i);
                          return next;
                        })}
                        className="w-4 h-4 rounded border-zinc-600 text-emerald-500 focus:ring-emerald-500/20 bg-zinc-800"
                      />
                      <span className={`text-xs ${checkedItems.has(i) ? "text-zinc-500 line-through" : "text-zinc-300"}`}>{item}</span>
                    </label>
                  ))}
                </div>
                <div className="text-[10px] text-zinc-500 pt-2 border-t border-zinc-800">
                  {checkedItems.size}/{exp.workupChecklist.length} completed
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Toast */}
      {toast && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[60] flex items-center gap-2 px-4 py-2.5 rounded-lg bg-zinc-800 border border-zinc-700 text-sm text-emerald-400 shadow-xl animate-fade-in">
          <CheckCircle2 className="w-4 h-4" /> {toast}
        </div>
      )}
    </div>
  );
}
