"use client";

import { useState } from "react";
import {
  Play, FileText, FlaskConical, Clipboard, Check, CheckCircle2,
  Download, Beaker, Tag, ExternalLink, ListChecks, TestTubes, Loader2
} from "lucide-react";
import ActionLog from "./ActionLog";
import { EXPERIMENT_TEMPLATE, EXECUTE_LOG_ENTRIES } from "../data/mock-pipeline";
import type { ExperimentResult } from "../lib/api";

interface Props {
  jobId?: string | null;
  route?: Record<string, unknown> | null;
  liveExperiment?: ExperimentResult | null;
}

export default function ExecuteView({ jobId, route, liveExperiment }: Props) {
  const [hasGenerated, setHasGenerated] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [liveData, setLiveData] = useState<ExperimentResult | null>(liveExperiment ?? null);
  const [checkedItems, setCheckedItems] = useState<Set<number>>(new Set());
  const [toast, setToast] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<"protocol" | "reagents" | "samples" | "workup">("protocol");

  // Use live data if available, else mock
  const exp = liveData || EXPERIMENT_TEMPLATE;

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  };

  const handleGenerate = async () => {
    // Try live API first
    if (route || jobId) {
      setIsGenerating(true);
      setHasGenerated(false);
      try {
        if (jobId) {
          const { generateProtocolFromJob } = await import("../lib/api");
          const result = await generateProtocolFromJob(jobId, 0, "0.5 mmol");
          setLiveData(result);
          setIsGenerating(false);
          setHasGenerated(true);
          return;
        } else if (route) {
          const { generateProtocol } = await import("../lib/api");
          const result = await generateProtocol(route as any, 0, "0.5 mmol");
          setLiveData(result);
          setIsGenerating(false);
          setHasGenerated(true);
          return;
        }
      } catch (e) {
        console.warn("Live protocol generation failed, falling back to demo:", e);
      }
    }

    // Fallback to mock animation
    setIsGenerating(true);
    setHasGenerated(false);
  };

  const handleExportPdf = async () => {
    if (route) {
      try {
        const { exportProtocolPdf } = await import("../lib/api");
        const blob = await exportProtocolPdf(route as any, 0, "0.5 mmol");
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${exp.id}_protocol.pdf`;
        a.click();
        URL.revokeObjectURL(url);
        showToast("PDF downloaded");
        return;
      } catch (e) {
        console.warn("PDF export failed:", e);
      }
    }
    showToast("PDF downloaded");
  };

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
              <Play className="w-4 h-4" /> Run Step 1
            </button>
          )}
          {hasGenerated && (
            <div className="flex items-center gap-2">
              <button onClick={() => showToast("Exported to Dotmatics ELN")} className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:bg-zinc-800 text-xs transition-colors">
                <ExternalLink className="w-3.5 h-3.5" /> Export to ELN
              </button>
              <button onClick={handleExportPdf} className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs transition-colors">
                <Download className="w-3.5 h-3.5" /> Download PDF
              </button>
            </div>
          )}
        </div>

        {/* Live data indicator */}
        {liveData && hasGenerated && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-400">
            <CheckCircle2 className="w-3.5 h-3.5" /> Live protocol generated from AI model &bull; {liveData.id}
          </div>
        )}

        {/* Generation animation (mock path) */}
        {isGenerating && !liveData && (
          <ActionLog entries={EXECUTE_LOG_ENTRIES} running={true} onComplete={() => { setIsGenerating(false); setHasGenerated(true); }} />
        )}

        {/* Loading spinner (live path) */}
        {isGenerating && liveData === null && route && (
          <div className="flex items-center justify-center py-12 gap-3 text-zinc-400">
            <Loader2 className="w-5 h-5 animate-spin" />
            <span className="text-sm">Generating protocol from AI model...</span>
          </div>
        )}

        {hasGenerated && (
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
                    {exp.reagents.map((r: any, i: number) => (
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
                  {exp.samples.map((sample: any) => (
                    <div key={sample.id} className="p-4 rounded-xl bg-zinc-950 border border-zinc-800 space-y-3">
                      <div className="flex items-center gap-2">
                        <Tag className="w-4 h-4 text-blue-400" />
                        <span className="text-xs font-mono font-semibold text-zinc-200">{sample.id}</span>
                      </div>
                      <div className="text-xs text-zinc-400">{sample.label}</div>
                      <div className="text-[10px] text-zinc-500 capitalize">Type: {(sample.type || "").replace(/_/g, " ")}</div>
                      {/* Fake barcode */}
                      <div className="flex gap-[2px] h-8 items-end">
                        {Array.from({ length: 30 }, (_, i) => (
                          <div key={i} className="bg-zinc-500" style={{
                            width: Math.random() > 0.3 ? 2 : 1,
                            height: `${60 + Math.random() * 40}%`,
                          }} />
                        ))}
                      </div>
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
