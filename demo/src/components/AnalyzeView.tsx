"use client";

import { useState, useRef, useCallback } from "react";
import {
  FileUp, CheckCircle2, AlertTriangle, Clock, ChevronDown, ChevronUp,
  Activity, Beaker, Eye, FileText, Loader2, AlertCircle
} from "lucide-react";
import type { AnalysisFileResult } from "../lib/api";

interface Props {
  productSmiles?: string;
  expectedMw?: number;
}

function StatusBadge({ status }: { status: string }) {
  const config: Record<string, { icon: typeof CheckCircle2; text: string; cls: string }> = {
    interpreted: { icon: CheckCircle2, text: "Interpreted", cls: "text-emerald-400 bg-emerald-500/15 border-emerald-500/30" },
    anomaly: { icon: AlertTriangle, text: "Anomaly", cls: "text-red-400 bg-red-500/15 border-red-500/30" },
    pending: { icon: Clock, text: "Pending", cls: "text-zinc-400 bg-zinc-500/15 border-zinc-500/30" },
  };
  const { icon: Icon, text, cls } = config[status] || config.pending;
  return (
    <span className={`inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full border ${cls}`}>
      <Icon className="w-3 h-3" /> {text}
    </span>
  );
}

function InstrumentBadge({ instrument }: { instrument: string }) {
  const colors: Record<string, string> = {
    LCMS: "text-blue-400 bg-blue-500/15 border-blue-500/30",
    HPLC: "text-purple-400 bg-purple-500/15 border-purple-500/30",
    NMR: "text-amber-400 bg-amber-500/15 border-amber-500/30",
    IR: "text-cyan-400 bg-cyan-500/15 border-cyan-500/30",
  };
  return (
    <span className={`text-[10px] px-2 py-0.5 rounded-full border font-medium ${colors[instrument] || "text-zinc-400 bg-zinc-500/15 border-zinc-500/30"}`}>
      {instrument}
    </span>
  );
}

export default function AnalyzeView({ productSmiles, expectedMw }: Props) {
  const [hasAnalyzed, setHasAnalyzed] = useState(false);
  const [expandedFile, setExpandedFile] = useState<string | null>(null);
  const [liveResults, setLiveResults] = useState<AnalysisFileResult[] | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    setIsUploading(true);
    setError(null);
    try {
      const { uploadAndAnalyze } = await import("../lib/api");
      const result = await uploadAndAnalyze(
        Array.from(files),
        productSmiles,
        expectedMw,
      );
      setLiveResults(result.files);
      setHasAnalyzed(true);
      if (result.files.length > 0) {
        setExpandedFile(result.files[0].id);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setIsUploading(false);
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    handleFileUpload(e.dataTransfer.files);
  }, [productSmiles, expectedMw]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h2 className="text-lg font-semibold">Analytical Inbox</h2>
          <p className="text-xs text-zinc-500">Upload and auto-interpret LCMS / HPLC / NMR data</p>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-xs text-red-400">
            <AlertCircle className="w-4 h-4 shrink-0" /> {error}
          </div>
        )}

        {/* Drop zone */}
        {!hasAnalyzed && !isUploading && (
          <div
            className="border-2 border-dashed border-zinc-700 rounded-xl p-12 text-center hover:border-zinc-600 transition-colors cursor-pointer"
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <FileUp className="w-10 h-10 text-zinc-600 mx-auto mb-3" />
            <p className="text-sm text-zinc-400">Drag &amp; drop instrument files or click to browse</p>
            <p className="text-[10px] text-zinc-600 mt-1">Supports: .raw, .d, .mzML, .fid, .jdx, .csv</p>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".raw,.d,.mzML,.mzxml,.fid,.jdx,.dx,.csv,.txt,.spc,.spa"
              className="hidden"
              onChange={e => handleFileUpload(e.target.files)}
            />
          </div>
        )}

        {/* Upload spinner */}
        {isUploading && (
          <div className="flex items-center justify-center py-12 gap-3 text-zinc-400">
            <Loader2 className="w-5 h-5 animate-spin" />
            <span className="text-sm">Uploading and analyzing instrument files...</span>
          </div>
        )}

        {/* Results */}
        {hasAnalyzed && liveResults && (
          <>
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-400">
              <CheckCircle2 className="w-3.5 h-3.5" /> {liveResults.length} files analyzed
            </div>

            <div className="space-y-2">
              {liveResults.map((file) => {
                const isExpanded = expandedFile === file.id;
                const interp = file.interpretation;
                return (
                  <div key={file.id} className={`rounded-xl border transition-all ${
                    file.status === "anomaly" ? "border-red-500/30 bg-red-950/10" :
                    isExpanded ? "border-emerald-500/30 bg-zinc-900/80" :
                    "border-zinc-800 bg-zinc-900/40 hover:border-zinc-700"
                  }`}>
                    <div className="flex items-center gap-3 p-4 cursor-pointer" onClick={() => setExpandedFile(isExpanded ? null : file.id)}>
                      <FileText className={`w-5 h-5 shrink-0 ${
                        file.status === "anomaly" ? "text-red-400" :
                        file.status === "interpreted" ? "text-emerald-400" : "text-zinc-500"
                      }`} />
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-mono text-zinc-200 truncate">{file.filename}</div>
                        <div className="flex items-center gap-2 mt-1 text-[10px] text-zinc-500">
                          <span>Sample: {file.sampleId}</span>
                          <span>&bull;</span>
                          <span>{file.timestamp}</span>
                          <span>&bull;</span>
                          <span>{file.fileSize}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <InstrumentBadge instrument={file.instrument} />
                        <StatusBadge status={file.status} />
                        {isExpanded ? <ChevronUp className="w-4 h-4 text-zinc-500" /> : <ChevronDown className="w-4 h-4 text-zinc-500" />}
                      </div>
                    </div>

                    {isExpanded && interp && (
                      <div className="px-4 pb-4 space-y-3 animate-fade-in">
                        <div className={`p-3 rounded-lg border text-xs ${
                          file.status === "anomaly"
                            ? "bg-red-950/30 border-red-500/20 text-red-300"
                            : "bg-zinc-950 border-zinc-800 text-zinc-300"
                        }`}>
                          <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">AI Interpretation</div>
                          {interp.summary}
                        </div>

                        {file.status !== "anomaly" && (
                          <div className="grid grid-cols-3 gap-3">
                            <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 text-center">
                              <div className="text-[10px] text-zinc-500">Conversion</div>
                              <div className="text-lg font-bold text-emerald-400">{interp.conversion}%</div>
                            </div>
                            <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 text-center">
                              <div className="text-[10px] text-zinc-500">Purity</div>
                              <div className="text-lg font-bold text-emerald-400">{interp.purity}%</div>
                            </div>
                            <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 text-center">
                              <div className="text-[10px] text-zinc-500">Product</div>
                              <div className="text-lg font-bold text-emerald-400">{interp.majorProductConfirmed ? "Confirmed" : "Unconfirmed"}</div>
                            </div>
                          </div>
                        )}

                        {interp.impurities.length > 0 && (
                          <div className="bg-zinc-950 rounded-lg p-3 border border-zinc-800">
                            <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">Identified Impurities</div>
                            <div className="space-y-1.5">
                              {interp.impurities.map((imp, i) => (
                                <div key={i} className="flex items-center justify-between text-xs">
                                  <span className="text-zinc-300">{imp.identity}</span>
                                  <div className="flex items-center gap-2">
                                    <span className={imp.percentage > 2 ? "text-amber-400" : "text-zinc-400"}>{imp.percentage}%</span>
                                    {imp.flag && <span className="text-[10px] text-amber-400">{imp.flag}</span>}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {interp.anomalies.length > 0 && (
                          <div className="bg-red-950/30 rounded-lg p-3 border border-red-500/20">
                            <div className="text-[10px] text-red-400 uppercase tracking-wider mb-2">Anomalies Detected</div>
                            <div className="space-y-1.5">
                              {interp.anomalies.map((a, i) => (
                                <div key={i} className="flex items-start gap-2 text-xs text-red-300">
                                  <AlertTriangle className="w-3 h-3 mt-0.5 shrink-0" /> {a}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Upload more */}
            <div className="text-center">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
              >
                Upload more files
              </button>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".raw,.d,.mzML,.mzxml,.fid,.jdx,.dx,.csv,.txt,.spc,.spa"
                className="hidden"
                onChange={e => handleFileUpload(e.target.files)}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
