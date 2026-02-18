"use client";

import { useState, useEffect } from "react";
import {
  Brain, CheckCircle2, XCircle, AlertTriangle, ArrowRight, TrendingUp,
  ChevronDown, ChevronUp, BookOpen, Shield, Zap, Loader2, AlertCircle
} from "lucide-react";
import type {
  InsightsResponse, RankingExplanation, LearnStatsResponse,
} from "../lib/api";

interface Props {
  jobId?: string | null;
  targetSmiles?: string | null;
  selectedRouteIdx?: number;
}

function OutcomeBadge({ outcome }: { outcome: string }) {
  const config: Record<string, { icon: typeof CheckCircle2; cls: string; label: string }> = {
    success: { icon: CheckCircle2, cls: "text-emerald-400 bg-emerald-500/15 border-emerald-500/30", label: "Success" },
    failure: { icon: XCircle, cls: "text-red-400 bg-red-500/15 border-red-500/30", label: "Failure" },
    partial: { icon: AlertTriangle, cls: "text-amber-400 bg-amber-500/15 border-amber-500/30", label: "Partial" },
  };
  const { icon: Icon, cls, label } = config[outcome] || config.partial;
  return (
    <span className={`inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full border ${cls}`}>
      <Icon className="w-3 h-3" /> {label}
    </span>
  );
}

function InsightTypeBadge({ type }: { type: string }) {
  const colors: Record<string, string> = {
    failure_avoidance: "text-red-400 bg-red-500/15 border-red-500/30",
    optimization: "text-blue-400 bg-blue-500/15 border-blue-500/30",
    preference: "text-emerald-400 bg-emerald-500/15 border-emerald-500/30",
  };
  const labels: Record<string, string> = {
    failure_avoidance: "Failure Avoidance",
    optimization: "Optimization",
    preference: "Preference",
  };
  return (
    <span className={`text-[10px] px-2 py-0.5 rounded-full border ${colors[type] || ""}`}>
      {labels[type] || type}
    </span>
  );
}

export default function LearnView({ jobId, targetSmiles, selectedRouteIdx = 0 }: Props) {
  const [activeSection, setActiveSection] = useState<"ranking" | "memory" | "experiments">("ranking");
  const [expandedExp, setExpandedExp] = useState<string | null>(null);

  const [liveInsights, setLiveInsights] = useState<InsightsResponse | null>(null);
  const [liveRanking, setLiveRanking] = useState<RankingExplanation | null>(null);
  const [liveStats, setLiveStats] = useState<LearnStatsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function fetchLiveData() {
      setIsLoading(true);
      setError(null);
      try {
        const { getInsights, getRankingExplanation, getPastExperiments } = await import("../lib/api");

        const [insightsRes, statsRes] = await Promise.all([
          getInsights(targetSmiles),
          getPastExperiments(50),
        ]);

        if (cancelled) return;

        setLiveInsights(insightsRes);
        setLiveStats(statsRes);

        if (jobId) {
          const rankingRes = await getRankingExplanation(jobId, selectedRouteIdx).catch(() => null);
          if (!cancelled && rankingRes) {
            setLiveRanking(rankingRes);
          }
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load learning data");
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }
    fetchLiveData();
    return () => { cancelled = true; };
  }, [jobId, targetSmiles]);

  const ranking = liveRanking;
  const insights = liveInsights?.insights || [];
  const experiments = liveStats?.past_experiments || [];

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-400" /> Institutional Memory
          </h2>
          <p className="text-xs text-zinc-500">Learn from every experiment. Prevent repeated failures. Compound knowledge over time.</p>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-xs text-red-400">
            <AlertCircle className="w-4 h-4 shrink-0" /> {error}
          </div>
        )}

        {/* Loading */}
        {isLoading && (
          <div className="flex items-center justify-center py-8 gap-2 text-zinc-400 text-xs">
            <Loader2 className="w-4 h-4 animate-spin" /> Loading institutional memory...
          </div>
        )}

        {!isLoading && !error && (
          <>
            {/* Section tabs */}
            <div className="flex border-b border-zinc-800">
              {([
                { id: "ranking" as const, icon: Shield, label: "Route Reasoning" },
                { id: "memory" as const, icon: Brain, label: "Active Insights" },
                { id: "experiments" as const, icon: BookOpen, label: "Past Experiments" },
              ]).map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveSection(tab.id)}
                  className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors ${
                    activeSection === tab.id
                      ? "border-purple-500 text-purple-400"
                      : "border-transparent text-zinc-500 hover:text-zinc-300"
                  }`}
                >
                  <tab.icon className="w-3.5 h-3.5" /> {tab.label}
                </button>
              ))}
            </div>

            {/* Route Ranking Explanation */}
            {activeSection === "ranking" && (
              <div className="space-y-4">
                {ranking ? (
                  <div className="p-4 rounded-xl bg-zinc-900 border border-zinc-800">
                    <h3 className="text-sm font-semibold text-zinc-200 mb-4">{ranking.question}</h3>
                    <div className="space-y-3">
                      {ranking.factors.map((f, i) => (
                        <div key={i} className="flex items-start gap-3">
                          <div className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 mt-0.5 ${
                            f.impact === "positive" ? "bg-emerald-500/20 text-emerald-400" :
                            f.impact === "neutral" ? "bg-zinc-500/20 text-zinc-400" :
                            "bg-red-500/20 text-red-400"
                          }`}>
                            {f.impact === "positive" ? <CheckCircle2 className="w-3.5 h-3.5" /> :
                             f.impact === "neutral" ? <ArrowRight className="w-3.5 h-3.5" /> :
                             <XCircle className="w-3.5 h-3.5" />}
                          </div>
                          <div>
                            <div className="text-xs font-medium text-zinc-200">{f.factor}</div>
                            <div className="text-xs text-zinc-400 mt-0.5">{f.detail}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8 text-zinc-500 text-sm">
                    {jobId ? "No ranking explanation available for this job." : "Run a synthesis plan to see route ranking reasoning."}
                  </div>
                )}
              </div>
            )}

            {/* Active memory insights */}
            {activeSection === "memory" && (
              <div className="space-y-3">
                {insights.length > 0 ? (
                  <>
                    <div className="flex items-center gap-6 px-4 py-2.5 rounded-lg bg-zinc-900/50 border border-zinc-800 text-xs">
                      <div><span className="text-zinc-500">Active insights:</span> <span className="text-purple-400 font-semibold">{insights.length}</span></div>
                      <div><span className="text-zinc-500">Total experiments indexed:</span> <span className="text-zinc-300 font-semibold">{liveStats?.total_experiments ?? 0}</span></div>
                      <div><span className="text-zinc-500">Times insights applied:</span> <span className="text-emerald-400 font-semibold">{liveInsights?.total_applications ?? 0}</span></div>
                    </div>
                    {insights.map((insight) => (
                      <div key={insight.id} className="p-4 rounded-xl bg-zinc-900 border border-zinc-800 hover:border-zinc-700 transition-colors">
                        <div className="flex items-start gap-3">
                          <Zap className={`w-4 h-4 shrink-0 mt-0.5 ${
                            insight.type === "failure_avoidance" ? "text-red-400" :
                            insight.type === "optimization" ? "text-blue-400" : "text-emerald-400"
                          }`} />
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <InsightTypeBadge type={insight.type} />
                              <span className="text-[10px] text-zinc-500">Confidence: {Math.round(insight.confidence * 100)}%</span>
                              <span className="text-[10px] text-zinc-500">&bull; Applied {insight.timesApplied}x</span>
                            </div>
                            <div className="text-xs text-zinc-200 font-medium">{insight.rule}</div>
                            <div className="text-[10px] text-zinc-500 mt-1">Source: {insight.source}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </>
                ) : (
                  <div className="text-center py-8 text-zinc-500 text-sm">
                    No insights yet. Record experiment outcomes to build institutional memory.
                  </div>
                )}
              </div>
            )}

            {/* Past experiments */}
            {activeSection === "experiments" && (
              <div className="space-y-2">
                {experiments.length > 0 ? (
                  <>
                    {liveStats && (
                      <div className="flex items-center gap-6 px-4 py-2.5 rounded-lg bg-zinc-900/50 border border-zinc-800 text-xs">
                        <div><span className="text-zinc-500">Success rate:</span> <span className="text-emerald-400 font-semibold">{(liveStats.success_rate * 100).toFixed(0)}%</span></div>
                        {liveStats.avg_yield != null && (
                          <div><span className="text-zinc-500">Avg yield:</span> <span className="text-zinc-300 font-semibold">{liveStats.avg_yield.toFixed(0)}%</span></div>
                        )}
                        <div><span className="text-zinc-500">Reactions recorded:</span> <span className="text-zinc-300 font-semibold">{liveStats.total_reactions}</span></div>
                      </div>
                    )}
                    {experiments.map((exp) => {
                      const isExpanded = expandedExp === exp.id;
                      return (
                        <div key={exp.id} className={`rounded-xl border transition-all ${
                          exp.outcome === "failure" ? "border-red-500/20 bg-red-950/5" :
                          isExpanded ? "border-purple-500/30 bg-zinc-900/80" :
                          "border-zinc-800 bg-zinc-900/40 hover:border-zinc-700"
                        }`}>
                          <div className="flex items-center gap-3 p-4 cursor-pointer" onClick={() => setExpandedExp(isExpanded ? null : exp.id)}>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2">
                                <span className="text-xs font-mono text-zinc-400">{exp.id}</span>
                                <OutcomeBadge outcome={exp.outcome} />
                              </div>
                              <div className="text-xs text-zinc-200 mt-1">{exp.reaction}</div>
                              <div className="text-[10px] text-zinc-500 mt-0.5">{exp.target} &bull; {exp.date}</div>
                            </div>
                            <div className="flex items-center gap-2 shrink-0">
                              {exp.yield && <span className="text-xs text-zinc-300">{exp.yield}</span>}
                              {isExpanded ? <ChevronUp className="w-4 h-4 text-zinc-500" /> : <ChevronDown className="w-4 h-4 text-zinc-500" />}
                            </div>
                          </div>

                          {isExpanded && (
                            <div className="px-4 pb-4 space-y-3 animate-fade-in">
                              <div className="bg-zinc-950 rounded-lg p-3 border border-zinc-800">
                                <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">Conditions</div>
                                <div className="text-xs text-zinc-300">{exp.conditions}</div>
                              </div>
                              <div className={`rounded-lg p-3 border ${
                                exp.outcome === "failure" ? "bg-red-950/20 border-red-500/20" : "bg-zinc-950 border-zinc-800"
                              }`}>
                                <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1">Notes</div>
                                <div className="text-xs text-zinc-300">{exp.notes}</div>
                              </div>
                              <div className="rounded-lg p-3 border border-purple-500/20 bg-purple-950/10">
                                <div className="text-[10px] text-purple-400 uppercase tracking-wider mb-1">Impact on Planning</div>
                                <div className="text-xs text-purple-300">{exp.impactOnPlanning}</div>
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </>
                ) : (
                  <div className="text-center py-8 text-zinc-500 text-sm">
                    No experiments recorded yet. Record outcomes via the Learn API to build institutional memory.
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
