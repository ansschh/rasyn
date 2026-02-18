"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import {
  Atom, Play, RotateCcw, Settings2, SlidersHorizontal,
  ChevronRight, Sparkles, FlaskConical, Target,
  Activity, BookOpen, MessageSquare, FileText, Zap, ArrowRight,
  Search, TestTubes, Brain, BarChart3, Shield, Package, AlertCircle,
  Loader2, Send
} from "lucide-react";
import type { AppState, NoveltyMode, Objective, ProjectTab, Constraint } from "../types";
import RouteCard from "../components/RouteCard";
import StepDetail from "../components/StepDetail";
import SourceView from "../components/SourceView";
import DiscoverView from "../components/DiscoverView";
import ExecuteView from "../components/ExecuteView";
import AnalyzeView from "../components/AnalyzeView";
import LearnView from "../components/LearnView";
import AdminPanel from "../components/AdminPanel";
import { startPlan, chatWithCopilot } from "../lib/api";
import type { PlanResult, ApiRoute, EvidenceHit, CopilotMessage } from "../lib/api";
import { useJobStream } from "../hooks/useJobStream";

// ─── Inline example targets (legitimate quick-start examples) ────────────────

const EXAMPLE_TARGETS = [
  { name: "Osimertinib", smiles: "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C", mw: 499.6, category: "EGFR Inhibitor" },
  { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(C(C)C(=O)O)cc1", mw: 206.3, category: "NSAID" },
  { name: "Sildenafil", smiles: "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12", mw: 474.6, category: "PDE5 Inhibitor" },
];

const DEFAULT_CONSTRAINTS: Constraint[] = [
  { id: "no-pd", label: "Avoid palladium catalysts", type: "avoid_element", active: true },
  { id: "no-cryo", label: "No cryogenic conditions (< -20\u00B0C)", type: "avoid_condition", active: true },
  { id: "no-azide", label: "Avoid hazardous azides", type: "avoid_element", active: false },
  { id: "stock-prefer", label: "Prefer stocked building blocks", type: "prefer", active: true },
  { id: "min-pg", label: "Minimize protecting groups", type: "limit", active: false },
];

const PROJECT_TABS: { id: ProjectTab; icon: typeof Search; label: string }[] = [
  { id: "discover", icon: Search, label: "Discover" },
  { id: "plan", icon: FlaskConical, label: "Plan" },
  { id: "source", icon: Package, label: "Source" },
  { id: "execute", icon: TestTubes, label: "Execute" },
  { id: "analyze", icon: BarChart3, label: "Analyze" },
  { id: "learn", icon: Brain, label: "Learn" },
  { id: "admin", icon: Shield, label: "Admin" },
];

export default function Home() {
  // App state
  const [appState, setAppState] = useState<AppState>("idle");
  const [targetSmiles, setTargetSmiles] = useState("");
  const [targetName, setTargetName] = useState("");
  const [constraints, setConstraints] = useState<Constraint[]>(DEFAULT_CONSTRAINTS);
  const [noveltyMode, setNoveltyMode] = useState<NoveltyMode>("balanced");
  const [objective, setObjective] = useState<Objective>("default");
  const [selectedRouteIdx, setSelectedRouteIdx] = useState<number | null>(null);
  const [expandedStep, setExpandedStep] = useState<number | null>(null);
  const [rightTab, setRightTab] = useState<"log" | "evidence" | "copilot">("log");
  const [activeTab, setActiveTab] = useState<ProjectTab>("plan");
  const [promptText, setPromptText] = useState(
    "Design a scalable multi-step route to this target.\nConstraints: no Pd, no cryo < -20\u00B0C, avoid hazardous azides, prefer stocked building blocks, minimize protecting groups.\nGoal scale: 50 g."
  );

  // API state
  const [jobId, setJobId] = useState<string | null>(null);
  const [liveResult, setLiveResult] = useState<PlanResult | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const { events: sseEvents, status: streamStatus, result: streamResult, error: streamError } = useJobStream(jobId);

  // Copilot state
  const [copilotMessages, setCopilotMessages] = useState<CopilotMessage[]>([]);
  const [copilotInput, setCopilotInput] = useState("");
  const [copilotLoading, setCopilotLoading] = useState(false);
  const copilotEndRef = useRef<HTMLDivElement>(null);

  // When SSE stream completes, store the result
  useEffect(() => {
    if (streamResult) {
      setLiveResult(streamResult);
      setAppState("results");
      if (streamResult.routes.length > 0) {
        setSelectedRouteIdx(0);
      }
    }
  }, [streamResult]);

  // Handle stream errors
  useEffect(() => {
    if (streamError && streamStatus === "failed") {
      setApiError(streamError);
      setAppState("idle");
    }
  }, [streamError, streamStatus]);

  // Scroll copilot to bottom
  useEffect(() => {
    copilotEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [copilotMessages]);

  const routes: ApiRoute[] = liveResult?.routes || [];
  const selectedRoute = selectedRouteIdx !== null ? routes[selectedRouteIdx] : routes[0] || null;
  const evidence: EvidenceHit[] = liveResult?.evidence || [];

  const handlePlan = useCallback(async () => {
    if (!targetSmiles.trim()) {
      setApiError("Enter a SMILES string first.");
      return;
    }
    setAppState("planning");
    setSelectedRouteIdx(null);
    setExpandedStep(null);
    setRightTab("log");
    setActiveTab("plan");
    setApiError(null);
    setLiveResult(null);
    setCopilotMessages([]);

    try {
      // Build constraints dict from active UI toggles
      const activeConstraints: Record<string, boolean> = {};
      for (const c of constraints) {
        if (c.active) {
          // Map UI constraint IDs to backend keys
          const keyMap: Record<string, string> = {
            "no-pd": "no_pd",
            "no-cryo": "no_cryo",
            "no-azide": "no_azide",
            "stock-prefer": "stock_prefer",
            "min-pg": "min_pg",
          };
          const backendKey = keyMap[c.id] || c.id;
          activeConstraints[backendKey] = true;
        }
      }

      const resp = await startPlan({
        smiles: targetSmiles.trim(),
        top_k: 5,
        models: noveltyMode === "conservative" ? ["retro_v2"] : ["retro_v2", "llm"],
        constraints: Object.keys(activeConstraints).length > 0 ? activeConstraints : null,
        novelty_mode: noveltyMode,
        objective: objective,
      });
      setJobId(resp.job_id);
    } catch (err) {
      setApiError(err instanceof Error ? err.message : "Failed to start planning job");
      setAppState("idle");
    }
  }, [targetSmiles, constraints, noveltyMode, objective]);

  const handleReplan = useCallback(() => {
    handlePlan();
  }, [handlePlan]);

  const handleCopilotSend = useCallback(async () => {
    if (!copilotInput.trim() || copilotLoading) return;
    const msg = copilotInput.trim();
    setCopilotInput("");
    const userMsg: CopilotMessage = { role: "user", content: msg };
    setCopilotMessages(prev => [...prev, userMsg]);
    setCopilotLoading(true);
    try {
      const resp = await chatWithCopilot({
        message: msg,
        context: {
          smiles: targetSmiles || undefined,
          route: selectedRoute || null,
          constraints: constraints.filter(c => c.active).map(c => c.label),
          job_id: jobId || undefined,
        },
        history: copilotMessages,
      });
      setCopilotMessages(prev => [...prev, { role: "assistant", content: resp.reply }]);
    } catch (e) {
      setCopilotMessages(prev => [...prev, { role: "assistant", content: `Error: ${e instanceof Error ? e.message : "Failed to get response"}` }]);
    } finally {
      setCopilotLoading(false);
    }
  }, [copilotInput, copilotLoading, targetSmiles, selectedRoute, constraints, copilotMessages, jobId]);

  const selectExample = (example: typeof EXAMPLE_TARGETS[0]) => {
    setTargetSmiles(example.smiles);
    setTargetName(example.name);
  };

  const toggleConstraint = (id: string) => {
    setConstraints(prev => prev.map(c => c.id === id ? { ...c, active: !c.active } : c));
  };

  // Convert SSE events to log entries for display
  const sseLogEntries = sseEvents.map((evt) => ({
    message: evt.message || evt.kind,
    status: (evt.kind === "completed" ? "done" :
             evt.kind === "failed" ? "warning" :
             evt.kind === "warning" ? "warning" : "done") as "done" | "warning",
  }));

  // ─── IDLE STATE ────────────────────────────────────────────────
  if (appState === "idle") {
    return (
      <div className="min-h-screen flex flex-col">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4 border-b border-zinc-800/50">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-emerald-500/15 flex items-center justify-center">
              <Atom className="w-5 h-5 text-emerald-400" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">Rasyn</h1>
              <p className="text-[10px] text-zinc-500 -mt-0.5">Closed-Loop Chemistry OS</p>
            </div>
          </div>
          <div className="flex items-center gap-3 text-xs text-zinc-500">
            <div className="flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-emerald-500" />
              <span>RetroTransformer v2 + RSGPT-3.2B</span>
            </div>
          </div>
        </header>

        {/* Hero Section */}
        <div className="flex-1 flex items-center justify-center px-6">
          <div className="w-full max-w-3xl space-y-8">
            {/* Title */}
            <div className="text-center space-y-3">
              <h2 className="text-4xl font-bold tracking-tight">
                The operating system for{" "}
                <span className="bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                  synthesis
                </span>
              </h2>
              <p className="text-zinc-400 text-lg max-w-xl mx-auto">
                Discover &rarr; Plan &rarr; Execute &rarr; Analyze &rarr; Learn. The complete closed-loop pipeline from target to bench.
              </p>
            </div>

            {/* API Error */}
            {apiError && (
              <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-sm text-red-400">
                <AlertCircle className="w-4 h-4 shrink-0" /> {apiError}
              </div>
            )}

            {/* SMILES Input */}
            <div className="space-y-3">
              <label className="text-xs text-zinc-500 uppercase tracking-wider">Target Molecule (SMILES)</label>
              <input
                type="text"
                value={targetSmiles}
                onChange={e => { setTargetSmiles(e.target.value); setTargetName(""); }}
                placeholder="Enter SMILES string, e.g. CC(=O)Oc1ccccc1C(=O)O"
                className="w-full px-4 py-3 rounded-xl bg-zinc-900 border border-zinc-800 text-sm text-zinc-200 font-mono focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 placeholder-zinc-600"
              />
              {/* Example targets */}
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-[10px] text-zinc-600">Examples:</span>
                {EXAMPLE_TARGETS.map(t => (
                  <button
                    key={t.name}
                    onClick={() => selectExample(t)}
                    className={`text-[11px] px-2.5 py-1 rounded-lg border transition-colors ${
                      targetSmiles === t.smiles
                        ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-400"
                        : "border-zinc-800 text-zinc-400 hover:border-zinc-700 hover:text-zinc-300"
                    }`}
                  >
                    {t.name} <span className="text-zinc-600">({t.mw})</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Prompt Box */}
            <div className="space-y-3">
              <label className="text-xs text-zinc-500 uppercase tracking-wider">Planning Instructions</label>
              <textarea
                value={promptText}
                onChange={e => setPromptText(e.target.value)}
                rows={4}
                className="w-full px-4 py-3 rounded-xl bg-zinc-900 border border-zinc-800 text-sm text-zinc-200 resize-none focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 placeholder-zinc-600"
                placeholder="Describe your synthesis goals and constraints..."
              />
            </div>

            {/* Plan Button */}
            <button
              onClick={handlePlan}
              disabled={!targetSmiles.trim()}
              className="w-full py-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-700 disabled:cursor-not-allowed text-white font-semibold text-lg flex items-center justify-center gap-3 transition-all hover:shadow-lg hover:shadow-emerald-500/20 pulse-glow"
            >
              <Play className="w-5 h-5" />
              Plan Synthesis
            </button>

            {/* Pipeline steps */}
            <div className="grid grid-cols-5 gap-3 pt-4">
              {[
                { icon: Search, label: "Discover", desc: "Search + cluster precedents" },
                { icon: FlaskConical, label: "Plan", desc: "Multi-engine routes" },
                { icon: TestTubes, label: "Execute", desc: "Protocols + ELN" },
                { icon: BarChart3, label: "Analyze", desc: "Auto-interpret LCMS/NMR" },
                { icon: Brain, label: "Learn", desc: "Institutional memory" },
              ].map(({ icon: Icon, label, desc }, i) => (
                <div key={label} className="text-center space-y-2 relative">
                  <Icon className="w-5 h-5 text-emerald-500 mx-auto" />
                  <div className="text-xs font-medium text-zinc-300">{label}</div>
                  <div className="text-[10px] text-zinc-500">{desc}</div>
                  {i < 4 && (
                    <ArrowRight className="w-3 h-3 text-zinc-700 absolute -right-2 top-1" />
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ─── PLANNING + RESULTS STATE ──────────────────────────────────
  const displayName = targetName || (targetSmiles.length > 30 ? targetSmiles.slice(0, 30) + "..." : targetSmiles);

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-4 py-2 border-b border-zinc-800/50 shrink-0">
        <div className="flex items-center gap-3">
          <button onClick={() => setAppState("idle")} className="flex items-center gap-2 text-zinc-400 hover:text-zinc-200 transition-colors">
            <Atom className="w-5 h-5 text-emerald-400" />
            <span className="text-sm font-bold">Rasyn</span>
          </button>
          <ChevronRight className="w-4 h-4 text-zinc-600" />
          <span className="text-sm text-zinc-300">{displayName}</span>
          {appState === "results" && routes.length > 0 && (
            <>
              <ChevronRight className="w-4 h-4 text-zinc-600" />
              <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400">{routes.length} routes</span>
            </>
          )}
        </div>
        <div className="flex items-center gap-2">
          {appState === "results" && (
            <button
              onClick={handleReplan}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border border-zinc-700 text-zinc-300 hover:bg-zinc-800 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Replan
            </button>
          )}
        </div>
      </header>

      {/* Tab Navigation */}
      {appState === "results" && (
        <div className="flex items-center border-b border-zinc-800/50 px-4 shrink-0 bg-zinc-950/50">
          {PROJECT_TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors ${
                activeTab === tab.id
                  ? "border-emerald-500 text-emerald-400"
                  : "border-transparent text-zinc-500 hover:text-zinc-300"
              }`}
            >
              <tab.icon className="w-3.5 h-3.5" />
              {tab.label}
            </button>
          ))}
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Planning animation (full width) */}
        {appState === "planning" && (
          <div className="flex-1 overflow-y-auto p-6">
            <div className="max-w-2xl mx-auto">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-xl bg-emerald-500/15 flex items-center justify-center animate-pulse">
                  <Zap className="w-5 h-5 text-emerald-400" />
                </div>
                <div>
                  <h2 className="text-lg font-semibold">Planning synthesis route...</h2>
                  <p className="text-xs text-zinc-500">Analyzing {displayName}</p>
                </div>
              </div>

              {apiError && (
                <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-xs text-red-400">
                  <AlertCircle className="w-4 h-4 inline mr-2" />{apiError}
                </div>
              )}

              <div className="space-y-1 max-h-[500px] overflow-y-auto p-3 bg-[#0d0d14] rounded-lg border border-zinc-800/50 font-mono text-xs">
                {sseLogEntries.length > 0 ? (
                  sseLogEntries.map((entry, i) => (
                    <div key={i} className="flex items-start gap-2 py-1.5 px-2 rounded animate-fade-in">
                      <span className={`${entry.status === "warning" ? "text-amber-300" : "text-emerald-300"}`}>
                        {entry.message}
                      </span>
                    </div>
                  ))
                ) : (
                  <div className="flex items-center gap-2 py-1.5 px-2 text-zinc-500">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    <span>Waiting for worker to pick up job...</span>
                  </div>
                )}
                {streamStatus === "streaming" && (
                  <div className="flex items-center gap-2 py-1.5 px-2 text-zinc-500">
                    <Zap className="w-3 h-3 animate-pulse" />
                    <span className="animate-pulse">Processing...</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Results: tab-dependent views */}
        {appState === "results" && activeTab === "discover" && (
          <DiscoverView targetName={displayName} smiles={targetSmiles} liveData={liveResult?.discovery ?? null} />
        )}

        {appState === "results" && activeTab === "execute" && (
          <ExecuteView
            jobId={jobId}
            route={selectedRoute as any}
          />
        )}

        {appState === "results" && activeTab === "analyze" && (
          <AnalyzeView
            productSmiles={targetSmiles}
            expectedMw={liveResult?.safety?.druglikeness?.mw}
          />
        )}

        {appState === "results" && activeTab === "learn" && (
          <LearnView jobId={jobId} targetSmiles={targetSmiles} selectedRouteIdx={selectedRouteIdx ?? 0} />
        )}

        {appState === "results" && activeTab === "admin" && (
          <AdminPanel />
        )}

        {appState === "results" && activeTab === "source" && (
          <SourceView liveSourcing={liveResult?.sourcing ?? null} />
        )}

        {/* Plan tab: 3-pane layout */}
        {appState === "results" && activeTab === "plan" && (
          <>
            {/* LEFT PANEL */}
            <div className="w-72 border-r border-zinc-800/50 overflow-y-auto shrink-0 bg-zinc-950/30">
              <div className="p-4 space-y-5">
                {/* Target */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <Target className="w-3.5 h-3.5 text-emerald-500" />
                    <span className="text-[10px] text-zinc-500 uppercase tracking-wider">Target</span>
                  </div>
                  <div className="p-3 rounded-lg bg-zinc-900 border border-zinc-800">
                    <div className="text-sm font-medium text-zinc-200">{displayName}</div>
                    <div className="smiles-display text-[10px] text-zinc-500 mt-1 break-all">{targetSmiles}</div>
                  </div>
                </div>

                {/* Constraints */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <Settings2 className="w-3.5 h-3.5 text-blue-500" />
                    <span className="text-[10px] text-zinc-500 uppercase tracking-wider">Constraints</span>
                  </div>
                  <div className="space-y-1.5">
                    {constraints.map(c => (
                      <label key={c.id} className="flex items-center gap-2 p-2 rounded-lg hover:bg-zinc-900 cursor-pointer transition-colors">
                        <input
                          type="checkbox"
                          checked={c.active}
                          onChange={() => toggleConstraint(c.id)}
                          className="w-3.5 h-3.5 rounded border-zinc-600 text-emerald-500 focus:ring-emerald-500/20 bg-zinc-800"
                        />
                        <span className={`text-xs ${c.active ? "text-zinc-300" : "text-zinc-500"}`}>{c.label}</span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Novelty Slider */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <Sparkles className="w-3.5 h-3.5 text-purple-500" />
                    <span className="text-[10px] text-zinc-500 uppercase tracking-wider">Novelty Level</span>
                  </div>
                  <div className="space-y-2">
                    {(["conservative", "balanced", "exploratory"] as NoveltyMode[]).map(mode => (
                      <button
                        key={mode}
                        onClick={() => setNoveltyMode(mode)}
                        className={`w-full flex items-center gap-3 p-2.5 rounded-lg border text-left text-xs transition-all ${
                          noveltyMode === mode
                            ? mode === "conservative" ? "border-emerald-500/40 bg-emerald-500/5 text-emerald-300" :
                              mode === "balanced" ? "border-blue-500/40 bg-blue-500/5 text-blue-300" :
                              "border-purple-500/40 bg-purple-500/5 text-purple-300"
                            : "border-zinc-800 text-zinc-400 hover:border-zinc-700"
                        }`}
                      >
                        <div className={`w-2 h-2 rounded-full ${
                          mode === "conservative" ? "bg-emerald-500" :
                          mode === "balanced" ? "bg-blue-500" :
                          "bg-purple-500"
                        }`} />
                        <div>
                          <div className="font-medium capitalize">{mode}</div>
                          <div className="text-[10px] text-zinc-500">
                            {mode === "conservative" ? "Only well-precedented reactions" :
                             mode === "balanced" ? "Mix of known and novel steps" :
                             "Include AI-discovered novel chemistry"}
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Objective */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <SlidersHorizontal className="w-3.5 h-3.5 text-amber-500" />
                    <span className="text-[10px] text-zinc-500 uppercase tracking-wider">Objective</span>
                  </div>
                  <div className="grid grid-cols-2 gap-1.5">
                    {(["default", "fastest", "cheapest", "safest", "greenest"] as Objective[]).map(obj => (
                      <button
                        key={obj}
                        onClick={() => setObjective(obj)}
                        className={`px-3 py-2 text-xs rounded-lg border transition-colors ${
                          objective === obj
                            ? "border-amber-500/50 bg-amber-500/10 text-amber-400"
                            : "border-zinc-800 text-zinc-400 hover:border-zinc-700 hover:text-zinc-300"
                        }`}
                      >
                        {obj === "default" ? "Default" : obj.charAt(0).toUpperCase() + obj.slice(1)}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* CENTER PANEL */}
            <div className="flex-1 overflow-y-auto bg-[var(--bg-primary)]">
              <div className="p-4 space-y-4">
                {routes.length === 0 ? (
                  <div className="flex items-center justify-center py-16 text-zinc-500 text-sm">
                    No routes found. The retrosynthesis models did not find viable routes for this target.
                  </div>
                ) : (
                  <>
                    {/* Route Cards */}
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <h3 className="text-sm font-semibold text-zinc-300">Route Leaderboard</h3>
                        <span className="text-[10px] text-zinc-500">{routes.length} routes &bull; ranked by combined score</span>
                      </div>
                      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3">
                        {routes.map((route, idx) => (
                          <RouteCard
                            key={route.route_id}
                            route={route}
                            selected={selectedRoute?.route_id === route.route_id}
                            onSelect={() => { setSelectedRouteIdx(idx); setExpandedStep(null); }}
                          />
                        ))}
                      </div>
                    </div>

                    {/* Selected Route Steps */}
                    {selectedRoute && (
                      <div className="space-y-3 mt-6">
                        <div className="flex items-center gap-2">
                          <h3 className="text-sm font-semibold text-zinc-300">Route #{selectedRoute.rank} &mdash; Step Details</h3>
                          <span className="text-[10px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-400">{selectedRoute.steps.length} steps</span>
                        </div>
                        <div className="space-y-2">
                          {selectedRoute.steps.map((step, i) => (
                            <StepDetail
                              key={i}
                              step={step}
                              stepNumber={i + 1}
                              isExpanded={expandedStep === i}
                              onToggle={() => setExpandedStep(expandedStep === i ? null : i)}
                            />
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>

            {/* RIGHT PANEL */}
            <div className="w-80 border-l border-zinc-800/50 flex flex-col shrink-0 bg-zinc-950/30">
              {/* Tabs */}
              <div className="flex border-b border-zinc-800/50 shrink-0 bg-zinc-950/90 backdrop-blur-sm z-10">
                {[
                  { id: "log" as const, icon: Activity, label: "Action Log" },
                  { id: "evidence" as const, icon: BookOpen, label: "Evidence" },
                  { id: "copilot" as const, icon: MessageSquare, label: "Copilot" },
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setRightTab(tab.id)}
                    className={`flex-1 flex items-center justify-center gap-1.5 py-3 text-xs font-medium transition-colors border-b-2 ${
                      rightTab === tab.id
                        ? "border-emerald-500 text-emerald-400"
                        : "border-transparent text-zinc-500 hover:text-zinc-300"
                    }`}
                  >
                    <tab.icon className="w-3.5 h-3.5" />
                    {tab.label}
                  </button>
                ))}
              </div>

              <div className="flex-1 overflow-y-auto p-4">
                {/* Action Log Tab */}
                {rightTab === "log" && (
                  <div className="space-y-4">
                    {sseLogEntries.length > 0 ? (
                      <div className="space-y-1 p-3 bg-[#0d0d14] rounded-lg border border-zinc-800/50 font-mono text-xs">
                        {sseLogEntries.map((entry, i) => (
                          <div key={i} className="flex items-start gap-2 py-1 px-1">
                            <span className={`${entry.status === "warning" ? "text-amber-300" : "text-zinc-300"}`}>
                              {entry.message}
                            </span>
                          </div>
                        ))}
                        {liveResult && (
                          <div className="mt-2 pt-2 border-t border-zinc-800 text-emerald-400">
                            Completed in {liveResult.compute_time_ms?.toFixed(0) ?? "\u2014"}ms
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-center py-8 text-zinc-500 text-sm">
                        No action log entries.
                      </div>
                    )}
                  </div>
                )}

                {/* Evidence Tab */}
                {rightTab === "evidence" && (
                  <div className="space-y-4">
                    {evidence.length > 0 ? (() => {
                      const localHits = evidence.filter(h => h.similarity > 0).sort((a, b) => b.similarity - a.similarity);
                      const liveHits = evidence.filter(h => h.similarity === 0);
                      return (
                        <>
                          {/* Local reaction matches */}
                          {localHits.length > 0 && (
                            <div>
                              <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">
                                Reaction Fingerprint Matches ({localHits.length})
                              </div>
                              <div className="space-y-2">
                                {localHits.map((hit, i) => (
                                  <div key={`local-${i}`} className="p-3 rounded-lg border border-zinc-800 hover:border-zinc-700 transition-colors">
                                    <div className="flex items-start gap-2">
                                      <FlaskConical className="w-4 h-4 shrink-0 mt-0.5 text-purple-400" />
                                      <div className="min-w-0 flex-1">
                                        {hit.doi ? (
                                          <a href={hit.doi} target="_blank" rel="noopener noreferrer"
                                            className="text-xs text-purple-400 hover:text-purple-300 underline underline-offset-2">
                                            {hit.title || hit.source}
                                          </a>
                                        ) : (
                                          <div className="text-xs text-zinc-300">{hit.title || hit.source}</div>
                                        )}
                                        <div className="flex items-center gap-1.5 mt-1 text-[10px] text-zinc-500">
                                          <span>{hit.source}</span>
                                          {hit.year && <><span>&bull;</span><span>{hit.year}</span></>}
                                        </div>
                                        {hit.rxn_smiles && (
                                          <div className="smiles-display text-[10px] text-zinc-600 mt-1 truncate">{hit.rxn_smiles}</div>
                                        )}
                                      </div>
                                      <span className={`text-xs font-medium shrink-0 ${
                                        hit.similarity >= 0.9 ? "text-emerald-400" : hit.similarity >= 0.7 ? "text-amber-400" : "text-zinc-400"
                                      }`}>
                                        {Math.round(hit.similarity * 100)}%
                                      </span>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Live paper/patent hits */}
                          {liveHits.length > 0 && (
                            <div>
                              <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">
                                Literature &amp; Patents ({liveHits.length})
                              </div>
                              <div className="space-y-2">
                                {liveHits.map((hit, i) => (
                                  <div key={`live-${i}`} className="p-3 rounded-lg border border-zinc-800 hover:border-zinc-700 transition-colors">
                                    <div className="flex items-start gap-2">
                                      <BookOpen className="w-4 h-4 shrink-0 mt-0.5 text-blue-400" />
                                      <div className="min-w-0 flex-1">
                                        {hit.doi ? (
                                          <a href={hit.doi} target="_blank" rel="noopener noreferrer"
                                            className="text-xs text-blue-400 hover:text-blue-300 underline underline-offset-2">
                                            {hit.title || "View paper"}
                                          </a>
                                        ) : (
                                          <div className="text-xs text-zinc-300">{hit.title || "Untitled"}</div>
                                        )}
                                        <div className="flex items-center gap-1.5 mt-1 text-[10px] text-zinc-500">
                                          <span>{hit.source}</span>
                                          {hit.year && <><span>&bull;</span><span>{hit.year}</span></>}
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </>
                      );
                    })() : (
                      <div className="text-center py-8 text-zinc-500 text-sm">
                        No evidence found. The evidence search module returned no matching precedents.
                      </div>
                    )}
                  </div>
                )}

                {/* Copilot Tab */}
                {rightTab === "copilot" && (
                  <div className="flex flex-col h-full">
                    <div className="p-3 rounded-lg bg-zinc-900 border border-zinc-800 text-xs text-zinc-400 mb-3 shrink-0">
                      <div className="flex items-center gap-2 mb-2 text-emerald-400 font-medium">
                        <Sparkles className="w-3.5 h-3.5" /> Rasyn Copilot
                      </div>
                      Ask questions about the planned route, request alternatives, or explore specific steps.
                    </div>

                    {/* Chat messages */}
                    <div className="flex-1 overflow-y-auto space-y-3 mb-3">
                      {copilotMessages.length === 0 && (
                        <div className="text-center py-8 text-zinc-600 text-xs">
                          Start a conversation about your synthesis route.
                        </div>
                      )}
                      {copilotMessages.map((msg, i) => (
                        <div key={i} className="flex gap-2">
                          <div className={`w-5 h-5 rounded-full flex items-center justify-center shrink-0 mt-0.5 ${
                            msg.role === "user" ? "bg-blue-500/20" : "bg-emerald-500/20"
                          }`}>
                            {msg.role === "user" ? (
                              <span className="text-[9px] text-blue-400 font-bold">U</span>
                            ) : (
                              <Atom className="w-3 h-3 text-emerald-400" />
                            )}
                          </div>
                          <p className={`text-xs whitespace-pre-wrap ${msg.role === "user" ? "text-zinc-300" : "text-zinc-400"}`}>{msg.content}</p>
                        </div>
                      ))}
                      {copilotLoading && (
                        <div className="flex items-center gap-2 text-xs text-zinc-500">
                          <Loader2 className="w-3 h-3 animate-spin" /> Thinking...
                        </div>
                      )}
                      <div ref={copilotEndRef} />
                    </div>

                    {/* Input */}
                    <div className="shrink-0 pt-2 border-t border-zinc-800">
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={copilotInput}
                          onChange={e => setCopilotInput(e.target.value)}
                          onKeyDown={e => e.key === "Enter" && handleCopilotSend()}
                          placeholder="Ask about the route..."
                          disabled={copilotLoading}
                          className="flex-1 px-3 py-2 rounded-lg bg-zinc-900 border border-zinc-800 text-xs text-zinc-200 focus:outline-none focus:border-emerald-500/50 placeholder-zinc-600 disabled:opacity-50"
                        />
                        <button
                          onClick={handleCopilotSend}
                          disabled={copilotLoading || !copilotInput.trim()}
                          className="px-3 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:bg-zinc-700 disabled:cursor-not-allowed text-white transition-colors"
                        >
                          <Send className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
