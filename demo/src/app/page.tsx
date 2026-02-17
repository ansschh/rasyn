"use client";

import { useState, useCallback, useEffect } from "react";
import {
  Atom, Play, RotateCcw, ShoppingCart, Settings2, SlidersHorizontal,
  ChevronRight, Sparkles, FlaskConical, Target, Layers, Activity,
  BookOpen, MessageSquare, FileText, Zap, ArrowRight,
  Search, TestTubes, Brain, BarChart3, Shield, Package, Wifi, WifiOff
} from "lucide-react";
import type { AppState, NoveltyMode, ProjectTab, DemoRoute, Constraint } from "../types";
import {
  DEMO_TARGETS, ALL_ROUTES, DEFAULT_CONSTRAINTS,
  PLANNING_LOG_ENTRIES, REPLAN_LOG_ENTRIES, SHOPPING_LIST_OSIM,
} from "../data/mock-data";
import ActionLog from "../components/ActionLog";
import RouteCard from "../components/RouteCard";
import StepDetail from "../components/StepDetail";
import ShoppingList from "../components/ShoppingList";
import SourceView from "../components/SourceView";
import DiscoverView from "../components/DiscoverView";
import ExecuteView from "../components/ExecuteView";
import AnalyzeView from "../components/AnalyzeView";
import LearnView from "../components/LearnView";
import AdminPanel from "../components/AdminPanel";
import { startPlan, checkApiHealth, fetchResult } from "../lib/api";
import type { PlanResult } from "../lib/api";
import { useJobStream } from "../hooks/useJobStream";
import type { SSEEvent } from "../hooks/useJobStream";

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
  const [selectedTargetId, setSelectedTargetId] = useState("osimertinib");
  const [constraints, setConstraints] = useState<Constraint[]>(DEFAULT_CONSTRAINTS);
  const [noveltyMode, setNoveltyMode] = useState<NoveltyMode>("balanced");
  const [selectedRouteId, setSelectedRouteId] = useState<string | null>(null);
  const [expandedStep, setExpandedStep] = useState<number | null>(null);
  const [showShopping, setShowShopping] = useState(false);
  const [isReplanning, setIsReplanning] = useState(false);
  const [rightTab, setRightTab] = useState<"log" | "evidence" | "copilot">("log");
  const [activeTab, setActiveTab] = useState<ProjectTab>("plan");
  const [promptText, setPromptText] = useState(
    "Design a scalable multi-step route to this target.\nConstraints: no Pd, no cryo < -20\u00B0C, avoid hazardous azides, prefer stocked building blocks, minimize protecting groups.\nGoal scale: 50 g."
  );

  // Live API mode
  const [apiAvailable, setApiAvailable] = useState(false);
  const [liveMode, setLiveMode] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [liveResult, setLiveResult] = useState<PlanResult | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const { events: sseEvents, status: streamStatus, result: streamResult, error: streamError } = useJobStream(jobId);

  // Check API health on mount
  useEffect(() => {
    checkApiHealth().then(setApiAvailable);
  }, []);

  // When SSE stream completes, store the result
  useEffect(() => {
    if (streamResult) {
      setLiveResult(streamResult);
      setAppState("results");
    }
  }, [streamResult]);

  // Handle stream errors
  useEffect(() => {
    if (streamError && streamStatus === "failed") {
      setApiError(streamError);
      // Fall back to demo mode
      setLiveMode(false);
      setJobId(null);
    }
  }, [streamError, streamStatus]);

  const selectedTarget = DEMO_TARGETS.find(t => t.id === selectedTargetId)!;
  const routes = ALL_ROUTES[selectedTargetId] || [];
  const filteredRoutes = routes.filter(r =>
    noveltyMode === "conservative" ? r.noveltyLevel === "conservative" :
    noveltyMode === "exploratory" ? true :
    r.noveltyLevel !== "exploratory"
  );
  const selectedRoute = filteredRoutes.find(r => r.id === selectedRouteId) || filteredRoutes[0];

  const handlePlan = useCallback(async () => {
    setAppState("planning");
    setSelectedRouteId(null);
    setExpandedStep(null);
    setRightTab("log");
    setActiveTab("plan");
    setApiError(null);

    // Try live API if available
    if (apiAvailable) {
      try {
        const resp = await startPlan({
          smiles: selectedTarget.smiles,
          top_k: 5,
          models: ["retro_v2", "llm"],
        });
        setJobId(resp.job_id);
        setLiveMode(true);
        return; // SSE events will drive the UI
      } catch (err) {
        console.warn("API call failed, falling back to demo:", err);
        setApiError(err instanceof Error ? err.message : "API error");
      }
    }

    // Fallback: demo mode (mock data)
    setLiveMode(false);
    setJobId(null);
  }, [apiAvailable, selectedTarget]);

  const handlePlanComplete = useCallback(() => {
    setAppState("results");
    if (filteredRoutes.length > 0) {
      setSelectedRouteId(filteredRoutes[0].id);
    }
  }, [filteredRoutes]);

  // Convert SSE events to ActionLog entries for display
  const sseLogEntries = sseEvents.map((evt) => ({
    message: evt.message || evt.kind,
    status: (evt.kind === "completed" ? "done" :
             evt.kind === "failed" ? "warning" :
             evt.kind === "warning" ? "warning" : "done") as "done" | "warning" | "pending" | "running",
    detail: evt.data ? JSON.stringify(evt.data) : undefined,
    duration: 200,
  }));

  const handleReplan = useCallback(() => {
    setIsReplanning(true);
    setRightTab("log");
    setTimeout(() => {
      setIsReplanning(false);
    }, 3000);
  }, []);

  const toggleConstraint = (id: string) => {
    setConstraints(prev => prev.map(c => c.id === id ? { ...c, active: !c.active } : c));
  };

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
            <div className={`flex items-center gap-1.5 px-2 py-1 rounded-full ${apiAvailable ? "bg-emerald-500/10 text-emerald-400" : "bg-zinc-800 text-zinc-500"}`}>
              {apiAvailable ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
              <span className="text-[10px]">{apiAvailable ? "API Live" : "Demo Mode"}</span>
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

            {/* Target Selector */}
            <div className="space-y-3">
              <label className="text-xs text-zinc-500 uppercase tracking-wider">Select Target Molecule</label>
              <div className="grid grid-cols-3 gap-3">
                {DEMO_TARGETS.map(t => (
                  <button
                    key={t.id}
                    onClick={() => setSelectedTargetId(t.id)}
                    className={`p-4 rounded-xl border text-left transition-all ${
                      selectedTargetId === t.id
                        ? "border-emerald-500/50 bg-emerald-500/5"
                        : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-700"
                    }`}
                  >
                    <div className="text-sm font-medium text-zinc-200">{t.name}</div>
                    <div className="text-[10px] text-zinc-500 mt-1">MW: {t.molecularWeight} &bull; {t.category}</div>
                    <div className="smiles-display text-[10px] text-zinc-600 mt-2 truncate">{t.smiles}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Prompt Box */}
            <div className="space-y-3">
              <label className="text-xs text-zinc-500 uppercase tracking-wider">Planning Instructions</label>
              <div className="relative">
                <textarea
                  value={promptText}
                  onChange={e => setPromptText(e.target.value)}
                  rows={4}
                  className="w-full px-4 py-3 rounded-xl bg-zinc-900 border border-zinc-800 text-sm text-zinc-200 resize-none focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 placeholder-zinc-600"
                  placeholder="Describe your synthesis goals and constraints..."
                />
              </div>
            </div>

            {/* Plan Button */}
            <button
              onClick={handlePlan}
              className="w-full py-4 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-lg flex items-center justify-center gap-3 transition-all hover:shadow-lg hover:shadow-emerald-500/20 pulse-glow"
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
          <span className="text-sm text-zinc-300">{selectedTarget.name}</span>
          {appState === "results" && (
            <>
              <ChevronRight className="w-4 h-4 text-zinc-600" />
              <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400">{filteredRoutes.length} routes</span>
            </>
          )}
        </div>
        <div className="flex items-center gap-2">
          {appState === "results" && (
            <>
              <button
                onClick={handleReplan}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg border border-zinc-700 text-zinc-300 hover:bg-zinc-800 transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" /> Replan
              </button>
              <button
                onClick={() => setShowShopping(true)}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white transition-colors"
              >
                <ShoppingCart className="w-3.5 h-3.5" /> Execute
              </button>
            </>
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
                  <p className="text-xs text-zinc-500">
                    Analyzing {selectedTarget.name}
                    {liveMode && <span className="ml-2 text-emerald-400">(Live API)</span>}
                  </p>
                </div>
              </div>
              {apiError && (
                <div className="mb-4 p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-xs text-amber-300">
                  API: {apiError} &mdash; using demo mode
                </div>
              )}
              {liveMode ? (
                <div className="space-y-1 max-h-[500px] overflow-y-auto p-3 bg-[#0d0d14] rounded-lg border border-zinc-800/50 font-mono text-xs">
                  {sseLogEntries.map((entry, i) => (
                    <div key={i} className="flex items-start gap-2 py-1.5 px-2 rounded animate-fade-in">
                      <span className={`${entry.status === "warning" ? "text-amber-300" : "text-emerald-300"}`}>
                        {entry.message}
                      </span>
                    </div>
                  ))}
                  {streamStatus === "streaming" && (
                    <div className="flex items-center gap-2 py-1.5 px-2 text-zinc-500">
                      <Zap className="w-3 h-3 animate-pulse" />
                      <span className="animate-pulse">Processing...</span>
                    </div>
                  )}
                </div>
              ) : (
                <ActionLog entries={PLANNING_LOG_ENTRIES} running={true} onComplete={handlePlanComplete} />
              )}
            </div>
          </div>
        )}

        {/* Results: tab-dependent views */}
        {appState === "results" && activeTab === "discover" && (
          <DiscoverView targetName={selectedTarget.name} smiles={selectedTarget.smiles} liveData={liveResult?.discovery ?? null} />
        )}

        {appState === "results" && activeTab === "execute" && (
          <ExecuteView
            jobId={jobId}
            route={liveResult?.routes?.[0] ? liveResult.routes[0] as any : null}
          />
        )}

        {appState === "results" && activeTab === "analyze" && (
          <AnalyzeView
            productSmiles={selectedTarget.smiles}
          />
        )}

        {appState === "results" && activeTab === "learn" && (
          <LearnView />
        )}

        {appState === "results" && activeTab === "admin" && (
          <AdminPanel />
        )}

        {appState === "results" && activeTab === "source" && (
          <SourceView items={SHOPPING_LIST_OSIM} liveSourcing={liveResult?.sourcing ?? null} />
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
                    <div className="text-sm font-medium text-zinc-200">{selectedTarget.name}</div>
                    <div className="smiles-display text-[10px] text-zinc-500 mt-1 break-all">{selectedTarget.smiles}</div>
                    <div className="text-[10px] text-zinc-600 mt-1">MW: {selectedTarget.molecularWeight} &bull; {selectedTarget.category}</div>
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

                {/* Scale & Objective */}
                <div>
                  <div className="flex items-center gap-2 mb-2">
                    <SlidersHorizontal className="w-3.5 h-3.5 text-amber-500" />
                    <span className="text-[10px] text-zinc-500 uppercase tracking-wider">Objective</span>
                  </div>
                  <div className="grid grid-cols-2 gap-1.5">
                    {["Fastest", "Cheapest", "Safest", "Greenest"].map(obj => (
                      <button key={obj} className="px-3 py-2 text-xs rounded-lg border border-zinc-800 text-zinc-400 hover:border-zinc-700 hover:text-zinc-300 transition-colors">
                        {obj}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* CENTER PANEL */}
            <div className="flex-1 overflow-y-auto bg-[var(--bg-primary)]">
              <div className="p-4 space-y-4">
                {/* Route Cards */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-zinc-300">Route Leaderboard</h3>
                    <span className="text-[10px] text-zinc-500">{filteredRoutes.length} routes &bull; ranked by combined score</span>
                  </div>
                  <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-3">
                    {filteredRoutes.map(route => (
                      <RouteCard
                        key={route.id}
                        route={route}
                        selected={selectedRoute?.id === route.id}
                        onSelect={() => { setSelectedRouteId(route.id); setExpandedStep(null); }}
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
                      {selectedRoute.steps.map(step => (
                        <StepDetail
                          key={step.stepNumber}
                          step={step}
                          isExpanded={expandedStep === step.stepNumber}
                          onToggle={() => setExpandedStep(expandedStep === step.stepNumber ? null : step.stepNumber)}
                        />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* RIGHT PANEL */}
            <div className="w-80 border-l border-zinc-800/50 overflow-y-auto shrink-0 bg-zinc-950/30">
              {/* Tabs */}
              <div className="flex border-b border-zinc-800/50 sticky top-0 bg-zinc-950/90 backdrop-blur-sm z-10">
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

              <div className="p-4">
                {/* Action Log Tab */}
                {rightTab === "log" && (
                  <div className="space-y-4">
                    {!isReplanning && liveMode && sseLogEntries.length > 0 ? (
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
                            Completed in {liveResult.compute_time_ms?.toFixed(0)}ms
                          </div>
                        )}
                      </div>
                    ) : !isReplanning ? (
                      <ActionLog entries={PLANNING_LOG_ENTRIES} running={false} />
                    ) : null}
                    {isReplanning && (
                      <div className="space-y-3">
                        <div className="text-xs text-amber-400 flex items-center gap-2">
                          <RotateCcw className="w-3.5 h-3.5 animate-spin" /> Replanning with updated constraints...
                        </div>
                        <ActionLog entries={REPLAN_LOG_ENTRIES} running={true} />
                      </div>
                    )}
                  </div>
                )}

                {/* Evidence Tab */}
                {rightTab === "evidence" && selectedRoute && (
                  <div className="space-y-4">
                    <div className="text-xs text-zinc-500 mb-3">
                      Aggregated precedents across all steps in Route #{selectedRoute.rank}
                    </div>
                    {selectedRoute.steps.flatMap(s => s.precedents.map(p => ({ ...p, step: s.stepNumber }))).sort((a, b) => b.similarity - a.similarity).slice(0, 12).map((p, i) => (
                      <div key={i} className="p-3 rounded-lg border border-zinc-800 hover:border-zinc-700 transition-colors">
                        <div className="flex items-start gap-2">
                          <FileText className={`w-4 h-4 shrink-0 mt-0.5 ${
                            p.type === "patent" ? "text-blue-400" :
                            p.type === "literature" ? "text-purple-400" :
                            "text-emerald-400"
                          }`} />
                          <div className="min-w-0">
                            <div className="text-xs text-zinc-300">{p.title}</div>
                            <div className="flex items-center gap-1.5 mt-1 text-[10px] text-zinc-500">
                              <span className="capitalize">{p.type}</span>
                              <span>&bull;</span>
                              <span>{p.year}</span>
                              <span>&bull;</span>
                              <span>Step {p.step}</span>
                            </div>
                          </div>
                          <span className={`text-xs font-medium shrink-0 ${
                            p.similarity >= 0.9 ? "text-emerald-400" : p.similarity >= 0.7 ? "text-amber-400" : "text-zinc-400"
                          }`}>
                            {Math.round(p.similarity * 100)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Copilot Tab */}
                {rightTab === "copilot" && (
                  <div className="space-y-4">
                    <div className="p-3 rounded-lg bg-zinc-900 border border-zinc-800 text-xs text-zinc-400">
                      <div className="flex items-center gap-2 mb-2 text-emerald-400 font-medium">
                        <Sparkles className="w-3.5 h-3.5" /> Rasyn Copilot
                      </div>
                      Ask questions about the planned route, request alternatives, or explore specific steps.
                    </div>

                    {/* Example conversations */}
                    <div className="space-y-3">
                      {[
                        { q: "Why was SNAr chosen over Buchwald coupling for step 2?", a: "SNAr was preferred because: (1) the 4-chloroquinazoline is highly activated toward nucleophilic substitution, (2) it avoids Pd catalysts per your constraints, and (3) precedent shows >85% yield under mild conditions (WO2013014448A1)." },
                        { q: "What if I need to scale beyond 500g?", a: "At >500g scale, I\u2019d recommend: (1) Switch step 1 from Pd/C hydrogenation to Fe/NH\u2084Cl reduction (avoids H\u2082 gas handling), (2) Use continuous flow for acrylamide formation (step 3) to control the exotherm. Overall route score adjusts from 0.91 to 0.87." },
                      ].map((conv, i) => (
                        <div key={i} className="space-y-2">
                          <div className="flex gap-2">
                            <div className="w-5 h-5 rounded-full bg-blue-500/20 flex items-center justify-center shrink-0 mt-0.5">
                              <span className="text-[9px] text-blue-400 font-bold">U</span>
                            </div>
                            <p className="text-xs text-zinc-300">{conv.q}</p>
                          </div>
                          <div className="flex gap-2">
                            <div className="w-5 h-5 rounded-full bg-emerald-500/20 flex items-center justify-center shrink-0 mt-0.5">
                              <Atom className="w-3 h-3 text-emerald-400" />
                            </div>
                            <p className="text-xs text-zinc-400">{conv.a}</p>
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Input */}
                    <div className="sticky bottom-0 pt-2">
                      <div className="flex gap-2">
                        <input
                          type="text"
                          placeholder="Ask about the route..."
                          className="flex-1 px-3 py-2 rounded-lg bg-zinc-900 border border-zinc-800 text-xs text-zinc-200 focus:outline-none focus:border-emerald-500/50 placeholder-zinc-600"
                        />
                        <button className="px-3 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white transition-colors">
                          <ArrowRight className="w-3.5 h-3.5" />
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

      {/* Shopping List Modal */}
      {showShopping && (
        <ShoppingList items={SHOPPING_LIST_OSIM} onClose={() => setShowShopping(false)} />
      )}
    </div>
  );
}
