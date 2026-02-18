"use client";

import { Shield, Beaker, Leaf, Package, Star, AlertTriangle, BookOpen } from "lucide-react";
import type { ApiRoute } from "../lib/api";

interface Props {
  route: ApiRoute;
  selected: boolean;
  onSelect: () => void;
}

function MetricRow({ icon, label, value, detail, status }: {
  icon: React.ReactNode;
  label: string;
  value: string;
  detail?: string;
  status: "good" | "warn" | "bad" | "neutral";
}) {
  const statusColor = {
    good: "text-emerald-400",
    warn: "text-amber-400",
    bad: "text-red-400",
    neutral: "text-zinc-400",
  }[status];

  return (
    <div className="flex items-start gap-2 py-1">
      <div className="shrink-0 mt-0.5">{icon}</div>
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-zinc-400 text-[11px]">{label}</span>
          <span className={`text-[11px] font-medium ${statusColor}`}>{value}</span>
        </div>
        {detail && <div className="text-[10px] text-zinc-600 mt-0.5">{detail}</div>}
      </div>
    </div>
  );
}

export default function RouteCard({ route, selected, onSelect }: Props) {
  const sb = route.score_breakdown;
  const isRecommended = route.rank === 1;

  // Extract raw metrics with safe defaults
  const confidence = sb?.model_confidence;
  const alertCount = sb?.safety_alert_count ?? 0;
  const alertNames = sb?.safety_alerts ?? [];
  const ae = sb?.atom_economy_pct;
  const ef = sb?.e_factor;
  const evCount = sb?.evidence_count ?? 0;
  const evTopSim = sb?.evidence_top_similarity;
  const evLocal = sb?.evidence_local_hits ?? 0;
  const evLive = sb?.evidence_live_hits ?? 0;

  return (
    <div
      onClick={onSelect}
      className={`p-4 rounded-xl border cursor-pointer transition-all duration-200 ${
        selected
          ? "border-emerald-500/50 bg-emerald-500/5 shadow-lg shadow-emerald-500/5"
          : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-700 hover:bg-zinc-900"
      } ${isRecommended ? "ring-1 ring-emerald-500/20" : ""}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-zinc-200">Route #{route.rank}</span>
          {isRecommended && (
            <span className="flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
              <Star className="w-3 h-3" /> Top
            </span>
          )}
        </div>
        {confidence != null && (
          <div className="text-right">
            <div className="text-xl font-bold text-emerald-400">{(confidence * 100).toFixed(1)}%</div>
            <div className="text-[9px] text-zinc-500">model confidence</div>
          </div>
        )}
      </div>

      {/* Constraint violation warning */}
      {route.constraint_violations && route.constraint_violations.length > 0 && (
        <div className="flex items-center gap-1.5 mb-2 p-2 rounded-lg bg-red-500/10 border border-red-500/20">
          <AlertTriangle className="w-3.5 h-3.5 text-red-400 shrink-0" />
          <span className="text-[10px] text-red-400">
            Violates: {route.constraint_violations.map(v => v.replace(/_/g, " ")).join(", ")}
          </span>
        </div>
      )}

      {/* Badges */}
      <div className="flex flex-wrap gap-1.5 mb-3">
        {route.all_purchasable && (
          <span className="text-[10px] px-2 py-0.5 rounded-full border bg-emerald-500/15 text-emerald-400 border-emerald-500/30">All Purchasable</span>
        )}
        <span className="text-[10px] px-2 py-0.5 rounded-full border bg-blue-500/15 text-blue-400 border-blue-500/30">
          {route.num_steps} step{route.num_steps !== 1 ? "s" : ""}
        </span>
        <span className="text-[10px] px-2 py-0.5 rounded-full border bg-zinc-500/15 text-zinc-400 border-zinc-500/30">
          {route.starting_materials.length} reagent{route.starting_materials.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Raw Metrics */}
      {sb && (
        <div className="divide-y divide-zinc-800/50">
          {/* Safety */}
          <MetricRow
            icon={alertCount === 0
              ? <Shield className="w-3.5 h-3.5 text-emerald-500" />
              : <AlertTriangle className="w-3.5 h-3.5 text-amber-500" />}
            label="Safety (PAINS/BRENK)"
            value={alertCount === 0 ? "Clear" : `${alertCount} alert${alertCount > 1 ? "s" : ""}`}
            detail={alertCount > 0 ? alertNames.join(", ") : undefined}
            status={alertCount === 0 ? "good" : alertCount > 2 ? "bad" : "warn"}
          />

          {/* Atom Economy */}
          {ae != null && (
            <MetricRow
              icon={<Leaf className="w-3.5 h-3.5 text-green-500" />}
              label="Atom Economy"
              value={`${ae.toFixed(1)}%`}
              detail="RDKit molecular weight calculation"
              status={ae > 70 ? "good" : ae > 40 ? "warn" : "bad"}
            />
          )}

          {/* E-Factor */}
          {ef != null && (
            <MetricRow
              icon={<Beaker className="w-3.5 h-3.5 text-teal-500" />}
              label="E-Factor"
              value={ef.toFixed(2)}
              detail="Waste/product MW ratio (lower = greener)"
              status={ef < 1.0 ? "good" : ef < 5.0 ? "warn" : "bad"}
            />
          )}

          {/* Evidence */}
          <MetricRow
            icon={<BookOpen className="w-3.5 h-3.5 text-purple-500" />}
            label="Literature Evidence"
            value={evCount > 0
              ? `${evCount} hit${evCount > 1 ? "s" : ""}${evTopSim != null ? ` (${(evTopSim * 100).toFixed(0)}% sim)` : ""}`
              : "None found"}
            detail={evCount > 0 ? `${evLocal} reaction match${evLocal !== 1 ? "es" : ""}, ${evLive} paper${evLive !== 1 ? "s" : ""}` : undefined}
            status={evCount >= 3 ? "good" : evCount > 0 ? "neutral" : "neutral"}
          />

          {/* Availability */}
          <MetricRow
            icon={<Package className="w-3.5 h-3.5 text-cyan-500" />}
            label="Starting Materials"
            value={route.all_purchasable ? "All available" : `${sb.starting_materials_total} identified`}
            status={route.all_purchasable ? "good" : "neutral"}
          />
        </div>
      )}

      {/* No metrics available */}
      {!sb && (
        <div className="text-[10px] text-zinc-500 italic">Metrics not available</div>
      )}
    </div>
  );
}
