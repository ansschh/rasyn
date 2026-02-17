"use client";

import { Shield, TrendingUp, Leaf, FlaskConical, Package, Star, Lightbulb } from "lucide-react";
import type { ApiRoute } from "../lib/api";

interface Props {
  route: ApiRoute;
  selected: boolean;
  onSelect: () => void;
}

function ScoreBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden">
      <div
        className={`h-full rounded-full score-bar-fill ${color}`}
        style={{ "--fill-width": `${value * 100}%`, width: `${value * 100}%` } as React.CSSProperties}
      />
    </div>
  );
}

export default function RouteCard({ route, selected, onSelect }: Props) {
  const sb = route.score_breakdown;
  const isRecommended = route.rank === 1;

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
        <div className="text-2xl font-bold text-emerald-400">{Math.round(route.overall_score * 100)}</div>
      </div>

      {/* Badges */}
      <div className="flex flex-wrap gap-1.5 mb-3">
        {route.all_purchasable && (
          <span className="text-[10px] px-2 py-0.5 rounded-full border bg-emerald-500/15 text-emerald-400 border-emerald-500/30">All Purchasable</span>
        )}
        <span className="text-[10px] px-2 py-0.5 rounded-full border bg-blue-500/15 text-blue-400 border-blue-500/30">
          {route.num_steps} steps
        </span>
        <span className="text-[10px] px-2 py-0.5 rounded-full border bg-zinc-500/15 text-zinc-400 border-zinc-500/30">
          {route.starting_materials.length} reagents
        </span>
      </div>

      {/* Score Breakdown */}
      {sb && (
        <div className="space-y-2 text-[11px]">
          {sb.safety != null && (
            <div className="flex items-center gap-2">
              <Shield className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
              <span className="text-zinc-400 w-20">Safety</span>
              <ScoreBar value={sb.safety} color="bg-emerald-500" />
              <span className="text-zinc-300 w-8 text-right">{Math.round(sb.safety * 100)}</span>
            </div>
          )}
          {sb.step_efficiency != null && (
            <div className="flex items-center gap-2">
              <TrendingUp className="w-3.5 h-3.5 text-blue-500 shrink-0" />
              <span className="text-zinc-400 w-20">Efficiency</span>
              <ScoreBar value={sb.step_efficiency} color="bg-blue-500" />
              <span className="text-zinc-300 w-8 text-right">{Math.round(sb.step_efficiency * 100)}</span>
            </div>
          )}
          {sb.green_chemistry != null && (
            <div className="flex items-center gap-2">
              <Leaf className="w-3.5 h-3.5 text-green-500 shrink-0" />
              <span className="text-zinc-400 w-20">Greenness</span>
              <ScoreBar value={sb.green_chemistry} color="bg-green-500" />
              <span className="text-zinc-300 w-8 text-right">{Math.round(sb.green_chemistry * 100)}</span>
            </div>
          )}
          {sb.precedent != null && (
            <div className="flex items-center gap-2">
              <FlaskConical className="w-3.5 h-3.5 text-purple-500 shrink-0" />
              <span className="text-zinc-400 w-20">Precedent</span>
              <ScoreBar value={sb.precedent} color="bg-purple-500" />
              <span className="text-zinc-300 w-8 text-right">{Math.round(sb.precedent * 100)}</span>
            </div>
          )}
          {sb.availability != null && (
            <div className="flex items-center gap-2">
              <Package className="w-3.5 h-3.5 text-cyan-500 shrink-0" />
              <span className="text-zinc-400 w-20">Availability</span>
              <ScoreBar value={sb.availability} color="bg-cyan-500" />
              <span className="text-zinc-300 w-8 text-right">{Math.round(sb.availability * 100)}</span>
            </div>
          )}
          {sb.roundtrip_confidence != null && (
            <div className="flex items-center gap-2">
              <Lightbulb className="w-3.5 h-3.5 text-amber-500 shrink-0" />
              <span className="text-zinc-400 w-20">Confidence</span>
              <ScoreBar value={sb.roundtrip_confidence} color="bg-amber-500" />
              <span className="text-zinc-300 w-8 text-right">{Math.round(sb.roundtrip_confidence * 100)}</span>
            </div>
          )}
        </div>
      )}

      {/* No breakdown available */}
      {!sb && (
        <div className="text-[10px] text-zinc-500 italic">Score breakdown not available</div>
      )}
    </div>
  );
}
