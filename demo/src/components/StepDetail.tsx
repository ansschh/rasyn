"use client";

import { ArrowRight, Beaker, ChevronDown, ChevronUp } from "lucide-react";
import type { ApiStep } from "../lib/api";

interface Props {
  step: ApiStep;
  stepNumber: number;
  isExpanded: boolean;
  onToggle: () => void;
}

export default function StepDetail({ step, stepNumber, isExpanded, onToggle }: Props) {
  return (
    <div className={`rounded-xl border transition-all duration-200 ${
      isExpanded ? "border-emerald-500/30 bg-zinc-900/80" : "border-zinc-800 bg-zinc-900/40 hover:border-zinc-700"
    }`}>
      {/* Step Header */}
      <div className="flex items-center gap-3 p-4 cursor-pointer" onClick={onToggle}>
        <div className="w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center text-sm font-bold shrink-0">
          {stepNumber}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-zinc-200 truncate">
              {step.rxn_class || `Step ${stepNumber}`}
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded-full border bg-zinc-500/15 text-zinc-400 border-zinc-500/30">
              {step.model}
            </span>
          </div>
          <div className="smiles-display text-[10px] text-zinc-500 mt-0.5 truncate">
            {step.reactants.join(" + ")} &rarr; {step.product}
          </div>
        </div>
        <div className="flex items-center gap-3 text-xs shrink-0">
          <div className="text-center">
            <div className="text-zinc-500">Score</div>
            <div className="font-medium text-emerald-400">{Math.round(step.score * 100)}%</div>
          </div>
          {isExpanded ? <ChevronUp className="w-4 h-4 text-zinc-500" /> : <ChevronDown className="w-4 h-4 text-zinc-500" />}
        </div>
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-4 animate-fade-in">
          {/* Reaction Display */}
          <div className="bg-zinc-950 rounded-lg p-4 border border-zinc-800">
            <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">Reaction</div>
            <div className="flex items-center gap-3 flex-wrap">
              {step.reactants.map((r, i) => (
                <div key={i} className="flex items-center gap-2">
                  {i > 0 && <span className="text-zinc-600 text-lg">+</span>}
                  <div className="px-3 py-2 rounded-lg border border-zinc-700 bg-zinc-800/50">
                    <div className="smiles-display text-xs text-zinc-300 font-mono">{r.length > 50 ? r.slice(0, 50) + "..." : r}</div>
                  </div>
                </div>
              ))}
              <ArrowRight className="w-5 h-5 text-emerald-500 shrink-0 mx-2" />
              <div className="px-3 py-2 rounded-lg border border-emerald-500/30 bg-emerald-500/5">
                <div className="smiles-display text-xs text-emerald-300 font-mono">{step.product.length > 50 ? step.product.slice(0, 50) + "..." : step.product}</div>
              </div>
            </div>
          </div>

          {/* Conditions */}
          {step.conditions && Object.keys(step.conditions).length > 0 && (
            <div className="bg-zinc-950 rounded-lg p-4 border border-zinc-800">
              <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">Predicted Conditions</div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
                {Object.entries(step.conditions).map(([key, val]) => (
                  <div key={key} className="flex items-center gap-2">
                    <Beaker className="w-3.5 h-3.5 text-blue-400" />
                    <div>
                      <div className="text-zinc-500 capitalize">{key.replace(/_/g, " ")}</div>
                      <div className="text-zinc-200">{String(val)}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Meta info */}
          <div className="flex items-center gap-4 text-[10px] text-zinc-500">
            <span>Model: <span className="text-zinc-300">{step.model}</span></span>
            <span>Score: <span className="text-emerald-400">{(step.score * 100).toFixed(1)}%</span></span>
            {step.rxn_class && <span>Class: <span className="text-zinc-300">{step.rxn_class}</span></span>}
          </div>
        </div>
      )}
    </div>
  );
}
