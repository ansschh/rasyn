"use client";

import { useEffect, useState, useRef } from "react";
import { CheckCircle2, Loader2, AlertTriangle, Circle } from "lucide-react";
import type { ActionLogEntry } from "../types";

interface Props {
  entries: Omit<ActionLogEntry, "id" | "timestamp">[];
  running: boolean;
  onComplete?: () => void;
}

export default function ActionLog({ entries, running, onComplete }: Props) {
  const [visibleCount, setVisibleCount] = useState(0);
  const [activeIdx, setActiveIdx] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!running) {
      setVisibleCount(0);
      setActiveIdx(-1);
      return;
    }
    let idx = 0;
    setVisibleCount(0);
    setActiveIdx(0);

    const runNext = () => {
      if (idx >= entries.length) {
        setActiveIdx(-1);
        onComplete?.();
        return;
      }
      setActiveIdx(idx);
      setVisibleCount(idx + 1);

      const delay = entries[idx].duration || 500;
      idx++;
      setTimeout(runNext, delay);
    };
    const t = setTimeout(runNext, 300);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [visibleCount]);

  const StatusIcon = ({ status, isActive }: { status: string; isActive: boolean }) => {
    if (isActive) return <Loader2 className="w-4 h-4 text-emerald-400 animate-spin shrink-0" />;
    switch (status) {
      case "done": return <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />;
      case "warning": return <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0" />;
      default: return <Circle className="w-4 h-4 text-zinc-600 shrink-0" />;
    }
  };

  return (
    <div ref={containerRef} className="space-y-1 max-h-[500px] overflow-y-auto p-3 bg-[#0d0d14] rounded-lg border border-zinc-800/50 font-mono text-xs">
      {entries.slice(0, visibleCount).map((entry, i) => {
        const isActive = i === activeIdx;
        return (
          <div
            key={i}
            className={`flex items-start gap-2 py-1.5 px-2 rounded transition-all duration-300 ${
              isActive ? "bg-emerald-500/10 border border-emerald-500/20" : ""
            } animate-fade-in`}
          >
            <StatusIcon status={entry.status} isActive={isActive} />
            <div className="min-w-0">
              <span className={`${isActive ? "text-emerald-300" : entry.status === "warning" ? "text-amber-300" : "text-zinc-300"}`}>
                {entry.message}
              </span>
              {entry.detail && (
                <div className="text-zinc-500 text-[10px] mt-0.5 leading-tight">{entry.detail}</div>
              )}
            </div>
          </div>
        );
      })}
      {running && visibleCount < entries.length && (
        <div className="flex items-center gap-2 py-1.5 px-2 text-zinc-500">
          <Loader2 className="w-3 h-3 animate-spin" />
          <span className="animate-pulse">Processing...</span>
        </div>
      )}
    </div>
  );
}
