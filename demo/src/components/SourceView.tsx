"use client";

import { useState } from "react";
import {
  Package, Check, Clock, ExternalLink, Download, CheckCircle2,
  RefreshCw, AlertTriangle, AlertCircle, Loader2
} from "lucide-react";
import type { SourcingResult } from "../lib/api";

interface Props {
  liveSourcing?: SourcingResult | null;
}

export default function SourceView({ liveSourcing }: Props) {
  const [toast, setToast] = useState<string | null>(null);
  const [alternatesLoading, setAlternatesLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  };

  const handleSuggestAlternates = async (smiles: string) => {
    setAlternatesLoading(smiles);
    setError(null);
    try {
      const { findAlternates } = await import("../lib/api");
      const result = await findAlternates(smiles, 5);
      showToast(`Found ${result.alternates.length} alternate building blocks`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to find alternates");
    } finally {
      setAlternatesLoading(null);
    }
  };

  // No sourcing data yet
  if (!liveSourcing) {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <div className="text-center space-y-3">
          <Package className="w-10 h-10 text-zinc-600 mx-auto" />
          <p className="text-sm text-zinc-400">Run a synthesis plan to see sourcing data</p>
          <p className="text-xs text-zinc-600">Sourcing quotes are automatically fetched when a plan completes</p>
        </div>
      </div>
    );
  }

  const items = liveSourcing.items.filter(i => i.vendor !== "not_found");
  const notFoundItems = liveSourcing.items.filter(i => i.vendor === "not_found");
  const summary = liveSourcing.summary;

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">Source &amp; Procurement</h2>
            <p className="text-xs text-zinc-500">Building block availability, pricing, and lead times</p>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-xs text-red-400">
            <AlertCircle className="w-4 h-4 shrink-0" /> {error}
          </div>
        )}

        {/* Metrics */}
        {summary && (
          <div className="grid grid-cols-4 gap-3">
            <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800 text-center">
              <div className="text-[10px] text-zinc-500">Compounds</div>
              <div className="text-xl font-bold text-zinc-200">{summary.total_compounds}</div>
            </div>
            <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800 text-center">
              <div className="text-[10px] text-zinc-500">Available</div>
              <div className="text-xl font-bold text-emerald-400">{summary.available}/{summary.total_compounds}</div>
            </div>
            <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800 text-center">
              <div className="text-[10px] text-zinc-500">In-Stock Offers</div>
              <div className="text-xl font-bold text-emerald-400">{summary.in_stock_offers}</div>
            </div>
            <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800 text-center">
              <div className="text-[10px] text-zinc-500">Est. Total</div>
              <div className="text-xl font-bold text-emerald-400">
                {liveSourcing.total_estimated_cost != null ? `$${liveSourcing.total_estimated_cost.toFixed(2)}` : "\u2014"}
              </div>
            </div>
          </div>
        )}

        {/* Status */}
        {summary && (
          <div className={`px-4 py-3 rounded-lg border text-sm ${
            summary.not_available === 0
              ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
              : "bg-amber-500/10 border-amber-500/20 text-amber-400"
          }`}>
            {summary.not_available === 0 ? (
              <span className="flex items-center gap-2"><Check className="w-4 h-4" /> All compounds sourced.</span>
            ) : (
              <span className="flex items-center gap-2"><Clock className="w-4 h-4" /> {summary.not_available} compound(s) not found from vendors.</span>
            )}
          </div>
        )}

        {/* Available items table */}
        {items.length > 0 && (
          <div className="bg-zinc-950 rounded-xl border border-zinc-800 overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-zinc-800 text-zinc-500">
                  <th className="text-left p-3 font-medium">SMILES</th>
                  <th className="text-left p-3 font-medium">Vendor</th>
                  <th className="text-left p-3 font-medium">Catalog #</th>
                  <th className="text-center p-3 font-medium">In Stock</th>
                  <th className="text-right p-3 font-medium">Price/g</th>
                  <th className="text-center p-3 font-medium">Lead Time</th>
                  <th className="text-center p-3 font-medium">Link</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item, i) => (
                  <tr key={i} className="border-b border-zinc-800/50 hover:bg-zinc-800/30 transition-colors">
                    <td className="p-3">
                      <span className="smiles-display text-[10px] text-zinc-300">{item.smiles.length > 35 ? item.smiles.slice(0, 35) + "..." : item.smiles}</span>
                    </td>
                    <td className="p-3 text-zinc-400">{item.vendor}</td>
                    <td className="p-3 text-zinc-400 font-mono text-[10px]">{item.catalog_id || "\u2014"}</td>
                    <td className="p-3 text-center">
                      {item.in_stock ? (
                        <span className="inline-flex items-center gap-1 text-emerald-400"><Check className="w-3 h-3" /> Yes</span>
                      ) : (
                        <span className="text-zinc-500">No</span>
                      )}
                    </td>
                    <td className="p-3 text-right text-zinc-200">{item.price_per_gram != null ? `$${item.price_per_gram.toFixed(2)}` : "\u2014"}</td>
                    <td className="p-3 text-center text-zinc-400">{item.lead_time_days != null ? `${item.lead_time_days}d` : "\u2014"}</td>
                    <td className="p-3 text-center">
                      {item.url && (
                        <a href={item.url} target="_blank" rel="noopener noreferrer" className="text-emerald-400 hover:text-emerald-300">
                          <ExternalLink className="w-3 h-3 inline" />
                        </a>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Not found items */}
        {notFoundItems.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs text-zinc-500">Not found from vendors:</div>
            {notFoundItems.map((item, i) => (
              <div key={i} className="flex items-center justify-between p-3 rounded-lg border border-amber-500/20 bg-amber-500/5">
                <span className="smiles-display text-[10px] text-zinc-300">{item.smiles}</span>
                <button
                  onClick={() => handleSuggestAlternates(item.smiles)}
                  disabled={alternatesLoading === item.smiles}
                  className="flex items-center gap-1.5 px-2 py-1 rounded-lg border border-zinc-700 text-zinc-300 hover:bg-zinc-800 text-[10px] transition-colors disabled:opacity-50"
                >
                  {alternatesLoading === item.smiles ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <RefreshCw className="w-3 h-3" />
                  )}
                  Find Alternates
                </button>
              </div>
            ))}
          </div>
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
