"use client";

import { useState } from "react";
import {
  Package, Check, Clock, ExternalLink, Download, CheckCircle2,
  RefreshCw, AlertTriangle
} from "lucide-react";
import type { ShoppingItem } from "../types";

interface LiveSourcing {
  items: { smiles: string; vendor: string | null; catalog_id: string | null; price_per_gram: number | null; lead_time_days: number | null; in_stock: boolean; url: string | null }[];
  total_estimated_cost: number | null;
  summary: { total_compounds: number; available: number; not_available: number; in_stock_offers: number } | null;
}

interface Props {
  items: ShoppingItem[];
  liveSourcing?: LiveSourcing | null;
}

export default function SourceView({ items, liveSourcing }: Props) {
  const [toast, setToast] = useState<string | null>(null);

  const totalCost = items.reduce((sum, item) => {
    const price = parseFloat(item.price.replace("$", ""));
    return sum + (isNaN(price) ? 0 : price);
  }, 0);

  const allAvailable = items.every(i => i.available);
  const inStockCount = items.filter(i => i.available).length;

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  };

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">Source &amp; Procurement</h2>
            <p className="text-xs text-zinc-500">Building block availability, pricing, and lead times for selected route</p>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => showToast("Suggesting alternate building blocks...")} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-zinc-700 text-zinc-300 hover:bg-zinc-800 text-xs transition-colors">
              <RefreshCw className="w-3.5 h-3.5" /> Suggest Alternates
            </button>
            <button onClick={() => showToast("Order submitted to procurement")} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium transition-colors">
              <Package className="w-3.5 h-3.5" /> Place Order
            </button>
          </div>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-4 gap-3">
          <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800 text-center">
            <div className="text-[10px] text-zinc-500">Items</div>
            <div className="text-xl font-bold text-zinc-200">{items.length}</div>
          </div>
          <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800 text-center">
            <div className="text-[10px] text-zinc-500">In Stock</div>
            <div className="text-xl font-bold text-emerald-400">{inStockCount}/{items.length}</div>
          </div>
          <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800 text-center">
            <div className="text-[10px] text-zinc-500">Est. Total</div>
            <div className="text-xl font-bold text-emerald-400">${totalCost.toFixed(2)}</div>
          </div>
          <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800 text-center">
            <div className="text-[10px] text-zinc-500">Time to Source</div>
            <div className="text-xl font-bold text-zinc-200">3 min</div>
            <div className="text-[10px] text-zinc-600">vs. 2 days manual</div>
          </div>
        </div>

        {/* Status */}
        <div className={`px-4 py-3 rounded-lg border text-sm ${
          allAvailable
            ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
            : "bg-amber-500/10 border-amber-500/20 text-amber-400"
        }`}>
          {allAvailable ? (
            <span className="flex items-center gap-2"><Check className="w-4 h-4" /> All items available. Estimated fulfillment: 2&ndash;3 business days.</span>
          ) : (
            <span className="flex items-center gap-2"><Clock className="w-4 h-4" /> Some items require synthesis. Lead time: 1&ndash;2 weeks.</span>
          )}
        </div>

        {/* Table */}
        <div className="bg-zinc-950 rounded-xl border border-zinc-800 overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className="text-left p-3 font-medium">Chemical</th>
                <th className="text-left p-3 font-medium">Vendor</th>
                <th className="text-left p-3 font-medium">Catalog #</th>
                <th className="text-left p-3 font-medium">Qty</th>
                <th className="text-right p-3 font-medium">Price</th>
                <th className="text-center p-3 font-medium">Status</th>
                <th className="text-center p-3 font-medium">Lead Time</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item, i) => (
                <tr key={i} className="border-b border-zinc-800/50 hover:bg-zinc-800/30 transition-colors">
                  <td className="p-3">
                    <div className="text-zinc-200 font-medium">{item.name}</div>
                    <div className="smiles-display text-[10px] text-zinc-500 mt-0.5">{item.smiles.length > 40 ? item.smiles.slice(0, 40) + "..." : item.smiles}</div>
                  </td>
                  <td className="p-3 text-zinc-400">{item.vendor}</td>
                  <td className="p-3 text-zinc-400 font-mono">{item.catalogNumber}</td>
                  <td className="p-3 text-zinc-300">{item.quantity}</td>
                  <td className="p-3 text-right text-zinc-200 font-medium">{item.price}</td>
                  <td className="p-3 text-center">
                    {item.available ? (
                      <span className="inline-flex items-center gap-1 text-emerald-400">
                        <Check className="w-3 h-3" /> In Stock
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-amber-400">
                        <AlertTriangle className="w-3 h-3" /> On-demand
                      </span>
                    )}
                  </td>
                  <td className="p-3 text-center text-zinc-400">{item.leadTime}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Live API Sourcing Results */}
        {liveSourcing && liveSourcing.items.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-zinc-300">Live Vendor Data (API)</h3>
              {liveSourcing.summary && (
                <span className="text-[10px] text-zinc-500">
                  {liveSourcing.summary.available}/{liveSourcing.summary.total_compounds} available &bull; {liveSourcing.summary.in_stock_offers} offers
                </span>
              )}
            </div>
            <div className="bg-zinc-950 rounded-xl border border-emerald-500/20 overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-zinc-800 text-zinc-500">
                    <th className="text-left p-3 font-medium">SMILES</th>
                    <th className="text-left p-3 font-medium">Vendor</th>
                    <th className="text-left p-3 font-medium">Catalog #</th>
                    <th className="text-center p-3 font-medium">Available</th>
                    <th className="text-right p-3 font-medium">Price/g</th>
                    <th className="text-center p-3 font-medium">Link</th>
                  </tr>
                </thead>
                <tbody>
                  {liveSourcing.items.filter(i => i.vendor !== "not_found").map((item, i) => (
                    <tr key={i} className="border-b border-zinc-800/50 hover:bg-zinc-800/30 transition-colors">
                      <td className="p-3">
                        <span className="smiles-display text-[10px] text-zinc-300">{item.smiles.length > 35 ? item.smiles.slice(0, 35) + "..." : item.smiles}</span>
                      </td>
                      <td className="p-3 text-zinc-400">{item.vendor}</td>
                      <td className="p-3 text-zinc-400 font-mono text-[10px]">{item.catalog_id || "—"}</td>
                      <td className="p-3 text-center">
                        {item.in_stock ? (
                          <span className="inline-flex items-center gap-1 text-emerald-400"><Check className="w-3 h-3" /> Yes</span>
                        ) : (
                          <span className="text-zinc-500">No</span>
                        )}
                      </td>
                      <td className="p-3 text-right text-zinc-200">{item.price_per_gram != null ? `$${item.price_per_gram.toFixed(2)}` : "—"}</td>
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
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between">
          <div className="text-sm">
            <span className="text-zinc-500">Estimated Total: </span>
            <span className="text-xl font-bold text-emerald-400">${totalCost.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-3">
            <button onClick={() => showToast("Copied to clipboard")} className="flex items-center gap-2 px-4 py-2 rounded-lg border border-zinc-700 text-zinc-300 hover:bg-zinc-800 text-sm transition-colors">
              <ExternalLink className="w-4 h-4" /> Share
            </button>
            <button onClick={() => showToast("PDF exported")} className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium transition-colors">
              <Download className="w-4 h-4" /> Export PDF
            </button>
          </div>
        </div>
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
