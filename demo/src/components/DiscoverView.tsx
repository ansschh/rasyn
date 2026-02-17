"use client";

import { useState } from "react";
import {
  Search, BookOpen, ExternalLink, Loader2, AlertCircle
} from "lucide-react";
import { quickDiscovery } from "../lib/api";
import type { DiscoveryResult } from "../lib/api";

interface Props {
  targetName: string;
  smiles?: string;
  liveData?: DiscoveryResult | null;
}

export default function DiscoverView({ targetName, smiles, liveData }: Props) {
  const [searchQuery, setSearchQuery] = useState(
    `Find the closest precedents for forming the C\u2013N bond on ${targetName}. Prefer robust conditions. Avoid Pd.`
  );
  const [isSearching, setIsSearching] = useState(false);
  const [searchResult, setSearchResult] = useState<DiscoveryResult | null>(liveData ?? null);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async () => {
    setIsSearching(true);
    setError(null);
    try {
      const result = await quickDiscovery(searchQuery, smiles, 20);
      setSearchResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setIsSearching(false);
    }
  };

  const data = searchResult;

  return (
    <div className="h-full flex">
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-3xl mx-auto space-y-6">
          <div>
            <h2 className="text-lg font-semibold mb-1">Discover Precedents</h2>
            <p className="text-xs text-zinc-500">Search patents, literature, and databases with natural language + structure</p>
          </div>

          <div className="relative">
            <Search className="absolute left-4 top-3.5 w-4 h-4 text-zinc-500" />
            <textarea
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              rows={2}
              className="w-full pl-10 pr-4 py-3 rounded-xl bg-zinc-900 border border-zinc-800 text-sm text-zinc-200 resize-none focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 placeholder-zinc-600"
              placeholder="Natural language query + constraints..."
            />
            <button
              onClick={handleSearch}
              disabled={isSearching}
              className="absolute right-3 bottom-3 px-3 py-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium transition-colors disabled:opacity-50"
            >
              {isSearching ? "Searching..." : "Search"}
            </button>
          </div>

          {/* Error */}
          {error && (
            <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-xs text-red-400">
              <AlertCircle className="w-4 h-4 shrink-0" /> {error}
            </div>
          )}

          {/* Loading */}
          {isSearching && (
            <div className="flex items-center justify-center py-12 gap-3 text-zinc-400">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span className="text-sm">Searching literature databases...</span>
            </div>
          )}

          {/* Results */}
          {data && data.papers.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-zinc-300">Literature Results</h3>
                <span className="text-[10px] text-zinc-500">
                  {data.total_results} results &bull; {data.sources_queried.join(", ")}
                </span>
              </div>
              <div className="space-y-2">
                {data.papers.map((paper, i) => (
                  <div key={i} className="p-3 rounded-lg border border-zinc-800 hover:border-zinc-700 transition-colors">
                    <div className="flex items-start gap-2">
                      <BookOpen className="w-4 h-4 text-purple-400 shrink-0 mt-0.5" />
                      <div className="flex-1 min-w-0">
                        <div className="text-xs text-zinc-200 font-medium">{paper.title}</div>
                        <div className="flex items-center gap-1.5 mt-1 text-[10px] text-zinc-500 flex-wrap">
                          {paper.authors && <span>{paper.authors}</span>}
                          {paper.year && <><span>&bull;</span><span>{paper.year}</span></>}
                          {paper.journal && <><span>&bull;</span><span className="italic">{paper.journal}</span></>}
                          {paper.citation_count > 0 && <><span>&bull;</span><span>{paper.citation_count} citations</span></>}
                          <span>&bull;</span><span className="text-emerald-400">{paper.source}</span>
                        </div>
                        {paper.abstract && (
                          <div className="text-[10px] text-zinc-400 mt-1.5 line-clamp-2">{paper.abstract}</div>
                        )}
                      </div>
                      {paper.url && (
                        <a href={paper.url} target="_blank" rel="noopener noreferrer" className="p-1 rounded hover:bg-zinc-800 transition-colors shrink-0">
                          <ExternalLink className="w-3 h-3 text-zinc-500" />
                        </a>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Empty state */}
          {data && data.papers.length === 0 && !isSearching && (
            <div className="text-center py-12 text-zinc-500 text-sm">
              No papers found. Try a different query.
            </div>
          )}

          {/* No search yet */}
          {!data && !isSearching && !error && (
            <div className="text-center py-12 text-zinc-500 text-sm">
              Enter a query and click Search to discover literature precedents.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
