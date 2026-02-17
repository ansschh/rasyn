"use client";

import { useState } from "react";
import {
  Search, ChevronDown, ChevronUp, FileText, BookOpen, Building2,
  ShieldCheck, ShieldAlert, AlertTriangle, ExternalLink, Bookmark, Clock
} from "lucide-react";
import type { SearchCluster, EvidenceCard } from "../types";
import ActionLog from "./ActionLog";
import { SEARCH_CLUSTERS, DISCOVER_LOG_ENTRIES } from "../data/mock-pipeline";

interface LiveDiscovery {
  papers: { title: string; authors: string | null; year: number | null; doi: string | null; citation_count: number; source: string; journal: string | null; abstract: string | null; url: string | null }[];
  compound_info: Record<string, unknown>;
  sources_queried: string[];
  total_results: number;
}

interface Props {
  targetName: string;
  smiles?: string;
  liveData?: LiveDiscovery | null;
}

function ReliabilityBadge({ score }: { score: number }) {
  const color = score >= 0.85 ? "text-emerald-400 bg-emerald-500/15 border-emerald-500/30"
    : score >= 0.7 ? "text-amber-400 bg-amber-500/15 border-amber-500/30"
    : "text-red-400 bg-red-500/15 border-red-500/30";
  return (
    <span className={`text-[10px] px-2 py-0.5 rounded-full border font-medium ${color}`}>
      {Math.round(score * 100)}% reliable
    </span>
  );
}

function SourceIcon({ type }: { type: string }) {
  switch (type) {
    case "patent": return <FileText className="w-3.5 h-3.5 text-blue-400" />;
    case "literature": return <BookOpen className="w-3.5 h-3.5 text-purple-400" />;
    case "internal": return <Building2 className="w-3.5 h-3.5 text-emerald-400" />;
    default: return <FileText className="w-3.5 h-3.5 text-zinc-400" />;
  }
}

export default function DiscoverView({ targetName, smiles, liveData }: Props) {
  const [searchQuery, setSearchQuery] = useState(
    `Find the closest precedents for forming the C\u2013N bond on ${targetName}. Prefer robust conditions. Avoid Pd.`
  );
  const [hasSearched, setHasSearched] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [expandedCluster, setExpandedCluster] = useState<string | null>(null);

  const handleSearch = () => {
    setIsSearching(true);
    setHasSearched(false);
  };

  const handleSearchComplete = () => {
    setIsSearching(false);
    setHasSearched(true);
    setExpandedCluster(SEARCH_CLUSTERS[0].id);
  };

  return (
    <div className="h-full flex">
      {/* Main content */}
      <div className="flex-1 overflow-y-auto p-6">
        {/* Search bar */}
        <div className="max-w-3xl mx-auto space-y-6">
          <div>
            <h2 className="text-lg font-semibold mb-1">Discover Precedents</h2>
            <p className="text-xs text-zinc-500">Search patents, literature, and internal data with natural language + structure</p>
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

          {/* Metric overlay */}
          {hasSearched && (
            <div className="flex items-center gap-6 px-4 py-2.5 rounded-lg bg-zinc-900/50 border border-zinc-800 text-xs">
              <div><span className="text-zinc-500">Time-to-precedent:</span> <span className="text-emerald-400 font-semibold">45 seconds</span> <span className="text-zinc-600">(vs. 2 hours manual)</span></div>
              <div><span className="text-zinc-500">Sources searched:</span> <span className="text-zinc-300 font-semibold">1.2M patents + 847K papers + 2,847 internal docs</span></div>
            </div>
          )}

          {/* Search animation */}
          {isSearching && (
            <ActionLog entries={DISCOVER_LOG_ENTRIES} running={true} onComplete={handleSearchComplete} />
          )}

          {/* Live API papers (from retrosynthesis discovery) */}
          {liveData && liveData.papers.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-zinc-300">Literature Results (Live API)</h3>
                <span className="text-[10px] text-zinc-500">
                  {liveData.total_results} results &bull; {liveData.sources_queried.join(", ")}
                </span>
              </div>
              <div className="space-y-2">
                {liveData.papers.map((paper, i) => (
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

          {/* Cluster results */}
          {hasSearched && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-zinc-300">Reaction Clusters</h3>
                <span className="text-[10px] text-zinc-500">3 clusters &bull; 2,324 total precedents</span>
              </div>

              {SEARCH_CLUSTERS.map(cluster => {
                const isExpanded = expandedCluster === cluster.id;
                const isBlocked = cluster.failureModes.some(f => f.includes("BLOCKED"));
                return (
                  <div key={cluster.id} className={`rounded-xl border transition-all ${
                    isBlocked ? "border-red-500/20 bg-red-950/10 opacity-60" :
                    isExpanded ? "border-emerald-500/30 bg-zinc-900/80" : "border-zinc-800 bg-zinc-900/40 hover:border-zinc-700"
                  }`}>
                    <div
                      className="flex items-center gap-3 p-4 cursor-pointer"
                      onClick={() => setExpandedCluster(isExpanded ? null : cluster.id)}
                    >
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-zinc-200">{cluster.name}</span>
                          <ReliabilityBadge score={cluster.reliabilityScore} />
                          {isBlocked && (
                            <span className="text-[10px] px-2 py-0.5 rounded-full bg-red-500/15 border border-red-500/30 text-red-400">
                              Constraint violation
                            </span>
                          )}
                        </div>
                        <div className="text-xs text-zinc-500 mt-0.5">{cluster.reactionType} &bull; {cluster.evidenceCount} precedents</div>
                      </div>
                      {isExpanded ? <ChevronUp className="w-4 h-4 text-zinc-500" /> : <ChevronDown className="w-4 h-4 text-zinc-500" />}
                    </div>

                    {isExpanded && (
                      <div className="px-4 pb-4 space-y-4 animate-fade-in">
                        {/* Typical conditions */}
                        <div className="bg-zinc-950 rounded-lg p-3 border border-zinc-800">
                          <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">Typical Conditions</div>
                          <div className="space-y-1">
                            {cluster.typicalConditions.map((c, i) => (
                              <div key={i} className="text-xs text-zinc-300 flex items-start gap-2">
                                <span className="text-emerald-500 mt-0.5">{i + 1}.</span> {c}
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Failure modes */}
                        <div className="bg-zinc-950 rounded-lg p-3 border border-zinc-800">
                          <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">Known Failure Modes</div>
                          <div className="space-y-1">
                            {cluster.failureModes.map((f, i) => (
                              <div key={i} className={`text-xs flex items-start gap-2 ${
                                f.includes("BLOCKED") ? "text-red-400" : "text-amber-300"
                              }`}>
                                <AlertTriangle className="w-3 h-3 mt-0.5 shrink-0" /> {f}
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Evidence cards */}
                        <div>
                          <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">Top Evidence</div>
                          <div className="space-y-2">
                            {cluster.evidenceCards.map(card => (
                              <div key={card.id} className="p-3 rounded-lg border border-zinc-800 hover:border-zinc-700 transition-colors">
                                <div className="flex items-start gap-2">
                                  <SourceIcon type={card.type} />
                                  <div className="flex-1 min-w-0">
                                    <div className="text-xs text-zinc-200 font-medium">{card.title}</div>
                                    <div className="flex items-center gap-1.5 mt-1 text-[10px] text-zinc-500">
                                      <span className="capitalize">{card.type}</span>
                                      <span>&bull;</span>
                                      <span>{card.year}</span>
                                      {card.journal && <><span>&bull;</span><span className="italic">{card.journal}</span></>}
                                      {card.authors && <><span>&bull;</span><span>{card.authors}</span></>}
                                    </div>
                                    <div className="text-[10px] text-zinc-400 mt-1">{card.conditions}</div>
                                    {card.yield && <span className="text-[10px] text-emerald-400">Yield: {card.yield}</span>}
                                    {card.notes && <div className="text-[10px] text-zinc-500 mt-1 italic">{card.notes}</div>}
                                  </div>
                                  <div className="flex items-center gap-2 shrink-0">
                                    <span className={`text-xs font-medium ${
                                      card.similarity >= 0.9 ? "text-emerald-400" : card.similarity >= 0.7 ? "text-amber-400" : "text-zinc-400"
                                    }`}>{Math.round(card.similarity * 100)}%</span>
                                    <button className="p-1 rounded hover:bg-zinc-800 transition-colors">
                                      <Bookmark className="w-3 h-3 text-zinc-600" />
                                    </button>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
