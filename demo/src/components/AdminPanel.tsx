"use client";

import { useState, useEffect } from "react";
import {
  Shield, Users, Activity, Lock,
  Server, Globe, Eye, AlertTriangle, Loader2, AlertCircle,
  CheckCircle2, XCircle, Plug
} from "lucide-react";
import type { AuditLogResponse } from "../lib/api";
import type { IntegrationStatusItem } from "../lib/api";

export default function AdminPanel() {
  const [activeSection, setActiveSection] = useState<"audit" | "integrations" | "security">("audit");
  const [liveAuditLog, setLiveAuditLog] = useState<AuditLogResponse | null>(null);
  const [integrations, setIntegrations] = useState<IntegrationStatusItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function fetchData() {
      setIsLoading(true);
      setError(null);
      try {
        const { getAuditLog, getIntegrationStatus } = await import("../lib/api");
        const [auditResult, integResult] = await Promise.all([
          getAuditLog(100),
          getIntegrationStatus().catch(() => ({ integrations: [] })),
        ]);
        if (!cancelled) {
          setLiveAuditLog(auditResult);
          setIntegrations(integResult.integrations || []);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load admin data");
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }
    fetchData();
    return () => { cancelled = true; };
  }, []);

  const connectedCount = integrations.filter(i => i.status === "connected").length;
  const totalCount = integrations.length;

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Shield className="w-5 h-5 text-blue-400" /> Administration
          </h2>
          <p className="text-xs text-zinc-500">Audit trail, integrations, and access controls</p>
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/30 text-xs text-red-400">
            <AlertCircle className="w-4 h-4 shrink-0" /> {error}
          </div>
        )}

        {/* Section tabs */}
        <div className="flex border-b border-zinc-800">
          {([
            { id: "audit" as const, icon: Activity, label: "Audit Log" },
            { id: "integrations" as const, icon: Plug, label: `Integrations (${connectedCount}/${totalCount})` },
            { id: "security" as const, icon: Lock, label: "Security" },
          ]).map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveSection(tab.id)}
              className={`flex items-center gap-1.5 px-4 py-2.5 text-xs font-medium border-b-2 transition-colors ${
                activeSection === tab.id
                  ? "border-blue-500 text-blue-400"
                  : "border-transparent text-zinc-500 hover:text-zinc-300"
              }`}
            >
              <tab.icon className="w-3.5 h-3.5" /> {tab.label}
            </button>
          ))}
        </div>

        {/* Audit Log */}
        {activeSection === "audit" && (
          <div className="space-y-3">
            {isLoading && (
              <div className="flex items-center justify-center py-8 gap-2 text-zinc-400 text-xs">
                <Loader2 className="w-4 h-4 animate-spin" /> Loading audit log...
              </div>
            )}
            {liveAuditLog && (
              <>
                <div className="flex items-center justify-between text-xs text-zinc-500">
                  <span>{liveAuditLog.entries.length} events</span>
                  <span>{liveAuditLog.total} total</span>
                </div>
                <div className="bg-zinc-950 rounded-xl border border-zinc-800 overflow-hidden">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-zinc-800 text-zinc-500">
                        <th className="text-left p-3 font-medium">Timestamp</th>
                        <th className="text-left p-3 font-medium">User</th>
                        <th className="text-left p-3 font-medium">Action</th>
                        <th className="text-left p-3 font-medium">Resource</th>
                        <th className="text-left p-3 font-medium">Details</th>
                      </tr>
                    </thead>
                    <tbody>
                      {liveAuditLog.entries.map((entry, i) => (
                        <tr key={entry.id || i} className="border-b border-zinc-800/50 hover:bg-zinc-800/30 transition-colors">
                          <td className="p-3 text-zinc-400 font-mono whitespace-nowrap">{entry.timestamp}</td>
                          <td className="p-3 text-zinc-300">{entry.user}</td>
                          <td className="p-3">
                            <span className={`inline-flex items-center gap-1 ${
                              (entry.action || "").includes("anomaly") || (entry.action || "").includes("Flagged") || (entry.action || "").includes("BLOCKED") ? "text-amber-400" :
                              entry.user === "System (Auto)" || entry.user === "api_user" ? "text-blue-400" : "text-zinc-200"
                            }`}>
                              {(entry.action || "").includes("anomaly") || (entry.action || "").includes("Flagged") || (entry.action || "").includes("BLOCKED") ? <AlertTriangle className="w-3 h-3" /> :
                               entry.user === "System (Auto)" || entry.user === "api_user" ? <Activity className="w-3 h-3" /> :
                               <Eye className="w-3 h-3" />}
                              {entry.action}
                            </span>
                          </td>
                          <td className="p-3 text-zinc-400 max-w-[200px] truncate">{entry.resource}</td>
                          <td className="p-3 text-zinc-500 max-w-[200px] truncate">{entry.details}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
            {!isLoading && !liveAuditLog && !error && (
              <div className="text-center py-8 text-zinc-500 text-sm">
                No audit log entries found.
              </div>
            )}
          </div>
        )}

        {/* Integrations */}
        {activeSection === "integrations" && (
          <div className="space-y-4">
            {isLoading && (
              <div className="flex items-center justify-center py-8 gap-2 text-zinc-400 text-xs">
                <Loader2 className="w-4 h-4 animate-spin" /> Loading integration status...
              </div>
            )}
            {!isLoading && integrations.length > 0 && (
              <>
                {/* Group by category */}
                {["ai", "vendor", "literature", "eln", "infrastructure"].map(category => {
                  const items = integrations.filter(i => i.category === category);
                  if (items.length === 0) return null;
                  const categoryLabels: Record<string, string> = {
                    ai: "AI / Models",
                    vendor: "Chemical Vendors",
                    literature: "Literature & Data",
                    eln: "ELN / Lab Systems",
                    infrastructure: "Infrastructure",
                  };
                  return (
                    <div key={category}>
                      <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">
                        {categoryLabels[category] || category}
                      </div>
                      <div className="space-y-2">
                        {items.map(item => (
                          <div key={item.name} className="flex items-center gap-3 p-3 rounded-lg bg-zinc-950 border border-zinc-800">
                            {item.status === "connected" ? (
                              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                            ) : (
                              <XCircle className="w-4 h-4 text-zinc-600 shrink-0" />
                            )}
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2">
                                <span className="text-xs font-medium text-zinc-200">{item.name}</span>
                                <span className={`text-[10px] px-1.5 py-0.5 rounded-full border ${
                                  item.status === "connected"
                                    ? "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
                                    : "bg-zinc-800 text-zinc-500 border-zinc-700"
                                }`}>
                                  {item.status === "connected" ? "Connected" : "Not configured"}
                                </span>
                              </div>
                              <div className="text-[10px] text-zinc-500 mt-0.5 truncate">{item.detail}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </>
            )}
            {!isLoading && integrations.length === 0 && !error && (
              <div className="text-center py-8 text-zinc-500 text-sm">
                Could not fetch integration status from the server.
              </div>
            )}
          </div>
        )}

        {/* Security */}
        {activeSection === "security" && (
          <div className="space-y-4">
            {/* Permissions (real from RBAC) */}
            <div className="p-4 rounded-xl bg-zinc-900 border border-zinc-800">
              <div className="flex items-center gap-2 mb-3">
                <Users className="w-4 h-4 text-blue-400" />
                <span className="text-sm font-medium text-zinc-200">Roles &amp; Permissions (RBAC)</span>
              </div>
              <div className="space-y-2">
                {[
                  { role: "Admin", perms: ["Full access"] },
                  { role: "Researcher", perms: ["Plan routes", "Execute experiments", "Upload analysis", "Write outcomes", "View all data"] },
                  { role: "Viewer", perms: ["View routes", "View analytics", "Read-only access"] },
                ].map(r => (
                  <div key={r.role} className="flex items-center gap-3 p-3 rounded-lg bg-zinc-950 border border-zinc-800">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium text-zinc-200">{r.role}</span>
                      </div>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {r.perms.map(p => (
                          <span key={p} className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400">{p}</span>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Deployment */}
            <div className="p-4 rounded-xl bg-zinc-900 border border-zinc-800">
              <div className="flex items-center gap-2 mb-3">
                <Server className="w-4 h-4 text-purple-400" />
                <span className="text-sm font-medium text-zinc-200">Deployment</span>
              </div>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800">
                  <div className="text-zinc-500">Environment</div>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <Globe className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-zinc-200 font-medium">AWS EC2 (us-east-1)</span>
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800">
                  <div className="text-zinc-500">Data Storage</div>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <Lock className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-zinc-200 font-medium">PostgreSQL (encrypted at rest)</span>
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800">
                  <div className="text-zinc-500">Model Inference</div>
                  <div className="text-zinc-200 font-medium mt-0.5">GPU server (RetroTx v2 + RSGPT)</div>
                </div>
                <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800">
                  <div className="text-zinc-500">Guardrails</div>
                  <div className="text-zinc-200 font-medium mt-0.5">CWC Schedule 1-3 screening active</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
