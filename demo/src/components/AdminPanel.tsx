"use client";

import { useState, useEffect } from "react";
import {
  Shield, Users, Activity, Lock,
  Server, Globe, Eye, AlertTriangle, Loader2, AlertCircle
} from "lucide-react";
import type { AuditLogResponse } from "../lib/api";

export default function AdminPanel() {
  const [activeSection, setActiveSection] = useState<"audit" | "security">("audit");
  const [liveAuditLog, setLiveAuditLog] = useState<AuditLogResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function fetchAuditLog() {
      setIsLoading(true);
      setError(null);
      try {
        const { getAuditLog } = await import("../lib/api");
        const result = await getAuditLog(100);
        if (!cancelled) {
          setLiveAuditLog(result);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Failed to load audit log");
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }
    fetchAuditLog();
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Shield className="w-5 h-5 text-blue-400" /> Administration
          </h2>
          <p className="text-xs text-zinc-500">Enterprise controls: SSO, audit trail, permissions</p>
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

        {/* Security */}
        {activeSection === "security" && (
          <div className="space-y-4">
            {/* SSO */}
            <div className="p-4 rounded-xl bg-zinc-900 border border-zinc-800">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Lock className="w-4 h-4 text-emerald-400" />
                  <span className="text-sm font-medium text-zinc-200">Single Sign-On (SSO)</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-emerald-400">Enabled</span>
                  <div className="w-8 h-4 rounded-full bg-emerald-500 relative">
                    <div className="absolute right-0.5 top-0.5 w-3 h-3 rounded-full bg-white" />
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800">
                  <div className="text-zinc-500">Provider</div>
                  <div className="text-zinc-200 font-medium mt-0.5">Okta (SAML 2.0)</div>
                </div>
                <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800">
                  <div className="text-zinc-500">Active Users</div>
                  <div className="text-zinc-200 font-medium mt-0.5">12 / 25 seats</div>
                </div>
              </div>
            </div>

            {/* Permissions */}
            <div className="p-4 rounded-xl bg-zinc-900 border border-zinc-800">
              <div className="flex items-center gap-2 mb-3">
                <Users className="w-4 h-4 text-blue-400" />
                <span className="text-sm font-medium text-zinc-200">Roles &amp; Permissions</span>
              </div>
              <div className="space-y-2">
                {[
                  { role: "Admin", users: 2, perms: ["Full access", "User management", "Audit logs", "Billing"] },
                  { role: "Researcher", users: 8, perms: ["Plan routes", "Run experiments", "View analytics", "Export data"] },
                  { role: "Viewer", users: 2, perms: ["View routes", "View analytics", "Read-only access"] },
                ].map(r => (
                  <div key={r.role} className="flex items-center gap-3 p-3 rounded-lg bg-zinc-950 border border-zinc-800">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium text-zinc-200">{r.role}</span>
                        <span className="text-[10px] text-zinc-500">{r.users} users</span>
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
                    <span className="text-zinc-200 font-medium">AWS VPC (us-east-1)</span>
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800">
                  <div className="text-zinc-500">Data Residency</div>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <Lock className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-zinc-200 font-medium">All data in customer VPC</span>
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800">
                  <div className="text-zinc-500">Model Inference</div>
                  <div className="text-zinc-200 font-medium mt-0.5">On-premises GPU (A100)</div>
                </div>
                <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800">
                  <div className="text-zinc-500">Compliance</div>
                  <div className="text-zinc-200 font-medium mt-0.5">SOC 2 Type II (in progress)</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
