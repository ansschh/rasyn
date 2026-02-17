/**
 * React hook for consuming SSE job events from the Rasyn API.
 *
 * Usage:
 *   const { events, status, result, error } = useJobStream(jobId);
 */

"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { getStreamUrl, fetchResult } from "../lib/api";
import type { PlanResult } from "../lib/api";

export interface SSEEvent {
  kind: string;
  message: string;
  data: Record<string, unknown>;
  ts: string;
}

type StreamStatus = "idle" | "connecting" | "streaming" | "completed" | "failed" | "error";

interface UseJobStreamReturn {
  events: SSEEvent[];
  status: StreamStatus;
  result: PlanResult | null;
  error: string | null;
}

export function useJobStream(jobId: string | null): UseJobStreamReturn {
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [result, setResult] = useState<PlanResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  // Fetch final result when job completes
  const fetchFinalResult = useCallback(async (id: string) => {
    try {
      const res = await fetchResult(id);
      setResult(res);
    } catch (err) {
      setError(`Failed to fetch result: ${err}`);
    }
  }, []);

  useEffect(() => {
    if (!jobId) {
      setEvents([]);
      setStatus("idle");
      setResult(null);
      setError(null);
      return;
    }

    // Reset state for new job
    setEvents([]);
    setStatus("connecting");
    setResult(null);
    setError(null);

    const url = getStreamUrl(jobId);
    const es = new EventSource(url);
    esRef.current = es;

    es.onopen = () => {
      setStatus("streaming");
    };

    // Listen to all event types
    const eventTypes = [
      "planning_started",
      "model_running",
      "step_complete",
      "enriching",
      "info",
      "warning",
      "completed",
      "failed",
    ];

    for (const eventType of eventTypes) {
      es.addEventListener(eventType, (e: MessageEvent) => {
        try {
          const parsed: SSEEvent = JSON.parse(e.data);
          setEvents((prev) => [...prev, parsed]);

          if (eventType === "completed") {
            setStatus("completed");
            fetchFinalResult(jobId);
            es.close();
          } else if (eventType === "failed") {
            setStatus("failed");
            setError(parsed.message || "Job failed");
            es.close();
          }
        } catch {
          // Ignore unparseable events
        }
      });
    }

    es.onerror = () => {
      // EventSource will auto-reconnect on transient errors.
      // If the connection is permanently lost, close it.
      if (es.readyState === EventSource.CLOSED) {
        setStatus("error");
        setError("Connection lost");
      }
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [jobId, fetchFinalResult]);

  return { events, status, result, error };
}
