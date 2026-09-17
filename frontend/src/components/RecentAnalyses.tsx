import { useNavigate } from "react-router-dom";
import { getSessionResult, loadHistory } from "../lib/report";
import type { HistoryEntry } from "../lib/report";

// Lightweight, local-only history: metadata persists in localStorage, but the full
// result object is only kept in memory for this browser session (never uploaded or
// written to disk), so an entry can only be reopened until the page is reloaded.
export default function RecentAnalyses({ refreshKey }: { refreshKey: number }) {
  const navigate = useNavigate();
  const entries: HistoryEntry[] = loadHistory();
  void refreshKey;

  if (entries.length === 0) return null;

  return (
    <div className="mt-20 border-t pt-10" style={{ borderColor: "var(--page-border)" }}>
      <p className="text-xs uppercase tracking-widest opacity-60 mb-4">Recent Analyses</p>
      <div className="flex flex-col divide-y" style={{ borderColor: "var(--page-border)" }}>
        {entries.map((entry) => {
          const cached = getSessionResult(entry.id);
          return (
            <button
              key={entry.id}
              disabled={!cached}
              onClick={() =>
                cached &&
                navigate("/results", {
                  state: { result: cached, mode: entry.mode, filename: entry.filename },
                })
              }
              className="w-full flex items-center justify-between gap-4 py-3 text-left text-sm disabled:cursor-not-allowed disabled:opacity-40 hover:opacity-100 opacity-80 transition-opacity"
              style={{ borderColor: "var(--page-border)" }}
              title={cached ? "Reopen this result" : "Full result no longer available this session"}
            >
              <span className="truncate">{entry.filename}</span>
              <span className="shrink-0 opacity-60">{entry.resultLabel}</span>
              <span className="shrink-0 font-display">{(entry.score * 100).toFixed(1)}%</span>
              <span className="shrink-0 text-xs opacity-40 hidden sm:inline">
                {new Date(entry.timestamp).toLocaleTimeString()}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
