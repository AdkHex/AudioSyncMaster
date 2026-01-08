import { CheckCircle, XCircle, Download } from "lucide-react";

interface SyncResult {
  id: string;
  videoFile: string;
  audioFile: string;
  startDelay: number | null;
  endDelay: number | null;
  confidence: "high" | "medium" | "low" | null;
  status: "success" | "error";
  error?: string;
}

interface ResultsTableProps {
  results: SyncResult[];
  onExport: () => void;
}

export const ResultsTable = ({ results, onExport }: ResultsTableProps) => {
  const ConfidenceBadge = ({ confidence }: { confidence: "high" | "medium" | "low" | null }) => {
    if (!confidence) return <span className="text-muted-foreground">—</span>;
    
    const config = {
      high: { label: "High", class: "bg-success/15 text-success border-success/30" },
      medium: { label: "Medium", class: "bg-warning/15 text-warning border-warning/30" },
      low: { label: "Low", class: "bg-destructive/15 text-destructive border-destructive/30" }
    };
    
    return (
      <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${config[confidence].class}`}>
        {config[confidence].label}
      </span>
    );
  };

  const formatDelay = (delay: number | null) => {
    if (delay === null) return "—";
    const sign = delay >= 0 ? "+" : "";
    return `${sign}${delay.toFixed(1)}ms`;
  };

  return (
    <div className="rounded-xl border border-border overflow-hidden bg-card animate-slide-up">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-border bg-secondary/30">
        <h3 className="font-semibold">Sync Results</h3>
        <button
          onClick={onExport}
          className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-primary 
                     hover:bg-accent rounded-lg transition-colors"
        >
          <Download className="w-4 h-4" />
          Export CSV
        </button>
      </div>
      
      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border bg-muted/50">
              <th className="text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider px-5 py-3">
                Video File
              </th>
              <th className="text-left text-xs font-semibold text-muted-foreground uppercase tracking-wider px-5 py-3">
                Audio File
              </th>
              <th className="text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider px-5 py-3">
                Start Delay
              </th>
              <th className="text-right text-xs font-semibold text-muted-foreground uppercase tracking-wider px-5 py-3">
                End Delay
              </th>
              <th className="text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider px-5 py-3">
                Confidence
              </th>
              <th className="text-center text-xs font-semibold text-muted-foreground uppercase tracking-wider px-5 py-3">
                Status
              </th>
            </tr>
          </thead>
          <tbody>
            {results.map((result) => (
              <tr
                key={result.id}
                className="border-b border-border/50 hover:bg-muted/30 transition-colors"
              >
                <td className="px-5 py-4 text-sm font-medium truncate max-w-[200px]" title={result.videoFile}>
                  {result.videoFile}
                </td>
                <td className="px-5 py-4 text-sm text-muted-foreground truncate max-w-[200px]" title={result.audioFile}>
                  {result.audioFile}
                </td>
                <td className="px-5 py-4 text-sm text-right font-mono">
                  {formatDelay(result.startDelay)}
                </td>
                <td className="px-5 py-4 text-sm text-right font-mono">
                  {formatDelay(result.endDelay)}
                </td>
                <td className="px-5 py-4 text-center">
                  <ConfidenceBadge confidence={result.confidence} />
                </td>
                <td className="px-5 py-4 text-center">
                  {result.status === "success" ? (
                    <CheckCircle className="w-5 h-5 text-success inline-block" />
                  ) : (
                    <XCircle className="w-5 h-5 text-destructive inline-block" />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {/* Summary Footer */}
      <div className="px-5 py-3 bg-muted/30 border-t border-border flex items-center justify-between text-sm">
        <span className="text-muted-foreground">
          {results.length} files processed
        </span>
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5 text-success">
            <CheckCircle className="w-4 h-4" />
            {results.filter(r => r.status === "success").length} success
          </span>
          <span className="flex items-center gap-1.5 text-destructive">
            <XCircle className="w-4 h-4" />
            {results.filter(r => r.status === "error").length} failed
          </span>
        </div>
      </div>
    </div>
  );
};
