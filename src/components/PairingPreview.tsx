import { AlertTriangle, ArrowRight, Pencil, RotateCcw, X } from "lucide-react";
import { memo, useState } from "react";

import { Button, Card, CardHeader, Notice, Pill, Spinner } from "@/components/ui";
import type { FileItem, PairingReport } from "@/lib/types";

interface PairingPreviewProps {
  pairing: PairingReport | null;
  loading: boolean;
  /** Audio the user has selected, offered as alternatives for a wrong match. */
  audioFiles: FileItem[];
  /** Videos the user has selected. Needed to resolve an unmatched file's full
   *  path: the report carries only its basename, and an unmatched video is by
   *  definition absent from the pair list. */
  videoFiles: FileItem[];
  /** Repoint a video at a different audio track, or null to exclude it. */
  onRepair: (videoPath: string, audioPath: string | null) => void;
  /** Drop every hand edit and go back to what the matcher produced. */
  onResetRepairs: () => void;
  manualCount: number;
  disabled?: boolean;
}

/** Shows exactly what will be compared, how each pair was arrived at, and lets
 *  a wrong one be corrected before any work starts.
 *
 *  Matching is a guess. Showing the user a mistake and then offering no way to
 *  fix it left renaming files on disk as the only recourse. */
export const PairingPreview = memo(function PairingPreview({
  pairing,
  loading,
  audioFiles,
  videoFiles,
  onRepair,
  onResetRepairs,
  manualCount,
  disabled = false,
}: PairingPreviewProps) {
  const [editing, setEditing] = useState<string | null>(null);

  if (loading) {
    return (
      <Card className="flex items-center gap-2.5 px-4 py-3">
        <Spinner className="h-3.5 w-3.5" />
        <p className="text-xs text-muted-foreground">Working out pairings…</p>
      </Card>
    );
  }

  if (!pairing) return null;

  const { pairs, unmatchedPrimary, unmatchedSecondary, method, warning } = pairing;

  return (
    <div className="space-y-3">
      {warning && (
        <Notice icon={<AlertTriangle className="h-3.5 w-3.5" aria-hidden />}>{warning}</Notice>
      )}

      <Card>
        <CardHeader>
          <h2 className="text-[13px] font-semibold">Pairing preview</h2>
          <span className="flex-1" />
          {manualCount > 0 && (
            <Button variant="ghost" size="sm" onClick={onResetRepairs} disabled={disabled}>
              <RotateCcw className="h-3 w-3" aria-hidden />
              Reset {manualCount} edit{manualCount === 1 ? "" : "s"}
            </Button>
          )}
          <span className="text-[11px] text-muted-foreground">matched by {method}</span>
          <Pill>
            {pairs.length} pair{pairs.length === 1 ? "" : "s"}
          </Pill>
        </CardHeader>

        {pairs.length > 0 && (
          <ul className="max-h-72 divide-y divide-border overflow-y-auto">
            {pairs.map((pair) => {
              const isEditing = editing === pair.primaryPath;
              const manual = pair.method === "chosen by hand";

              return (
                <li key={pair.primaryPath} className="group px-4 py-2">
                  <div className="flex items-center gap-2 text-xs">
                    <span className="min-w-0 flex-1 truncate" title={pair.primaryName}>
                      {pair.primaryName}
                    </span>
                    <ArrowRight className="h-3 w-3 shrink-0 text-muted-foreground" aria-hidden />

                    {isEditing ? (
                      <select
                        autoFocus
                        value={pair.secondaryPath}
                        onChange={(event) => {
                          onRepair(pair.primaryPath, event.target.value || null);
                          setEditing(null);
                        }}
                        onBlur={() => setEditing(null)}
                        className="min-w-0 flex-1 rounded-md border border-border bg-card px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-primary"
                        aria-label={`Audio for ${pair.primaryName}`}
                      >
                        {audioFiles.map((file) => (
                          <option key={file.path} value={file.path}>
                            {file.name}
                          </option>
                        ))}
                        <option value="">— skip this video —</option>
                      </select>
                    ) : (
                      <span
                        className="min-w-0 flex-1 truncate text-muted-foreground"
                        title={pair.secondaryName}
                      >
                        {pair.secondaryName}
                      </span>
                    )}

                    {manual && !isEditing && <Pill tone="success">by hand</Pill>}
                    {pair.score < 1 && !manual && !isEditing && (
                      <Pill
                        tone="warning"
                        title="Paired by filename similarity rather than an episode number"
                      >
                        {Math.round(pair.score * 100)}%
                      </Pill>
                    )}

                    {!isEditing && (
                      <button
                        type="button"
                        onClick={() => setEditing(pair.primaryPath)}
                        disabled={disabled || audioFiles.length === 0}
                        title="Pair this video with a different audio track"
                        className="shrink-0 rounded p-1 text-muted-foreground opacity-0 transition hover:bg-sunken hover:text-foreground focus-visible:opacity-100 disabled:opacity-0 group-hover:opacity-100"
                      >
                        <Pencil className="h-3 w-3" aria-hidden />
                        <span className="sr-only">Change the audio for {pair.primaryName}</span>
                      </button>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        )}

        {(unmatchedPrimary.length > 0 || unmatchedSecondary.length > 0) && (
          <div className="space-y-2 border-t border-border px-4 py-2.5 text-[11px]">
            {unmatchedPrimary.length > 0 && (
              <div>
                <p className="mb-1 text-destructive">
                  {unmatchedPrimary.length} video
                  {unmatchedPrimary.length === 1 ? "" : "s"} with no audio match
                </p>
                <ul className="space-y-1">
                  {unmatchedPrimary.slice(0, 6).map((name) => (
                    <li key={name} className="flex items-center gap-2">
                      <span className="min-w-0 flex-1 truncate text-muted-foreground">
                        {name}
                      </span>
                      <UnmatchedPicker
                        name={name}
                        audioFiles={audioFiles}
                        disabled={disabled}
                        onPick={(audioPath) => {
                          const match = videoFiles.find((file) => file.name === name);
                          if (match) onRepair(match.path, audioPath);
                        }}
                      />
                    </li>
                  ))}
                </ul>
                {unmatchedPrimary.length > 6 && (
                  <p className="mt-1 text-muted-foreground">
                    and {unmatchedPrimary.length - 6} more
                  </p>
                )}
              </div>
            )}
            {unmatchedSecondary.length > 0 && (
              <p className="text-muted-foreground">
                {unmatchedSecondary.length} audio file
                {unmatchedSecondary.length === 1 ? "" : "s"} unused
              </p>
            )}
          </div>
        )}
      </Card>
    </div>
  );
});

/** Pair a video the matcher gave up on. */
function UnmatchedPicker({
  name,
  audioFiles,
  disabled,
  onPick,
}: {
  name: string;
  audioFiles: FileItem[];
  disabled: boolean;
  onPick: (audioPath: string) => void;
}) {
  const [open, setOpen] = useState(false);

  if (audioFiles.length === 0) return null;

  if (!open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        disabled={disabled}
        className="shrink-0 rounded px-1.5 py-0.5 text-[10.5px] text-primary transition hover:bg-sunken disabled:opacity-40"
      >
        Pair manually
      </button>
    );
  }

  return (
    <span className="flex shrink-0 items-center gap-1">
      <select
        autoFocus
        defaultValue=""
        onChange={(event) => {
          if (event.target.value) onPick(event.target.value);
          setOpen(false);
        }}
        onBlur={() => setOpen(false)}
        className="rounded-md border border-border bg-card px-2 py-1 text-[11px] focus:outline-none focus:ring-1 focus:ring-primary"
        aria-label={`Audio for ${name}`}
      >
        <option value="" disabled>
          Choose audio…
        </option>
        {audioFiles.map((file) => (
          <option key={file.path} value={file.path}>
            {file.name}
          </option>
        ))}
      </select>
      <button
        type="button"
        onClick={() => setOpen(false)}
        className="rounded p-0.5 text-muted-foreground hover:text-foreground"
      >
        <X className="h-3 w-3" aria-hidden />
        <span className="sr-only">Cancel</span>
      </button>
    </span>
  );
}
