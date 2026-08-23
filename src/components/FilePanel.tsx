import { FileAudio, FileVideo, FolderOpen, Music, Trash2, X } from "lucide-react";
import { memo, useMemo } from "react";

import { formatDuration, formatSize, type FileItem, type MediaProbe } from "@/lib/types";

interface FilePanelProps {
  kind: "video" | "audio";
  title: string;
  hint: string;
  files: FileItem[];
  folder: string | null;
  recentFolder: string | null;
  probes: Record<string, MediaProbe>;
  dragActive: boolean;
  disabled: boolean;
  onBrowse: () => void;
  onRemove: (id: string) => void;
  onClear: () => void;
}

/** One column of the selection area: the chosen files plus their stream status. */
export const FilePanel = memo(function FilePanel({
  kind,
  title,
  hint,
  files,
  folder,
  recentFolder,
  probes,
  dragActive,
  disabled,
  onBrowse,
  onRemove,
  onClear,
}: FilePanelProps) {
  const Icon = kind === "video" ? FolderOpen : Music;
  const FileIcon = kind === "video" ? FileVideo : FileAudio;
  const accent = kind === "video" ? "text-warning" : "text-success";

  const totalSize = useMemo(
    () => files.reduce((sum, file) => sum + (file.size ?? 0), 0),
    [files],
  );

  return (
    <section
      aria-label={title}
      className={`relative flex flex-col rounded-lg border bg-card transition-colors ${
        dragActive ? "border-primary ring-2 ring-primary/40" : "border-border"
      }`}
    >
      <header className="flex items-center justify-between gap-2 border-b border-border px-4 py-3">
        <div className="flex min-w-0 items-center gap-2">
          <Icon className={`h-4 w-4 shrink-0 ${accent}`} aria-hidden />
          <div className="min-w-0">
            <h2 className="text-sm font-medium text-foreground">{title}</h2>
            <p className="truncate text-[11px] text-muted-foreground">
              {files.length > 0
                ? `${files.length} file${files.length === 1 ? "" : "s"} · ${formatSize(totalSize)}`
                : hint}
            </p>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {files.length > 0 && (
            <button
              type="button"
              onClick={onClear}
              disabled={disabled}
              className="rounded p-1.5 text-muted-foreground transition-colors hover:bg-secondary hover:text-destructive disabled:opacity-40"
              title={`Clear ${title.toLowerCase()}`}
            >
              <Trash2 className="h-3.5 w-3.5" aria-hidden />
              <span className="sr-only">Clear {title}</span>
            </button>
          )}
          <button
            type="button"
            onClick={onBrowse}
            disabled={disabled}
            className="rounded border border-border px-2.5 py-1 text-xs font-medium text-foreground transition-colors hover:bg-secondary disabled:opacity-40"
          >
            Browse
          </button>
        </div>
      </header>

      <div className="min-h-[120px] flex-1 p-2">
        {files.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-1 px-4 py-8 text-center">
            <p className="text-xs text-muted-foreground">
              Drop files here, or click Browse
            </p>
            {recentFolder && (
              <p className="max-w-full truncate text-[10px] text-muted-foreground/70">
                Last used: {recentFolder}
              </p>
            )}
          </div>
        ) : (
          <ul className="max-h-64 space-y-1 overflow-y-auto pr-1">
            {files.map((file) => {
              const probe = probes[file.path];
              const streamMissing =
                probe && (kind === "video" ? !probe.hasVideo : !probe.hasAudio);
              return (
                <li
                  key={file.id}
                  className="group flex items-center gap-2 rounded bg-secondary/50 px-2 py-1.5"
                >
                  <FileIcon className={`h-3.5 w-3.5 shrink-0 ${accent}`} aria-hidden />
                  <span className="min-w-0 flex-1 truncate text-xs text-foreground" title={file.name}>
                    {file.name}
                  </span>
                  {probe?.duration != null && (
                    <span className="shrink-0 font-mono text-[10px] text-muted-foreground">
                      {formatDuration(probe.duration)}
                    </span>
                  )}
                  <span className="shrink-0 text-[10px] text-muted-foreground">
                    {formatSize(file.size)}
                  </span>
                  {streamMissing && (
                    <span className="shrink-0 rounded bg-destructive/15 px-1.5 py-0.5 text-[10px] text-destructive">
                      No {kind === "video" ? "video" : "audio"}
                    </span>
                  )}
                  <button
                    type="button"
                    onClick={() => onRemove(file.id)}
                    disabled={disabled}
                    className="shrink-0 rounded p-0.5 opacity-0 transition-opacity hover:bg-destructive/20 focus-visible:opacity-100 group-hover:opacity-100 disabled:opacity-0"
                    title={`Remove ${file.name}`}
                  >
                    <X className="h-3 w-3 text-muted-foreground" aria-hidden />
                    <span className="sr-only">Remove {file.name}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {folder && (
        <footer className="truncate border-t border-border px-4 py-1.5 text-[10px] text-muted-foreground" title={folder}>
          {folder}
        </footer>
      )}

      {dragActive && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center rounded-lg bg-primary/10 text-xs font-medium text-primary">
          Drop to add {kind} files
        </div>
      )}
    </section>
  );
});
