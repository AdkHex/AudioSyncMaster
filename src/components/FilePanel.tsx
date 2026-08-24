import { FileAudio, FileVideo, FolderOpen, Music, Trash2, Upload, X } from "lucide-react";
import { memo, useMemo } from "react";

import { Button, Card, CardHeader, IconButton, Pill } from "@/components/ui";
import { cx } from "@/lib/cx";
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

/** One column of the selection step: an empty drop target, or the chosen files
 *  with their probed duration and stream status. */
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

  const totalSize = useMemo(
    () => files.reduce((sum, file) => sum + (file.size ?? 0), 0),
    [files],
  );

  // Empty: a real drop target rather than a bordered box with a sentence in it.
  if (files.length === 0) {
    return (
      <section
        aria-label={title}
        className={cx(
          "flex min-h-[196px] flex-col items-center justify-center gap-2.5 rounded-xl border-[1.5px] border-dashed px-5 py-7 text-center transition-colors",
          dragActive
            ? "border-solid border-primary bg-accent ring-4 ring-primary/20"
            : "border-border-strong bg-card",
        )}
      >
        <span
          className={cx(
            "grid h-[42px] w-[42px] place-items-center rounded-[11px] transition-colors",
            dragActive ? "bg-primary text-primary-foreground" : "bg-sunken text-muted-foreground",
          )}
        >
          {dragActive ? (
            <Upload className="h-5 w-5" aria-hidden />
          ) : (
            <Icon className="h-5 w-5" aria-hidden />
          )}
        </span>

        <h3 className={cx("text-[13px] font-semibold", dragActive && "text-primary")}>
          {dragActive ? `Drop to add ${kind} files` : title}
        </h3>

        {!dragActive && (
          <>
            <p className="max-w-[30ch] text-[11.5px] leading-relaxed text-muted-foreground">
              {hint}
            </p>
            <Button size="sm" onClick={onBrowse} disabled={disabled} className="mt-0.5">
              Browse…
            </Button>
            {recentFolder && (
              <p
                className="max-w-full truncate font-mono text-[11px] text-muted-foreground/70"
                title={recentFolder}
              >
                Last used: {recentFolder}
              </p>
            )}
          </>
        )}
      </section>
    );
  }

  return (
    <Card
      aria-label={title}
      className={cx(
        "flex flex-col transition-colors",
        dragActive && "border-primary ring-4 ring-primary/20",
      )}
    >
      <CardHeader>
        <div className="min-w-0 flex-1">
          <h3 className="text-[13px] font-semibold">{title}</h3>
          <p className="mt-px text-[11.5px] text-muted-foreground">
            {files.length} file{files.length === 1 ? "" : "s"} · {formatSize(totalSize)}
          </p>
        </div>
        <IconButton
          label={`Clear ${title.toLowerCase()}`}
          onClick={onClear}
          disabled={disabled}
          className="hover:text-destructive"
        >
          <Trash2 className="h-3.5 w-3.5" aria-hidden />
        </IconButton>
        <Button size="sm" onClick={onBrowse} disabled={disabled}>
          Browse
        </Button>
      </CardHeader>

      <ul className="flex max-h-[190px] flex-col gap-0.5 overflow-y-auto p-1.5">
        {files.map((file) => {
          const probe = probes[file.path];
          const streamMissing = probe && (kind === "video" ? !probe.hasVideo : !probe.hasAudio);
          return (
            <li
              key={file.id}
              className="group flex items-center gap-2.5 rounded-md px-2.5 py-2 hover:bg-sunken"
            >
              <FileIcon className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />
              <span className="min-w-0 flex-1 truncate text-xs" title={file.name}>
                {file.name}
              </span>

              {probe?.duration != null && (
                <span className="tabular shrink-0 font-mono text-[11px] text-muted-foreground">
                  {formatDuration(probe.duration)}
                </span>
              )}
              <span className="tabular shrink-0 font-mono text-[11px] text-muted-foreground">
                {formatSize(file.size)}
              </span>

              {streamMissing && (
                <Pill tone="destructive">No {kind === "video" ? "video" : "audio"}</Pill>
              )}

              <button
                type="button"
                onClick={() => onRemove(file.id)}
                disabled={disabled}
                title={`Remove ${file.name}`}
                className="shrink-0 rounded p-0.5 text-muted-foreground opacity-0 transition hover:bg-destructive/15 hover:text-destructive focus-visible:opacity-100 group-hover:opacity-100 disabled:opacity-0"
              >
                <X className="h-3 w-3" aria-hidden />
                <span className="sr-only">Remove {file.name}</span>
              </button>
            </li>
          );
        })}
      </ul>

      {folder && (
        <footer
          className="truncate border-t border-border px-4 py-2 font-mono text-[11px] text-muted-foreground"
          title={folder}
        >
          {folder}
        </footer>
      )}
    </Card>
  );
});
