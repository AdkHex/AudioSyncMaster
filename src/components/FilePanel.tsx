import { X } from "lucide-react";
import { memo, useMemo } from "react";

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

/** One input group in the sidebar.
 *
 *  A plain list rather than a bordered card: the heading and the spacing
 *  already group these rows, so a box around them is a third device doing the
 *  same job. */
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
  const totalSize = useMemo(
    () => files.reduce((sum, file) => sum + (file.size ?? 0), 0),
    [files],
  );

  return (
    <section aria-label={title}>
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h2 className="text-[11px] font-semibold text-muted-foreground">{title}</h2>
        {files.length > 0 ? (
          <div className="flex items-baseline gap-2.5">
            <button
              type="button"
              onClick={onClear}
              disabled={disabled}
              className="text-[11.5px] text-muted-foreground transition-colors hover:text-foreground disabled:opacity-40"
            >
              Clear
            </button>
            <button
              type="button"
              onClick={onBrowse}
              disabled={disabled}
              className="text-[11.5px] font-medium text-primary transition-opacity hover:opacity-80 disabled:opacity-40"
            >
              Change
            </button>
          </div>
        ) : null}
      </div>

      {files.length === 0 ? (
        <button
          type="button"
          onClick={onBrowse}
          disabled={disabled}
          className={cx(
            "w-full rounded-[9px] border border-dashed px-3.5 py-6 text-center text-xs transition-colors disabled:opacity-40",
            dragActive
              ? "border-primary/60 bg-primary/5 text-primary"
              : "border-border-strong text-muted-foreground hover:border-muted-foreground/40 hover:text-foreground",
          )}
        >
          {dragActive ? `Drop to add ${kind} files` : hint}
        </button>
      ) : (
        <>
          <ul className="flex flex-col">
            {files.map((file) => {
              const probe = probes[file.path];
              const missing =
                probe && (kind === "video" ? !probe.hasVideo : !probe.hasAudio);

              return (
                <li key={file.id} className="group flex items-baseline gap-2 py-[3px]">
                  <span
                    className="min-w-0 flex-1 truncate text-[12.5px]"
                    title={file.name}
                  >
                    {file.name}
                  </span>

                  {missing ? (
                    <span className="shrink-0 text-[11px] text-destructive">
                      no {kind === "video" ? "video" : "audio"}
                    </span>
                  ) : probe?.duration ? (
                    <span className="tabular shrink-0 font-mono text-[11px] text-muted-foreground">
                      {formatDuration(probe.duration)}
                    </span>
                  ) : null}

                  <span className="tabular shrink-0 font-mono text-[11px] text-muted-foreground">
                    {formatSize(file.size)}
                  </span>

                  <button
                    type="button"
                    onClick={() => onRemove(file.id)}
                    disabled={disabled}
                    className="-mr-1 shrink-0 rounded p-0.5 text-muted-foreground opacity-0 transition-opacity hover:text-foreground focus-visible:opacity-100 group-hover:opacity-100 disabled:opacity-0"
                  >
                    <X className="h-3 w-3" aria-hidden />
                    <span className="sr-only">Remove {file.name}</span>
                  </button>
                </li>
              );
            })}
          </ul>

          <p className="mt-1.5 flex items-baseline justify-between gap-2 font-mono text-[10.5px] text-muted-foreground">
            <span className="min-w-0 truncate" title={folder ?? undefined}>
              {folder ?? ""}
            </span>
            {files.length > 1 && (
              <span className="tabular shrink-0">{formatSize(totalSize)}</span>
            )}
          </p>
        </>
      )}

      {files.length === 0 && recentFolder && (
        <p
          className="mt-2 truncate font-mono text-[10.5px] text-muted-foreground"
          title={recentFolder}
        >
          {recentFolder}
        </p>
      )}
    </section>
  );
});
