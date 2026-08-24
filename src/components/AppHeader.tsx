import { History as HistoryIcon, Settings2, Terminal } from "lucide-react";
import { memo } from "react";

import { IconButton } from "@/components/ui";
import { cx } from "@/lib/cx";
import type { SyncMode } from "@/lib/types";

interface AppHeaderProps {
  mode: SyncMode;
  onModeChange: (mode: SyncMode) => void;
  disabled: boolean;
  showConsole: boolean;
  showHistory: boolean;
  onToggleConsole: () => void;
  onToggleHistory: () => void;
  onOpenSettings: () => void;
}

const MODES: { id: SyncMode; label: string }[] = [
  { id: "movie", label: "Movies" },
  { id: "series", label: "Series" },
  { id: "compare", label: "Compare" },
];

/** Wordmark, mode switch, and the panels that are not part of the main flow. */
export const AppHeader = memo(function AppHeader({
  mode,
  onModeChange,
  disabled,
  showConsole,
  showHistory,
  onToggleConsole,
  onToggleHistory,
  onOpenSettings,
}: AppHeaderProps) {
  return (
    <header className="flex h-11 shrink-0 items-center gap-3.5 border-b border-border px-3.5">
      <div className="flex items-center gap-2">
        <span className="grid h-5 w-5 place-items-center rounded-md bg-primary text-[10px] font-bold text-primary-foreground">
          A
        </span>
        <span className="text-[13px] font-semibold">AudioSyncMaster</span>
      </div>

      <nav className="ml-1.5 flex gap-0.5" aria-label="Sync mode">
        {MODES.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            aria-current={mode === id ? "page" : undefined}
            onClick={() => onModeChange(id)}
            disabled={disabled}
            className={cx(
              "rounded-[7px] px-3 py-[5px] text-[12.5px] transition-colors disabled:opacity-40",
              mode === id
                ? "bg-elevated font-medium text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {label}
          </button>
        ))}
      </nav>

      <span className="flex-1" />

      <IconButton label="Console" active={showConsole} onClick={onToggleConsole}>
        <Terminal className="h-4 w-4" aria-hidden />
      </IconButton>
      <IconButton
        label="History (Ctrl+H)"
        active={showHistory}
        onClick={onToggleHistory}
      >
        <HistoryIcon className="h-4 w-4" aria-hidden />
      </IconButton>
      <IconButton label="Settings (Ctrl+,)" onClick={onOpenSettings}>
        <Settings2 className="h-4 w-4" aria-hidden />
      </IconButton>
    </header>
  );
});
