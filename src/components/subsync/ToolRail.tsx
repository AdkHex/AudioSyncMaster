import {
  AudioLines,
  Captions,
  FileStack,
  Gauge,
  Languages,
  ScanText,
  Sparkles,
  SunDim,
  Timer,
  type LucideIcon,
} from "lucide-react";
import { memo } from "react";

import { Spinner } from "@/components/ui";
import { cx } from "@/lib/cx";
import { TOOLS, type ToolId } from "@/lib/subsync/tools";

const ICONS: Record<ToolId, LucideIcon> = {
  sync: Timer,
  ocr: ScanText,
  translate: Languages,
  fps: Gauge,
  generate: AudioLines,
  style: Captions,
  hdrSubs: SunDim,
  tonemap: Sparkles,
  formats: FileStack,
};

/** The list of tools. Navigation only: nothing here starts work, so the
 *  rail never holds an action a user could lose off the bottom edge. */
export const ToolRail = memo(function ToolRail({
  tool,
  runningTool,
  onSelect,
}: {
  tool: ToolId;
  runningTool: ToolId | null;
  onSelect: (tool: ToolId) => void;
}) {
  return (
    <nav aria-label="Subtitle tools" className="flex w-[212px] shrink-0 flex-col border-r border-border">
      <ul className="min-h-0 flex-1 overflow-y-auto px-2 py-3">
        {TOOLS.map((entry) => {
          const Icon = ICONS[entry.id];
          const active = entry.id === tool;
          return (
            <li key={entry.id}>
              <button
                type="button"
                aria-current={active ? "page" : undefined}
                title={entry.description}
                onClick={() => onSelect(entry.id)}
                className={cx(
                  "flex w-full items-start gap-2.5 rounded-[7px] px-2.5 py-[7px] text-left transition-colors",
                  "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                  active ? "bg-elevated text-foreground" : "text-muted-foreground hover:bg-secondary hover:text-foreground",
                )}
              >
                <Icon className={cx("mt-px h-4 w-4 shrink-0", active && "text-primary")} aria-hidden />
                <span className="min-w-0 flex-1">
                  <span className={cx("block text-[12.5px]", active && "font-medium")}>{entry.label}</span>
                  {active && (
                    <span className="mt-0.5 block text-[10.5px] leading-snug text-muted-foreground">
                      {entry.description}
                    </span>
                  )}
                </span>
                {runningTool === entry.id && (
                  <Spinner className="mt-0.5 h-3 w-3 border-[1.5px]" />
                )}
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
});
