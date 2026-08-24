import { Layers } from "lucide-react";
import { memo, useId } from "react";

import { Pill } from "@/components/ui";
import type { AudioTrackInfo } from "@/lib/types";

interface TrackSelectorProps {
  label: string;
  tracks: AudioTrackInfo[];
  value: number;
  onChange: (index: number) => void;
  disabled?: boolean;
  /** Frame rate of the file, shown because a mismatch explains steady drift. */
  fps?: number | null;
}

/** Choose which audio stream of a file to compare.
 *
 *  Only shown when a file actually has more than one: a single-track file has
 *  no choice to make, and a dropdown with one option is just noise.
 */
export const TrackSelector = memo(function TrackSelector({
  label,
  tracks,
  value,
  onChange,
  disabled = false,
  fps,
}: TrackSelectorProps) {
  const id = useId();

  if (tracks.length <= 1) {
    if (!fps) return null;
    return (
      <div className="flex items-center gap-2 text-[11.5px] text-muted-foreground">
        <span>{label}</span>
        <Pill tone="neutral">{fps.toFixed(3).replace(/\.?0+$/, "")} fps</Pill>
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <label
        htmlFor={id}
        className="flex items-center gap-1.5 text-[11.5px] font-medium text-foreground"
      >
        <Layers className="h-3.5 w-3.5 text-muted-foreground" aria-hidden />
        {label}
      </label>
      <select
        id={id}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className="min-w-0 flex-1 rounded-lg border border-border bg-sunken px-2.5 py-1.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50"
      >
        {tracks.map((track) => (
          <option key={track.index} value={track.index}>
            {track.label}
            {track.isDefault ? " — default" : ""}
          </option>
        ))}
      </select>
      {fps ? (
        <Pill tone="neutral">{fps.toFixed(3).replace(/\.?0+$/, "")} fps</Pill>
      ) : null}
    </div>
  );
});
