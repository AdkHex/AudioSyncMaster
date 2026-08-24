import { memo, useId } from "react";

import type { AudioTrackInfo } from "@/lib/types";

interface TrackSelectorProps {
  label: string;
  tracks: AudioTrackInfo[];
  value: number;
  onChange: (index: number) => void;
  disabled?: boolean;
}

/** Choose which audio stream of a file to compare.
 *
 *  Hidden when a file has only one: a dropdown offering a single option is a
 *  control that cannot be used. */
export const TrackSelector = memo(function TrackSelector({
  label,
  tracks,
  value,
  onChange,
  disabled = false,
}: TrackSelectorProps) {
  const id = useId();

  if (tracks.length <= 1) return null;

  return (
    <div>
      <label htmlFor={id} className="mb-1.5 block text-[11.5px] text-muted-foreground">
        {label}
      </label>
      <select
        id={id}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className="w-full appearance-none rounded-lg bg-elevated px-2.5 py-2 pr-7 text-[12.5px] text-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50"
        style={{
          // Inline so the chevron colour tracks the token rather than being a
          // second, drifting copy of it in the stylesheet.
          backgroundImage:
            "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23969696' stroke-width='2.5'><path d='m6 9 6 6 6-6'/></svg>\")",
          backgroundRepeat: "no-repeat",
          backgroundPosition: "right 9px center",
        }}
      >
        {tracks.map((track) => (
          <option key={track.index} value={track.index}>
            {track.label}
          </option>
        ))}
      </select>
    </div>
  );
});
