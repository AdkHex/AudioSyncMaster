import { FPS_PRESETS, type ExtractOptions, type MuxOptions } from "@/lib/subsync/types";

import { Field, Hint, Section, Segmented, Select, Toggle } from "./fields";
import { fpsPreset, parseIndexList } from "./logic";
import type { SettingsPanelProps } from "./props";

export function ConvertPanel({ options, onChange, disabled, videoFps }: SettingsPanelProps<"convert">) {
  const fps = options.fps ?? null;
  const preset = fps === null ? null : fpsPreset(fps);
  const value = fps === null ? "auto" : preset ? String(FPS_PRESETS.indexOf(preset)) : "other";
  return (
    <Section>
      <Field
        label="Frame rate for MicroDVD"
        hint="MicroDVD (.sub) counts frames, not time, so reading or writing it needs the video's frame rate. Other formats ignore this."
      >
        <Select<string>
          value={value}
          disabled={disabled}
          onChange={(v) => onChange({ fps: v === "auto" ? null : FPS_PRESETS[Number(v)].value })}
          choices={[
            {
              value: "auto",
              label: videoFps ? `From the file or video (${Math.round(videoFps * 1000) / 1000} fps)` : "From the file or video",
            },
            ...(value === "other" && fps !== null ? [{ value: "other", label: `${fps} fps` }] : []),
            ...FPS_PRESETS.map((p, i) => ({ value: String(i), label: `${p.label} fps` })),
          ]}
        />
      </Field>
    </Section>
  );
}

export function ExtractPanel({ options, onChange, disabled }: SettingsPanelProps<"extract">) {
  const all = options.tracks === "all";
  const listed = Array.isArray(options.tracks) ? options.tracks : [];
  return (
    <Section>
      <Field group label="Tracks" hint={all ? "Every subtitle track in the video, each to its own file." : "Only the tracks listed below."}>
        <Segmented<"all" | "selected">
          label="Tracks"
          value={all ? "all" : "selected"}
          disabled={disabled}
          onChange={(v) => onChange({ tracks: v === "all" ? "all" : listed.length ? listed : [0] })}
          choices={[
            { value: "all", label: "All" },
            { value: "selected", label: "Selected" },
          ]}
        />
      </Field>
      {!all && (
        <Field label="Track numbers" hint="Subtitle tracks counting from 1, separated by commas: 1, 3.">
          <input
            type="text"
            key={listed.join(",")}
            defaultValue={listed.map((n) => n + 1).join(", ")}
            aria-label="Track numbers"
            placeholder="1, 3"
            disabled={disabled}
            spellCheck={false}
            onBlur={(event) => {
              const parsed = parseIndexList(event.target.value);
              const valid = parsed !== null && parsed.every((n) => n >= 1);
              event.target.setCustomValidity(valid ? "" : "Use track numbers from 1, separated by commas");
              if (valid) onChange({ tracks: parsed.map((n) => n - 1) as ExtractOptions["tracks"] });
            }}
            className="w-full rounded-md border border-border-strong bg-input px-2 py-1 font-mono text-xs focus:outline-none focus:ring-2 focus:ring-ring/40 disabled:opacity-50 invalid:border-destructive"
          />
        </Field>
      )}
      {!all && <Hint>The track list of each video shows its numbers.</Hint>}
    </Section>
  );
}

export function MuxPanel({ options, onChange, disabled }: SettingsPanelProps<"mux">) {
  return (
    <Section>
      <Field
        group
        label="Container"
        hint={
          options.container === "mkv"
            ? "MKV holds every subtitle format, styles included."
            : "MP4 stores text subtitles as plain mov_text; styling and image subtitles are lost."
        }
      >
        <Segmented<MuxOptions["container"]>
          label="Container"
          value={options.container}
          disabled={disabled}
          onChange={(container) => onChange({ container })}
          choices={[
            { value: "mkv", label: "MKV" },
            { value: "mp4", label: "MP4" },
          ]}
        />
      </Field>
      <Toggle
        label="Keep the video's existing subtitles"
        hint="Off replaces them with the added tracks. Video and audio are always kept."
        checked={options.keepExisting}
        disabled={disabled}
        onChange={(keepExisting) => onChange({ keepExisting })}
      />
    </Section>
  );
}
