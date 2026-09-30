import { useState } from "react";

import type { AsrEngine, AsrModel, MediaRef, ProbedFile, SyncEngine } from "@/lib/subsync/types";
import { Btn } from "@/ui/kit";

import {
  Advanced,
  EnginePicker,
  Field,
  Hint,
  HintLink,
  NumberField,
  Panel,
  Section,
  Segmented,
  Select,
  Slider,
  Toggle,
  type EngineOption,
} from "./fields";
import { ASR_MODELS, asrModelLabel, engineStatus, findPack, installAction, syncEnginePatch } from "./logic";
import type { SettingsPanelProps } from "./props";

const SYNC_ENGINES: EngineOption<SyncEngine>[] = [
  {
    id: "audio",
    label: "Speech in the video",
    description: "Matches lines to when people talk in the video's own audio. Any language.",
    needs: "only the video",
  },
  {
    id: "subtitle",
    label: "A correctly timed subtitle",
    description: "Copies the timing of another subtitle for this video, in any language.",
    needs: "a reference subtitle file, or a video with a subtitle track",
  },
  {
    id: "reference",
    label: "Audio of the original release",
    description: "Compares this video's audio with the release the subtitle was timed for. Best for cut or extended edits.",
    needs: "the release the subtitle was made for (its video or audio)",
  },
  {
    id: "transcript",
    label: "Speech recognition",
    description: "Recognises the dialogue and aligns line by line. The subtitle must be in the spoken language.",
    needs: "the speech pack; slower than the others",
  },
];

function basename(path: string): string {
  return path.split(/[\\/]/).pop() || path;
}

/** Chooses the reference file and, from its probe, which stream to use. The
 *  probe is kept locally only to list tracks; the options store the path. */
function ReferencePicker({
  engine,
  reference,
  disabled,
  pickFile,
  onChange,
}: {
  engine: "subtitle" | "reference";
  reference: MediaRef | null | undefined;
  disabled: boolean;
  pickFile: SettingsPanelProps<"sync">["pickFile"];
  onChange: (reference: MediaRef | null) => void;
}) {
  const [probed, setProbed] = useState<ProbedFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const info = probed && reference && probed.path === reference.path ? probed : null;

  const choose = async () => {
    const file = await pickFile(engine === "subtitle" ? "any" : "video");
    if (!file) return;
    if (file.error) {
      setError(file.error);
      return;
    }
    setError(null);
    setProbed(file);
    if (engine === "subtitle") {
      const firstText = file.subtitleTracks.find((t) => t.kind === "text") ?? file.subtitleTracks[0];
      onChange({ path: file.path, track: file.kind === "subtitle" ? null : (firstText?.index ?? null) });
    } else {
      onChange({ path: file.path, audioTrack: file.audioTracks[0]?.index ?? 0 });
    }
  };

  const label = engine === "subtitle" ? "Reference subtitle" : "Reference release";
  return (
    <Field
      group
      label={label}
      hint={
        engine === "subtitle"
          ? "A subtitle file, or a video whose subtitle track is already in sync. Its language does not matter."
          : "The video or audio the subtitle was originally timed to."
      }
    >
      <div className="row" style={{ gap: 8 }}>
        <span className="tbox grow" title={reference?.path}>
          {reference?.path ? (
            <span className="truncate">{basename(reference.path)}</span>
          ) : (
            <span className="t3">None chosen</span>
          )}
        </span>
        <Btn disabled={disabled} onClick={() => void choose()}>
          Choose…
        </Btn>
        {reference?.path && (
          <Btn disabled={disabled} onClick={() => onChange(null)}>
            Clear
          </Btn>
        )}
      </div>
      {error && <Hint tone="destructive">{error}</Hint>}
      {!reference?.path && <Hint tone="warning">Choose a reference before running.</Hint>}

      {reference?.path && engine === "subtitle" && info && info.subtitleTracks.length > 0 && (
        <Select
          ariaLabel="Subtitle track"
          value={String(reference.track ?? info.subtitleTracks[0].index)}
          disabled={disabled}
          onChange={(v) => onChange({ ...reference, track: Number(v) })}
          choices={info.subtitleTracks.map((t) => ({
            value: String(t.index),
            label: [
              `Track ${t.index + 1}`,
              t.language,
              t.title,
              t.codec,
              t.kind === "image" ? "image, needs OCR first" : null,
              t.forced ? "forced" : null,
            ]
              .filter(Boolean)
              .join(" · "),
            disabled: t.kind === "image",
          }))}
        />
      )}
      {reference?.path && engine === "reference" && info && info.audioTracks.length > 1 && (
        <Select
          ariaLabel="Audio track"
          value={String(reference.audioTrack ?? 0)}
          disabled={disabled}
          onChange={(v) => onChange({ ...reference, audioTrack: Number(v) })}
          choices={info.audioTracks.map((t) => ({
            value: String(t.index),
            label: [`Audio ${t.index + 1}`, t.language, t.title, t.codec, t.channels ? `${t.channels} ch` : null]
              .filter(Boolean)
              .join(" · "),
          }))}
        />
      )}
      {reference?.path && !info && (engine === "subtitle" ? reference.track != null : (reference.audioTrack ?? 0) > 0) && (
        <Hint>
          Using {engine === "subtitle" ? `subtitle track ${(reference.track ?? 0) + 1}` : `audio track ${(reference.audioTrack ?? 0) + 1}`} of
          this file. Choose it again to pick another track.
        </Hint>
      )}
    </Field>
  );
}

export function SyncPanel({ options, onChange, caps, disabled, onInstallPack, pickFile }: SettingsPanelProps<"sync">) {
  const statuses = Object.fromEntries(SYNC_ENGINES.map((e) => [e.id, engineStatus(caps, "sync", e.id)]));
  const vadStatus = { energy: engineStatus(caps, "vad", "energy"), silero: engineStatus(caps, "vad", "silero") };
  const sileroAction = installAction(vadStatus.silero);
  const asrEngine: AsrEngine = options.asrEngine ?? "mlx-whisper";
  const asrStatus = engineStatus(caps, "generate", asrEngine);
  const asrPack = findPack(caps, asrStatus?.pack);

  return (
    <Panel>
      <EnginePicker
        packs={caps?.packs}
        title="Match against"
        label="Sync engine"
        value={options.engine}
        options={SYNC_ENGINES}
        statuses={statuses}
        loading={caps === null}
        disabled={disabled}
        onChange={(engine) => onChange(syncEnginePatch(options, engine))}
        onInstallPack={onInstallPack}
      />

      {(options.engine === "subtitle" || options.engine === "reference") && (
        <ReferencePicker
          engine={options.engine}
          reference={options.reference}
          disabled={disabled}
          pickFile={pickFile}
          onChange={(reference) => onChange({ reference })}
        />
      )}

      {options.engine === "transcript" && (
        <Section title="Speech recognition">
          <Field group label="Recogniser" hint="mlx-whisper runs on the Apple Silicon GPU; faster-whisper runs anywhere.">
            <Segmented<AsrEngine>
              label="Recogniser"
              value={asrEngine}
              disabled={disabled}
              onChange={(v) => onChange({ asrEngine: v })}
              choices={[
                {
                  value: "mlx-whisper",
                  label: "mlx-whisper",
                  title: engineStatus(caps, "generate", "mlx-whisper")?.reason ?? "Apple Silicon",
                  disabled: engineStatus(caps, "generate", "mlx-whisper")?.available === false,
                },
                {
                  value: "faster-whisper",
                  label: "faster-whisper",
                  title: engineStatus(caps, "generate", "faster-whisper")?.reason ?? "Any OS",
                  disabled: engineStatus(caps, "generate", "faster-whisper")?.available === false,
                },
              ]}
            />
          </Field>
          {asrStatus && !asrStatus.available && (
            <Hint tone="warning">
              {asrStatus.reason || "Not installed."}{" "}
              {asrStatus.pack && <HintLink onClick={() => onInstallPack(asrStatus.pack!)}>Install</HintLink>}
            </Hint>
          )}
          <Field label="Model" hint="Bigger models hear more words right; they take longer and a larger download.">
            <Select<AsrModel>
              value={options.asrModel ?? "large-v3-turbo"}
              disabled={disabled}
              onChange={(asrModel) => onChange({ asrModel })}
              choices={ASR_MODELS.map((m) => ({ value: m.id, label: asrModelLabel(m, asrPack) }))}
            />
          </Field>
        </Section>
      )}

      <Section title="Timing">
        <Toggle
          label="Detect frame-rate changes"
          hint="Also tries 23.976, 24, 25, 29.97 and 30 fps conversions and keeps the best. Fixes lines that drift further off over time."
          checked={options.detectFramerate}
          disabled={disabled}
          onChange={(detectFramerate) => onChange({ detectFramerate })}
        />
        <Toggle
          label="Different offset per scene"
          hint="For cut or extended editions, where the subtitle is right in some scenes and off in others."
          checked={options.allowSplits}
          disabled={disabled}
          onChange={(allowSplits) => onChange({ allowSplits })}
        />
        {options.allowSplits && (
          <Field
            label="Split penalty"
            hint="Higher means fewer splits, only where the evidence is very clear. Lower finds more cuts but may split on noise."
          >
            <Slider
              ariaLabel="Split penalty"
              min={1}
              max={20}
              step={0.5}
              value={options.splitPenalty}
              display={String(options.splitPenalty)}
              disabled={disabled}
              onChange={(splitPenalty) => onChange({ splitPenalty })}
            />
          </Field>
        )}
      </Section>

      <Advanced>
        <Field
          inline
          label="Largest shift to search"
          hint="The subtitle may be off by up to this much. Larger finds bigger offsets but takes longer and risks a wrong match."
        >
          <NumberField
            ariaLabel="Largest shift to search"
            value={options.maxOffsetS}
            min={1}
            max={3600}
            step={5}
            unit="s"
            width={100}
            disabled={disabled}
            onChange={(v) => v !== null && onChange({ maxOffsetS: v })}
          />
        </Field>
        {options.engine === "audio" && (
          <Field
            group
            label="Speech detector"
            hint="Energy is built in and fast. Silero is a neural detector that ignores music and effects better."
          >
            <Segmented<"energy" | "silero">
              label="Speech detector"
              value={options.vad}
              disabled={disabled}
              onChange={(vad) => onChange({ vad })}
              choices={[
                { value: "energy", label: "Energy", title: "Built in" },
                {
                  value: "silero",
                  label: "Silero",
                  title: vadStatus.silero?.reason ?? "Neural speech detector",
                  disabled: vadStatus.silero?.available === false && options.vad !== "silero",
                },
              ]}
            />
            {vadStatus.silero && !vadStatus.silero.available && (
              <Hint tone="warning">
                Silero: {vadStatus.silero.reason || "not installed."}{" "}
                {sileroAction && (
                  <HintLink onClick={() => onInstallPack(sileroAction.target)}>{sileroAction.label}</HintLink>
                )}
              </Hint>
            )}
          </Field>
        )}
      </Advanced>
    </Panel>
  );
}
