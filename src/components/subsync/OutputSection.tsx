import { useId } from "react";

import { CheckRow, INPUT_CLASS, LinkButton, SectionTitle, SelectBox } from "@/components/subsync/controls";
import { cx } from "@/lib/cx";
import { FORMAT_LABELS, outputFields } from "@/lib/subsync/tools";
import type { OutputFormat, OutputOptions, SubTask } from "@/lib/subsync/types";

/** Where results go and what they are written as. The source is never
 *  touched: results land beside it, or in a chosen folder. */
export function OutputSection({
  task,
  output,
  disabled,
  onChange,
  onChooseFolder,
}: {
  task: SubTask;
  output: OutputOptions;
  disabled: boolean;
  onChange: (patch: Partial<OutputOptions>) => void;
  onChooseFolder: () => void;
}) {
  const ids = useId();
  const fields = outputFields(task);
  const writesVideo = task === "tonemap" || task === "mux";

  return (
    <section aria-labelledby={`${ids}-title`}>
      <SectionTitle id={`${ids}-title`}>Output</SectionTitle>

      <div className="flex items-baseline gap-2">
        <span className="text-[11.5px] text-muted-foreground">Save to</span>
        <span className="min-w-0 flex-1 truncate font-mono text-[11px]" title={output.dir ?? undefined}>
          {output.dir ?? "Beside each source"}
        </span>
        <LinkButton tone="primary" disabled={disabled} onClick={onChooseFolder}>
          Choose…
        </LinkButton>
        {output.dir && (
          <LinkButton disabled={disabled} onClick={() => onChange({ dir: null })}>
            Reset
          </LinkButton>
        )}
      </div>

      {fields.formats && (
        <>
          <label htmlFor={`${ids}-format`} className="mt-3 block text-[11.5px] text-muted-foreground">
            Format
          </label>
          <SelectBox
            id={`${ids}-format`}
            className="mt-1"
            value={output.format ?? fields.formats[0]}
            disabled={disabled}
            onChange={(event) => onChange({ format: event.target.value as OutputFormat })}
          >
            {fields.formats.map((format) => (
              <option key={format} value={format}>
                {FORMAT_LABELS[format]}
              </option>
            ))}
          </SelectBox>
        </>
      )}

      <div className="mt-3 flex items-center gap-2.5">
        <label htmlFor={`${ids}-suffix`} className="text-[11.5px] text-muted-foreground">
          Name suffix
        </label>
        <input
          id={`${ids}-suffix`}
          type="text"
          value={output.suffix ?? ""}
          disabled={disabled}
          spellCheck={false}
          placeholder="none"
          onChange={(event) => onChange({ suffix: event.target.value })}
          className={cx(INPUT_CLASS, "w-28 font-mono")}
        />
      </div>
      <p className="mt-1 text-[10.5px] leading-snug text-muted-foreground/80">
        Added to the file name before the language, e.g. Movie{output.suffix || ""}.en.
        {writesVideo ? "mkv" : "srt"}
      </p>

      <CheckRow
        className="mt-3"
        label="Replace files that already exist"
        hint="Off: a file already there is never replaced."
        checked={!!output.overwrite}
        disabled={disabled}
        onChange={(overwrite) => onChange({ overwrite })}
      />

      {fields.mux && (
        <>
          <CheckRow
            className="mt-2.5"
            label="Also write a copy of the video with this subtitle added"
            hint="Needs the video beside each subtitle. Every original stream is kept."
            checked={!!output.mux}
            disabled={disabled}
            onChange={(mux) => onChange({ mux })}
          />
          {output.mux && (
            <div className="ml-[23px] mt-2 flex flex-col gap-1.5">
              <label className="flex items-center gap-2 text-[11.5px] text-muted-foreground">
                Track title
                <input
                  type="text"
                  value={output.muxTitle ?? ""}
                  placeholder="optional"
                  disabled={disabled}
                  onChange={(event) => onChange({ muxTitle: event.target.value })}
                  className={cx(INPUT_CLASS, "min-w-0 flex-1")}
                />
              </label>
              <CheckRow label="Default track" checked={!!output.muxDefault} disabled={disabled} onChange={(muxDefault) => onChange({ muxDefault })} />
              <CheckRow label="Forced track" checked={!!output.muxForced} disabled={disabled} onChange={(muxForced) => onChange({ muxForced })} />
            </div>
          )}
        </>
      )}
    </section>
  );
}
