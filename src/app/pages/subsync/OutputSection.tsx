/** Where results go and what they are written as. The source is never
 *  touched: results land beside it, or in a chosen folder. */

import { DEFAULT_OUTPUT, FORMAT_LABELS, outputFields } from "@/lib/subsync/tools";
import type { OutputFormat, OutputOptions, SubTask } from "@/lib/subsync/types";
import { Chk, Combo, TBox, Toggle } from "@/ui/kit";

import { Advanced, Section } from "./settings/fields";

const CHOOSE = "\u0000choose";

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
  const fields = outputFields(task);
  const writesVideo = task === "tonemap" || task === "mux";
  const example = `Movie${output.suffix || ""}.en.${writesVideo ? "mkv" : "srt"}`;
  // Tasks that name their output by language (OCR, Translate, Generate…)
  // have no suffix by default; theirs waits under More options.
  const suffixShown = !!DEFAULT_OUTPUT[task].suffix;
  const suffixRow = (
    <div className="row ss-inline" title={`Added to the name before the language: ${example}`}>
      <span className="grow t3">Name suffix</span>
      <TBox
        mono
        w={110}
        label="Name suffix"
        value={output.suffix ?? ""}
        placeholder="none"
        spellCheck={false}
        disabled={disabled}
        onChange={(suffix) => onChange({ suffix })}
      />
    </div>
  );
  const changed =
    (!suffixShown && !!output.suffix) ||
    (!!fields.formats && !!output.format && output.format !== fields.formats[0]) || !!output.overwrite || !!output.mux;

  return (
    <Section title="Output">
      <label className="ss-fld">
        <span className="t3">Save to</span>
        <Combo<string>
          w="100%"
          label="Save to"
          disabled={disabled}
          value={output.dir ?? ""}
          options={[
            { value: "", label: "Beside each source" },
            ...(output.dir ? [{ value: output.dir, label: output.dir }] : []),
            { value: CHOOSE, label: "Choose a folder…" },
          ]}
          onChange={(value) => {
            if (value === CHOOSE) onChooseFolder();
            else onChange({ dir: value || null });
          }}
        />
      </label>

      {suffixShown && suffixRow}

      {/* The mockup's Output is where and under what name; how, below. */}
      <Advanced open={changed}>
        {!suffixShown && suffixRow}
        {fields.formats && (
          <label className="ss-fld">
            <span className="t3">Format</span>
            <Combo<OutputFormat>
              w="100%"
              label="Format"
              disabled={disabled}
              value={output.format ?? fields.formats[0]}
              options={fields.formats.map((format) => ({ value: format, label: FORMAT_LABELS[format] }))}
              onChange={(format) => onChange({ format })}
            />
          </label>
        )}

        <div className="row ss-tgl" title="Off: a file already there is never replaced.">
          <span className="grow">Replace existing files</span>
          <Toggle name="Replace existing files" on={!!output.overwrite} disabled={disabled} onChange={(overwrite) => onChange({ overwrite })} />
        </div>

        {fields.mux && (
          <>
            <div className="row ss-tgl" title="Needs the video beside each subtitle. Every original stream is kept.">
              <span className="grow">Add to a copy of the video</span>
              <Toggle name="Add to a copy of the video" on={!!output.mux} disabled={disabled} onChange={(mux) => onChange({ mux })} />
            </div>
            {output.mux && (
              <div className="ss-fld ss-indent">
                <div className="row ss-inline">
                  <span className="grow t3">Track title</span>
                  <TBox
                    w={150}
                    label="Track title"
                    value={output.muxTitle ?? ""}
                    placeholder="Optional"
                    disabled={disabled}
                    onChange={(muxTitle) => onChange({ muxTitle })}
                  />
                </div>
                <div className="row" style={{ gap: 16 }}>
                  <Chk on={!!output.muxDefault} disabled={disabled} onChange={(muxDefault) => onChange({ muxDefault })}>
                    Default track
                  </Chk>
                  <Chk on={!!output.muxForced} disabled={disabled} onChange={(muxForced) => onChange({ muxForced })}>
                    Forced
                  </Chk>
                </div>
              </div>
            )}
          </>
        )}
      </Advanced>
    </Section>
  );
}
