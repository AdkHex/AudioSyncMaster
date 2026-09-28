import { ExternalLink, FolderOpen } from "lucide-react";
import { useState } from "react";

import { CueTable } from "@/components/subsync/CueTable";
import { LinkButton, SelectBox } from "@/components/subsync/controls";
import { formatReportValue, reportLabel } from "@/lib/subsync/format";
import { TASK_LABELS, useAsTargets } from "@/lib/subsync/tools";
import type { SubJobOutcome, SubTask, TaskOutputFile } from "@/lib/subsync/types";

export interface OutcomeActions {
  onReveal: (path: string) => void;
  onOpen: (path: string) => void;
  onUseAs: (output: TaskOutputFile, task: SubTask) => void;
}

function baseName(path: string): string {
  return path.replace(/^.*[\\/]/, "");
}

/** What one finished job did: its summary, anything worth a second look,
 *  the numbers behind it, the files it wrote, and the cues themselves. */
export function OutcomeView({ outcome, logs, actions }: { outcome: SubJobOutcome; logs: string[]; actions: OutcomeActions }) {
  const [showLog, setShowLog] = useState(false);
  const report = Object.entries(outcome.report ?? {}).filter(([, value]) => typeof value !== "object" || value === null || Array.isArray(value));

  return (
    <div className="flex flex-col gap-3">
      {outcome.summary && <p className="text-[12.5px] leading-relaxed">{outcome.summary}</p>}

      {outcome.warnings.length > 0 && (
        <ul className="flex flex-col gap-1 text-[11.5px] leading-snug text-warning" aria-label="Warnings">
          {outcome.warnings.map((warning, index) => (
            <li key={index}>{warning}</li>
          ))}
        </ul>
      )}

      {report.length > 0 && (
        <dl className="grid grid-cols-[minmax(120px,max-content)_minmax(0,1fr)] gap-x-4 gap-y-0.5 text-[11.5px]">
          {report.map(([key, value]) => (
            <div key={key} className="contents">
              <dt className="text-muted-foreground">{reportLabel(key)}</dt>
              <dd className="tabular min-w-0 truncate font-mono text-[11px]" title={formatReportValue(key, value)}>
                {formatReportValue(key, value)}
              </dd>
            </div>
          ))}
        </dl>
      )}

      {outcome.outputs.length > 0 && (
        <ul className="flex flex-col gap-1" aria-label="Files written">
          {outcome.outputs.map((output) => (
            <OutputRow key={output.path} output={output} actions={actions} />
          ))}
        </ul>
      )}

      {outcome.preview && outcome.preview.cues.length > 0 && (
        <CueTable cues={outcome.preview.cues} total={outcome.preview.count} label={`Cues of ${outcome.id}`} />
      )}

      {logs.length > 0 && (
        <div>
          <LinkButton aria-expanded={showLog} onClick={() => setShowLog((open) => !open)}>
            {showLog ? "Hide" : "Show"} the log ({logs.length} line{logs.length === 1 ? "" : "s"})
          </LinkButton>
          {showLog && (
            <pre className="mt-1.5 max-h-48 overflow-auto rounded-md bg-sunken px-2.5 py-2 font-mono text-[10.5px] leading-relaxed text-muted-foreground">
              {logs.join("\n")}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

function OutputRow({ output, actions }: { output: TaskOutputFile; actions: OutcomeActions }) {
  const targets = useAsTargets(output);
  const name = baseName(output.path);
  return (
    <li className="flex flex-wrap items-center gap-x-3 gap-y-1">
      <span className="min-w-0 flex-1 truncate font-mono text-[11.5px]" title={output.path}>
        {output.label ? `${output.label}: ` : ""}
        {name}
      </span>
      <LinkButton className="flex items-center gap-1" onClick={() => actions.onReveal(output.path)} aria-label={`Show ${name} in its folder`}>
        <FolderOpen className="h-3 w-3" aria-hidden />
        Reveal
      </LinkButton>
      {output.kind !== "folder" && (
        <LinkButton className="flex items-center gap-1" onClick={() => actions.onOpen(output.path)} aria-label={`Open ${name}`}>
          <ExternalLink className="h-3 w-3" aria-hidden />
          Open
        </LinkButton>
      )}
      {targets.length > 0 && (
        <SelectBox
          aria-label={`Use ${name} as input for another tool`}
          value=""
          onChange={(event) => {
            if (event.target.value) actions.onUseAs(output, event.target.value as SubTask);
          }}
          className="w-auto py-0.5 text-[11px]"
        >
          <option value="">Use as input for…</option>
          {targets.map((task) => (
            <option key={task} value={task}>
              {TASK_LABELS[task]}
            </option>
          ))}
        </SelectBox>
      )}
    </li>
  );
}
