/** The queue: every movie with its dub, the error the check measured, and
 *  where the pair is -- Ready, Waiting, a progress bar and its stage, Done,
 *  or why it failed. A row picks the pair the Timeline and the details
 *  show.
 *
 *  Before a run, a dub can be chosen by hand: an unpaired movie shows a
 *  "Choose a dub" box, and a paired one's dub name opens the same list
 *  (the chevron shows on hover), so the automatic pairing can always be
 *  corrected. */

import { ChevronDownRegular, MusicNote2Regular, VideoClipRegular } from "@fluentui/react-icons";
import { memo, type KeyboardEvent } from "react";

import type { FileItem } from "@/lib/types";
import { Box } from "@/ui/frame";
import { Combo, MidText, Status, Table, Tr, type Option } from "@/ui/kit";

import { plural, rowStatus, syncError, type QueueRow } from "./dubModel";

/** What a dub choice means besides a file. */
export const SKIP = "\u0000skip";
export const AUTO = "\u0000auto";

export type Repair = (videoPath: string, choice: string | null | "auto") => void;

export const DubQueue = memo(function DubQueue({
  rows,
  selected,
  onSelect,
  dubs,
  onRepair,
}: {
  rows: QueueRow[];
  selected: string | null;
  onSelect: (key: string) => void;
  /** Every dub added, to choose from. */
  dubs: FileItem[];
  /** Pairing by hand; absent while a run holds the queue. */
  onRepair?: Repair;
}) {
  const pairs = rows.filter((r) => r.paired).length;
  return (
    <Box body={false} title="Queue" sub={pairs > 0 ? plural(pairs, "pair") : undefined} label="Queue">
      <Table cols="minmax(0,1fr) 84px 190px" head={["Pair", " Sync error", "Status"]} label="Dub sync queue">
        {rows.map((row) => {
          const status = rowStatus(row, dubs.length);
          const error = syncError(row.job);
          return (
            <Tr
              key={row.key}
              on={row.key === selected}
              onClick={() => onSelect(row.key)}
              style={{ height: 48 }}
              label={row.videoName ?? row.dubName ?? undefined}
            >
              <span className="col" style={{ minWidth: 0, gap: 1 }}>
                {row.videoName ? (
                  <>
                    <span className="cell">
                      <span className="fi" aria-hidden>
                        <VideoClipRegular />
                      </span>
                      <MidText text={row.videoName} />
                    </span>
                    <DubLine row={row} dubs={dubs} onRepair={onRepair} />
                  </>
                ) : (
                  <>
                    <span className="cell">
                      <span className="fi" aria-hidden>
                        <MusicNote2Regular />
                      </span>
                      <MidText text={row.dubName ?? ""} />
                    </span>
                    <span className="cell t3 sm">No movie yet</span>
                  </>
                )}
              </span>
              <span className="r num" style={{ display: "flex" }}>
                {error ?? <span className="t3">—</span>}
              </span>
              <span className="dubq-st" title={status.title}>
                <Status s={status.s} text={status.text} pct={status.pct} />
              </span>
            </Tr>
          );
        })}
      </Table>
    </Box>
  );
});

/** The dub under a movie: its name, or a box to choose one. */
function DubLine({ row, dubs, onRepair }: { row: QueueRow; dubs: FileItem[]; onRepair?: Repair }) {
  const videoPath = row.videoPath!;
  const options: Option<string>[] = [
    ...dubs.map((dub) => ({ value: dub.path, label: dub.name })),
    { value: SKIP, label: "Skip this movie" },
    ...(row.manual ? [{ value: AUTO, label: "Pair automatically" }] : []),
  ];
  const choose = (value: string) => onRepair?.(videoPath, value === SKIP ? null : value === AUTO ? "auto" : value);
  // The row's own keys (Enter, Space pick the row) must not reach it from
  // the list, or the list could not be opened from the keyboard.
  const keep = (event: KeyboardEvent) => event.stopPropagation();

  if (!row.dubName || !row.dubPath) {
    if (!onRepair || dubs.length === 0) return <span className="cell t3 sm">{row.pending ? "Pairing…" : "No dub"}</span>;
    return (
      <span className="cell" onKeyDown={keep}>
        <Combo<string> sm placeholder="Choose a dub" value={null} options={options} w={200} label={`Dub for ${row.videoName}`} onChange={choose} />
      </span>
    );
  }
  return (
    <span className="cell t3 sm dubq-dub">
      <span className="fi" style={{ color: "var(--text-3)" }} aria-hidden>
        <MusicNote2Regular />
      </span>
      <MidText text={row.dubName} />
      {onRepair && (
        <>
          <span className="dubq-chev" aria-hidden>
            <ChevronDownRegular />
          </span>
          <select
            className="dubq-native"
            aria-label={`Dub for ${row.videoName}`}
            value={row.dubPath}
            onKeyDown={keep}
            onChange={(event) => choose(event.target.value)}
          >
            {!dubs.some((d) => d.path === row.dubPath) && <option value={row.dubPath}>{row.dubName}</option>}
            {options.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </>
      )}
    </span>
  );
}
