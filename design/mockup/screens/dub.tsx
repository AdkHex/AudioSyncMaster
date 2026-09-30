import {
  AddRegular,
  ArrowRedoRegular,
  ArrowRepeatAllRegular,
  ArrowUndoRegular,
  CheckmarkCircleFilled,
  ChevronDoubleLeftRegular,
  ChevronDoubleRightRegular,
  ChevronLeftRegular,
  ChevronRightRegular,
  CircleRegular,
  CopyRegular,
  CutRegular,
  DeleteRegular,
  FolderOpenRegular,
  HeadphonesSoundWaveRegular,
  MergeRegular,
  MusicNote2Regular,
  NextFrameRegular,
  PauseRegular,
  PlayRegular,
  PreviousFrameRegular,
  StopRegular,
  VideoClipRegular,
  ZoomFitRegular,
  ZoomInRegular,
  ZoomOutRegular,
} from "@fluentui/react-icons";
import type { ReactNode } from "react";

import { LOG, PLAN, STAGES, TOTAL } from "../data";
import { Btn, Cmd, Combo, DL, Empty, MidText, Ring, Seg, Status, TBox, Table, Toggle, Tr, Video, Wave, cx, type St } from "../kit";
import { AutoGo, Box, DockTabs, Window, useProgress, useProto, type LcdProps } from "../shell";

const hms = (s: number) => {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}:${String(m).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
};

/* --------------------------------- timeline --------------------------------- */

function Timeline({ t0 = 0, t1 = TOTAL, mode = "done", placed = 99, sel, grips, head }: { t0?: number; t1?: number; mode?: "loaded" | "placing" | "done"; placed?: number; sel?: number; grips?: boolean; head?: number }) {
  const span = t1 - t0;
  const pct = (t: number) => `${((t - t0) / span) * 100}%`;
  const step = [2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800].find((s) => span / s <= 10) ?? 1800;
  const ticks: number[] = [];
  for (let t = Math.ceil(t0 / step) * step; t <= t1 - step * 0.4; t += step) ticks.push(t);
  const dubs = PLAN.map((s, i) => (s.k === "dub" ? i : -1)).filter((i) => i >= 0);
  const zoomed = span < 2000;

  return (
    <div className="tl" style={{ paddingTop: 2 }}>
      <div className="ruler">
        {ticks.map((t) => <span key={t} style={{ left: pct(t) }}>{zoomed ? hms(t) : hms(t).slice(0, -3)}</span>)}
        {ticks.map((t) => <i key={`i${t}`} style={{ left: pct(t) }} />)}
      </div>
      <div className="lane" style={{ flex: 1, minHeight: 56 }}>
        <span className="lh">Original</span>
        <div className="lb">
          <div style={{ position: "absolute", inset: "14% 0" }}><Wave seed={Math.round(t0) + 3} n={zoomed ? 160 : 360} color="var(--wave-a)" /></div>
        </div>
      </div>
      <div className="lane" style={{ flex: 1, minHeight: 56 }}>
        <span className="lh">Dub</span>
        <div className="lb">
          {mode === "loaded" ? (
            <div className="segm dub" style={{ left: 0, width: `${(7810 / TOTAL) * 100}%` }}>
              <div style={{ position: "absolute", inset: "12% 0" }}><Wave seed={31} n={330} color="var(--wave-b)" style={{ opacity: 0.85 }} /></div>
            </div>
          ) : (
            PLAN.map((s, i) => {
              if (s.b <= t0 || s.a >= t1) return null;
              const a = Math.max(s.a, t0);
              const b = Math.min(s.b, t1);
              const n = dubs.indexOf(i);
              const ready = s.k === "dub" ? n < placed : dubs.filter((d) => d < i).length <= placed;
              const cls = mode === "placing" && !ready ? "pending" : s.k === "dub" ? "dub" : "fill";
              const w = (b - a) / span;
              return (
                <div key={i} className={cx("segm", cls, sel === i && "sel")} style={{ left: pct(a), width: `${w * 100}%` }}>
                  {cls === "dub" && (
                    <div style={{ position: "absolute", inset: "12% 0" }}>
                      <Wave seed={i * 7 + Math.round(t0)} n={Math.max(12, Math.round(w * (zoomed ? 160 : 360)))} color="var(--wave-b)" style={{ opacity: 0.85 }} />
                    </div>
                  )}
                </div>
              );
            })
          )}
          {grips && PLAN.slice(1).map((s) => (s.a > t0 && s.a < t1 ? <span key={s.a} className="grip" style={{ left: pct(s.a) }} /> : null))}
        </div>
      </div>
      {head !== undefined && head > t0 && head < t1 && <div className="head-line" style={{ left: `calc(14px + 76px + (100% - 104px) * ${(head - t0) / span})` }} />}
      <div className="minimap">
        {PLAN.filter((s) => s.k === "fill").map((s) => <i key={s.a} style={{ left: `${(s.a / TOTAL) * 100}%`, width: `max(2px, ${((s.b - s.a) / TOTAL) * 100}%)` }} />)}
        <b style={{ left: `${(t0 / TOTAL) * 100}%`, width: `${(span / TOTAL) * 100}%` }} />
      </div>
    </div>
  );
}

function Stages({ cur, pct }: { cur: number; pct: number }) {
  return (
    <div className="col">
      {STAGES.map((s, i) => (
        <div key={s} className="row" style={{ gap: 10, height: 29, color: i > cur ? "var(--text-3)" : undefined }}>
          <span style={{ width: 16, display: "grid", fontSize: 16 }}>
            {i < cur ? <CheckmarkCircleFilled className="ok" /> : i === cur ? <Ring size={16} /> : <CircleRegular style={{ color: "var(--text-4)" }} />}
          </span>
          <span className="grow truncate">{s}</span>
          {i === cur && <span className="t3 num">{Math.round(pct)}%</span>}
        </div>
      ))}
    </div>
  );
}

/* --------------------------------- screens --------------------------------- */

type DPhase = "empty" | "ready" | "run" | "done" | "edit" | "preview" | "season" | "failed";

const SEASON: { v: string; d: string; err?: string; s: St; t?: string; pct?: number }[] = [
  { v: "Goblin.S01E01.1080p.BluRay.mkv", d: "Goblin.S01E01.hin.eac3", err: "0.2 ms", s: "ok", t: "Done" },
  { v: "Goblin.S01E02.1080p.BluRay.mkv", d: "Goblin.S01E02.hin.eac3", err: "0.3 ms", s: "ok", t: "Done" },
  { v: "Goblin.S01E03.1080p.BluRay.mkv", d: "Goblin.S01E03.hin.eac3", s: "run", t: "Placing the cuts", pct: 42 },
  { v: "Goblin.S01E04.1080p.BluRay.mkv", d: "Goblin.S01E04.hin.eac3", s: "wait" },
  { v: "Goblin.S01E05.1080p.BluRay.mkv", d: "Goblin.S01E05.hin.eac3", s: "wait" },
];

export function DubSync({ phase }: { phase: DPhase }) {
  const { go } = useProto();
  const p = useProgress(45, 6000);
  const single = phase !== "season" && phase !== "failed";
  const running = phase === "run" || phase === "season";
  const cur = phase === "run" ? Math.min(8, Math.floor(p / 11.2)) : 3;
  const placed = phase === "run" ? (cur < 3 ? 0 : cur === 3 ? Math.round(((p - 33.6) / 11.2) * 6) : 99) : phase === "season" ? 3 : 99;

  const jobs = single
    ? [{ v: "Skyline Heist (2023) 1080p BluRay.mkv", d: "Skyline Heist (2023) Hindi TV rip.ac3", err: phase === "ready" || phase === "run" ? undefined : "0.2 ms", s: (phase === "ready" ? "ready" : phase === "run" ? "run" : "ok") as St, t: phase === "run" ? STAGES[cur] : phase === "ready" ? "Ready" : "Done", pct: p }]
    : phase === "failed"
      ? SEASON.map((j, i) => (i === 2 ? { ...j, s: "ok" as St, t: "Done · 1 stretch to check", err: "0.4 ms" } : i === 3 ? { ...j, s: "bad" as St, t: "Different edit" } : i === 4 ? { ...j, s: "ok" as St, t: "Done", err: "0.2 ms" } : j))
      : SEASON;
  const selRow = phase === "season" ? 2 : phase === "failed" ? 3 : 0;

  const lcd: LcdProps =
    phase === "empty"
      ? { l1: "Drop movies and their dubs" }
      : phase === "ready"
        ? { l1: "1 pair ready", l2: "The dub is 10m 20s shorter than the video" }
        : phase === "run"
          ? { icon: "run", l1: STAGES[cur], l2: "Skyline Heist (2023)", pct: p, time: `${Math.round(p)}%` }
          : phase === "season"
            ? { icon: "run", l1: "Syncing 3 of 5 · Placing the cuts", l2: "Goblin · Season 1 · 3 at a time", pct: 41, time: "6 min left" }
            : phase === "failed"
              ? { icon: "bad", l1: "4 synced · 1 failed", l2: "Goblin.S01E04: the dub is a different edit" }
              : phase === "edit"
                ? { icon: "warn", l1: "1 change not applied", l2: "Dub stretch 4 of 6 · 1:21:23.000 – 1:49:05.600" }
                : { icon: "ok", l1: "On the lips — 0.2 ms typical, 0.4 ms at worst", l2: "6 stretches · 5 cuts filled from the original" };

  let inspector: ReactNode;
  if (phase === "ready")
    inspector = (
      <Box title="Output">
        <div className="col" style={{ gap: 6 }}><span className="t3">Write the track as</span><Combo value="Same as the dub (AC3 5.1)" w="100%" /></div>
        <div className="col" style={{ gap: 6 }}><span className="t3">The dub was mastered at</span><Combo value="Detect from the audio" w="100%" /></div>
        <div className="row"><span className="grow">Add to a copy of the video</span><Toggle on /></div>
        <div className="row"><span className="grow t3">Language</span><TBox value="hin" w={80} /></div>
      </Box>
    );
  else if (running)
    inspector = (
      <Box title={phase === "run" ? "Skyline Heist (2023)" : "Goblin.S01E03"}>
        <Stages cur={cur} pct={phase === "run" ? (p % 11.2) * 9 : 42} />
      </Box>
    );
  else if (phase === "edit")
    inspector = (
      <Box title="Dub stretch 4 of 6">
        <DL rows={[["Start", "1:21:23.000"], ["End", "1:49:05.600"], ["Length", "27m 42.6s"], ["Taken from", "Dub 1:14:42.356"], ["Match", "0.13"]]} />
        <div className="col" style={{ gap: 6 }}>
          <span className="t3">Offset</span>
          <div className="row" style={{ gap: 2 }}>
            <Cmd icon={<ChevronDoubleLeftRegular />} title="−1 frame" />
            <Cmd icon={<ChevronLeftRegular />} title="−10 ms" />
            <TBox value="−400.644" unit="s" w={116} focus />
            <Cmd icon={<ChevronRightRegular />} title="+10 ms" />
            <Cmd icon={<ChevronDoubleRightRegular />} title="+1 frame" />
          </div>
        </div>
        <div className="links">
          <button type="button" className="cmd"><span className="ic"><CutRegular /></span>Split here</button>
          <button type="button" className="cmd"><span className="ic"><VideoClipRegular /></span>Use the original</button>
        </div>
      </Box>
    );
  else if (phase === "preview")
    inspector = (
      <Box title="Preview" end={<span className="t3 num sm" style={{ paddingRight: 6 }}>1:18:44:19</span>}>
        <Video />
        <div className="row" style={{ gap: 2, marginTop: -8 }}>
          <Cmd icon={<PauseRegular />} title="Pause (Space)" />
          <Cmd icon={<PreviousFrameRegular />} title="Back one frame" />
          <Cmd icon={<NextFrameRegular />} title="Forward one frame" />
          <Cmd icon={<ArrowRepeatAllRegular />} title="Loop 4 s" />
        </div>
        <div className="col" style={{ gap: 6 }}>
          <span className="t3">Listen to</span>
          <Seg items={["Dub", "Original", "Both"]} value={2} />
        </div>
      </Box>
    );
  else if (phase === "failed")
    inspector = (
      <Box title="Goblin.S01E04">
        <div><div className="t3">Result</div><div className="big bad">Not synced</div><div className="t2">The dub ends 38 minutes before the video. It is probably a different edit.</div></div>
        <div className="row" style={{ gap: 8 }}><Btn>Retry</Btn><Btn>Choose another dub</Btn></div>
      </Box>
    );
  else if (phase !== "empty")
    inspector = (
      <Box title="Skyline Heist (2023)">
        <div><div className="t3">Sync error</div><div className="big">0.2<small>ms</small></div><div className="t2">Typical · 0.4 ms at worst</div></div>
        <DL rows={[["Stretches", "6 of dub"], ["From original", "10m 22.2s · 5 cuts"], ["Fill level", "+4.4 dB"], ["Frame rate", "23.976 fps, both"], ["Voices moved", "2 scenes"]]} />
        <div className="links">
          <button type="button" className="cmd" onClick={() => go("dub-preview")}><span className="ic"><PlayRegular /></span>Play</button>
          <button type="button" className="cmd"><span className="ic"><FolderOpenRegular /></span>Show</button>
          <button type="button" className="cmd"><span className="ic"><CopyRegular /></span>Report</button>
        </div>
      </Box>
    );

  const dock =
    phase === "empty" ? undefined : phase === "failed" ? (
      <div style={{ height: 250, display: "flex", flexDirection: "column" }}>
        <DockTabs tabs={["Timeline", "Output", "History"]} on="Output" onTab={(t) => go(t === "History" ? "history" : "dub-done")} />
        <div className="log">
          {LOG.map(([t, tone, m], i) => <div key={i}><span className="t">{t}</span><span className={tone || undefined}>{m}</span></div>)}
        </div>
      </div>
    ) : (
      <div style={{ height: phase === "season" ? 250 : 300, display: "flex", flexDirection: "column" }}>
        <DockTabs
          tabs={["Timeline", "Output", "History"]}
          on="Timeline"
          onTab={(t) => go(t === "Output" ? "dub-failed" : t === "History" ? "history" : "dub-done")}
          tools={
            <>
              {phase === "edit" && <span className="t2 sm" style={{ marginRight: 8 }}>1 change</span>}
              {phase === "edit" && <Btn accent style={{ height: 28, marginRight: 8 }} onClick={() => go("dub-done")}>Apply</Btn>}
              <Cmd sm icon={<ArrowUndoRegular />} title="Undo" />
              <Cmd sm icon={<ArrowRedoRegular />} title="Redo" />
              <span className="vsep" />
              <Cmd sm icon={<CutRegular />} title="Split (S)" />
              <Cmd sm icon={<MergeRegular />} title="Merge with next" />
              <span className="vsep" />
              <Cmd sm icon={<ZoomOutRegular />} title="Zoom out" />
              <Cmd sm icon={<ZoomInRegular />} title="Zoom in" onClick={() => go("dub-edit")} />
              <Cmd sm icon={<ZoomFitRegular />} title="Whole film" onClick={() => go("dub-done")} />
              <span className="vsep" />
              <Cmd sm icon={<PlayRegular />} title="Play (Space)" onClick={() => go("dub-preview")} />
            </>
          }
        />
        {phase === "edit" ? (
          <Timeline t0={4650} t1={4960} sel={7} grips head={4790} />
        ) : (
          <Timeline mode={phase === "ready" ? "loaded" : (phase === "run" && cur < 4) || phase === "season" ? "placing" : "done"} placed={placed} head={phase === "preview" ? 4724.8 : undefined} />
        )}
      </div>
    );

  return (
    <Window
      page="dub"
      lcd={lcd}
      cols={phase === "empty" ? "minmax(0,1fr)" : "minmax(0,1fr) 300px"}
      dock={dock}
      util={phase === "failed" ? "output" : undefined}
      tools={
        <>
          <Cmd icon={<AddRegular />} disabled={running} onClick={() => go("dub-ready")}>Add movies</Cmd>
          <Cmd icon={<MusicNote2Regular />} disabled={running}>Add dubs</Cmd>
          {phase !== "empty" && <Cmd icon={<DeleteRegular />} disabled={running} title="Remove" />}
        </>
      }
      primary={
        running ? (
          <Btn icon={<StopRegular />} kbd="Esc" onClick={() => go("dub-ready")}>Stop</Btn>
        ) : (
          <Btn accent icon={<PlayRegular />} kbd="Enter" disabled={phase === "empty"} onClick={() => go("dub-run")}>{phase === "ready" || phase === "empty" ? "Sync" : "Sync again"}</Btn>
        )
      }
    >
      {phase === "empty" ? (
        <section className="box">
          <Empty icon={<HeadphonesSoundWaveRegular />} title="Drop movies and their dubs">
            <Btn icon={<AddRegular />} onClick={() => go("dub-ready")}>Add movies</Btn>
            <Btn icon={<MusicNote2Regular />} onClick={() => go("dub-ready")}>Add dubs</Btn>
          </Empty>
        </section>
      ) : (
        <>
          <Box body={false} title="Queue" sub={single ? "1 pair" : "5 pairs"}>
            <Table cols="minmax(0,1fr) 84px 190px" head={["Pair", " Sync error", "Status"]}>
              {jobs.map((j, i) => (
                <Tr key={j.v} on={i === selRow} onClick={() => go(j.s === "bad" ? "dub-failed" : "dub-done")} style={{ height: 48 }}>
                  <span className="col" style={{ minWidth: 0, gap: 1 }}>
                    <span className="cell"><span className="fi"><VideoClipRegular /></span><MidText text={j.v} /></span>
                    <span className="cell t3 sm"><span className="fi" style={{ color: "var(--text-3)" }}><MusicNote2Regular /></span><MidText text={j.d} /></span>
                  </span>
                  <span className="r num" style={{ display: "flex" }}>{j.err ?? <span className="t3">—</span>}</span>
                  <Status s={j.s} text={j.t} pct={j.pct} />
                </Tr>
              ))}
            </Table>
          </Box>
          {inspector}
        </>
      )}
      {phase === "run" && <AutoGo to="dub-done" ms={6300} />}
    </Window>
  );
}
