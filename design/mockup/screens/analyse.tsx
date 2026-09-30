import {
  AddRegular,
  ArrowExportRegular,
  ArrowRightRegular,
  ArrowSyncRegular,
  CheckmarkCircleFilled,
  CopyRegular,
  DeleteRegular,
  DismissRegular,
  FilterRegular,
  FolderOpenRegular,
  FolderRegular,
  MusicNote2Regular,
  PlayRegular,
  StopRegular,
  VideoClipMultipleRegular,
  VideoClipRegular,
} from "@fluentui/react-icons";
import type { ReactNode } from "react";

import { DUB, EPISODES, MATCH, MOVIES, type Movie } from "../data";
import { Btn, Chk, Cmd, Combo, DL, Empty, Meter, MidText, Ring, Seg, Status, Table, Tr, Wave, cx, type St } from "../kit";
import { AutoGo, Box, Window, useProgress, useProto, type LcdProps } from "../shell";
import { HistoryDock } from "./history";

/* ------------------------------- shared ------------------------------- */

function Name({ children, icon = <VideoClipRegular />, tail = 18 }: { children: string; icon?: ReactNode; tail?: number }) {
  return (
    <span className="cell">
      <span className="fi">{icon}</span>
      <MidText text={children} tail={tail} />
    </span>
  );
}

function Conf({ c }: { c?: number }) {
  if (!c) return <span className="t3">—</span>;
  return (
    <span className="cell num">
      <Meter pct={c} />
      {c}%
    </span>
  );
}

const Dash = () => <span className="t3">—</span>;
const WIN = [0.08, 0.24, 0.4, 0.56, 0.72, 0.88];

function Align() {
  return (
    <div className="col" style={{ gap: 2 }}>
      <div style={{ height: 32 }}><Wave seed={5} color="var(--wave-a)" /></div>
      <div style={{ height: 32 }}><Wave seed={5} color="var(--wave-b)" /></div>
    </div>
  );
}

function Chart({ cut }: { cut?: boolean }) {
  const W = 268, H = 110;
  const X = (x: number) => 30 + x * (W - 36);
  const pts = cut ? WIN.map((x, i) => [x, i < 2 ? 30 : i === 2 ? 58 : 86]) : WIN.map((x, i) => [x, 90 - x * 70 + (i === 2 ? 10 : 0)]);
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
      {[20, 55, 90].map((y) => <line key={y} x1="30" x2={W} y1={y} y2={y} stroke="var(--divider)" />)}
      <text x="24" y="24" fontSize="10" textAnchor="end" fill="var(--text-3)">{cut ? "−0.3" : "+5.3"}</text>
      <text x="24" y="94" fontSize="10" textAnchor="end" fill="var(--text-3)">{cut ? "+462" : "+1.2"}</text>
      <text x="30" y={H - 2} fontSize="10" fill="var(--text-3)">0:00</text>
      <text x={W} y={H - 2} fontSize="10" textAnchor="end" fill="var(--text-3)">2:43</text>
      {cut ? (
        <>
          <line x1={X(0.4)} x2={X(0.4)} y1="10" y2="96" stroke="var(--warn)" strokeDasharray="3 3" />
          <line x1={X(0)} x2={X(0.4)} y1="30" y2="30" stroke="var(--text-2)" strokeWidth="1.5" />
          <line x1={X(0.4)} x2={X(1)} y1="86" y2="86" stroke="var(--text-2)" strokeWidth="1.5" />
        </>
      ) : (
        <>
          <line x1={X(0)} x2={X(1)} y1="90" y2="90" stroke="var(--text-4)" strokeDasharray="3 3" />
          <line x1={X(0)} x2={X(1)} y1="90" y2="20" stroke="var(--warn)" strokeWidth="1.5" />
        </>
      )}
      {pts.map(([x, y], i) => <circle key={i} cx={X(x)} cy={y} r="3" fill={i === 2 && !cut ? "var(--layer)" : "var(--text)"} stroke="var(--text)" />)}
    </svg>
  );
}

const Links = ({ children }: { children: ReactNode }) => <div className="links">{children}</div>;
const L = ({ icon, children, onClick }: { icon: ReactNode; children: ReactNode; onClick?: () => void }) => (
  <button type="button" className="cmd" onClick={onClick}><span className="ic">{icon}</span>{children}</button>
);

/* ================================ MOVIES ================================ */

export type MPhase = "empty" | "drag" | "ready" | "run" | "done" | "writing" | "written";

const statusOf = (m: Movie): [St, string] =>
  m.kind === "ok" || m.kind === "fps" ? ["ok", "Matched"] : m.kind === "drift" ? ["warn", "Drifting"] : m.kind === "cut" ? ["warn", "Different cut"] : ["bad", "No audio"];

function MovieDetails({ m, phase, p }: { m: Movie; phase: MPhase; p: number }) {
  const { go } = useProto();
  const title = <MidText text={m.name} tail={22} />;
  if (phase === "ready")
    return (
      <Box title={title}>
        <DL rows={[["Duration", m.dur], ["Frame rate", `${m.fps} fps`], ["Size", m.size], ["Audio track", <Combo key="a" sm value="1 · English · TrueHD 7.1" w="100%" />]]} />
      </Box>
    );
  if (phase === "run") {
    const w = Math.min(6, Math.floor((p / 100) * 7));
    return (
      <Box title={title}>
        <div className="row"><span className="grow">Measuring</span><span className="t3 num">window {Math.max(1, w)} of 6</span></div>
        <div style={{ position: "relative", height: 64 }}>
          <Wave seed={9} color="var(--wave-a)" />
          {WIN.map((x, i) => (
            <span key={x} style={{ position: "absolute", top: 0, bottom: 0, left: `${x * 100}%`, width: "6%", borderRadius: 2, background: i < w ? "color-mix(in srgb, var(--accent) 28%, transparent)" : "transparent", boxShadow: "inset 0 0 0 1px var(--divider)" }} />
          ))}
        </div>
      </Box>
    );
  }
  if (m.kind === "fail")
    return (
      <Box title={title}>
        <div><div className="t3">Result</div><div className="big bad">Failed</div><div className="t2">The video has no audio track.</div></div>
      </Box>
    );
  if (m.kind === "cut")
    return (
      <Box title={title}>
        <div><div className="t3">Result</div><div className="big">Different cut</div><div className="t2">A scene is missing near 1:02:10.</div></div>
        <Chart cut />
        <Btn icon={<ArrowRightRegular />} onClick={() => go("dub-ready")}>Open in Dub sync</Btn>
      </Box>
    );
  return (
    <Box title={title}>
      <div>
        <div className="t3">Delay</div>
        <div className="big">{m.delay}<small>ms</small></div>
        <div className={m.kind === "drift" ? "warn" : "t2"}>{m.kind === "drift" ? `At the start · drifts ${m.drift}` : m.kind === "fps" ? "23.976 → 25 fps · sped up ×1.042708" : `${m.frames} frames · constant`}</div>
      </div>
      {m.kind === "drift" ? (
        <Chart />
      ) : (
        <div className="col" style={{ gap: 8 }}>
          <div className="row"><span className="sec grow">Alignment</span><Seg items={["Before", "After"]} value={1} /></div>
          <Align />
        </div>
      )}
      <DL rows={[["Confidence", `${m.conf}%`], ["At the start", m.start], ["At the end", m.end], ["Windows", m.win], ["Time", m.took]]} />
      <Links>
        <L icon={<PlayRegular />}>Preview</L>
        <L icon={<CopyRegular />}>Copy command</L>
      </Links>
    </Box>
  );
}

export function Movies({ phase, sel = 0, overlay, history, menu }: { phase: MPhase; sel?: number; overlay?: ReactNode; history?: boolean; menu?: boolean }) {
  const { go } = useProto();
  const p = useProgress(phase === "writing" ? 55 : 42, phase === "writing" ? 3000 : 5200);
  const hasFiles = phase !== "empty" && phase !== "drag";
  const results = phase === "done" || phase === "writing" || phase === "written";
  const cols = results ? "18px minmax(0,1fr) 88px 84px 132px" : "minmax(0,1fr) 88px 84px 132px";
  const picked = MOVIES.filter((m) => m.pick).length;
  const n = Math.min(6, Math.floor(p / 16) + 1);

  const lcd: LcdProps =
    phase === "empty" || phase === "drag"
      ? { l1: "Drop videos and a dub to begin" }
      : phase === "ready"
        ? { l1: "6 videos · 1 dub", l2: "Ready to analyse" }
        : phase === "run"
          ? { icon: "run", l1: `Analysing ${n} of 6`, l2: "3 at a time", pct: p, time: `${Math.max(1, Math.round((100 - p) / 40))} min left` }
          : phase === "writing"
            ? { icon: "run", l1: `Writing 2 of ${picked}`, l2: "Blade Runner 2049 (2017) 1080p BluRay x264.synced.mkv", pct: p }
            : phase === "written"
              ? { icon: "ok", l1: `${picked} corrected copies written`, l2: "Beside the originals, ending .synced" }
              : { icon: "warn", l1: "4 matched · 1 drifting · 1 different cut · 1 failed", l2: "6 analysed in 1m 58s", time: `${picked} selected` };

  return (
    <Window
      page="movies"
      lcd={lcd}
      overlay={overlay}
      menu={menu}
      dock={history ? <HistoryDock /> : undefined}
      util={history ? "history" : undefined}
      cols={hasFiles ? "minmax(0,1fr) 300px" : "minmax(0,1fr)"}
      tools={
        <>
          <Cmd icon={<AddRegular />} disabled={phase === "run"} onClick={() => go("movies-ready")}>Add videos</Cmd>
          <Cmd icon={<MusicNote2Regular />} disabled={phase === "run"} onClick={() => go("movies-ready")}>{hasFiles ? "Change dub" : "Add dub"}</Cmd>
          {hasFiles && <Cmd icon={<DeleteRegular />} disabled={phase === "run"} title="Remove" />}
          {results && <Cmd icon={<ArrowExportRegular />} title="Export" />}
          {results && <Cmd icon={<FilterRegular />} title="Filter" />}
        </>
      }
      primary={
        phase === "run" || phase === "writing" ? (
          <Btn icon={<StopRegular />} kbd="Esc" onClick={() => go("movies-ready")}>Stop</Btn>
        ) : results ? (
          <>
            <Btn icon={<ArrowSyncRegular />} onClick={() => go("movies-run")} />
            {phase === "written" ? <Btn accent icon={<FolderOpenRegular />}>Open folder</Btn> : <Btn accent onClick={() => go("movies-writing")}>Fix {picked}</Btn>}
          </>
        ) : (
          <Btn accent icon={<PlayRegular />} kbd="Enter" disabled={!hasFiles} onClick={() => go("movies-run")}>Analyse</Btn>
        )
      }
    >
      {hasFiles ? (
        <>
          <Box
            body={false}
            title="Videos"
            sub="6"
            end={<span className="cell t2" style={{ paddingRight: 6 }}><span className="fi"><MusicNote2Regular /></span><span className="t3">Dub</span> <MidText text={DUB.name} tail={20} /></span>}
          >
            <Table cols={cols} head={results ? ["", "Name", " Delay", "Confidence", "Status"] : ["Name", " Delay", "Confidence", "Status"]}>
              {MOVIES.map((m, i) => {
                const [s, text] = statusOf(m);
                const done = phase === "run" ? p > (i + 1) * 16 : results;
                const running = phase === "run" && !done && p >= i * 9 && i < 3 + Math.floor(p / 16);
                return (
                  <Tr key={m.name} on={i === sel} onClick={() => go(i === 2 ? "movies-drift" : i === 4 ? "movies-cut" : "movies-done")}>
                    {results && <Chk on={!!m.pick} />}
                    <Name>{m.name}</Name>
                    <span className="r num" style={{ display: "flex" }}>{done && m.delay ? `${m.delay} ms` : <Dash />}</span>
                    {done ? <Conf c={m.conf} /> : <Dash />}
                    {phase === "ready" ? (
                      m.kind === "fail" ? <Status s="warn" text="No audio" /> : <Status s="ready" />
                    ) : phase === "run" ? (
                      done ? <Status s={s} text={text} /> : running ? <Status s="run" pct={Math.min(99, 2 + (p - i * 9) * 2.4)} /> : <Status s="wait" />
                    ) : phase === "writing" && m.pick ? (
                      i === 0 ? <Status s="written" /> : i === 1 ? <Status s="writing" pct={p} text="Writing" /> : <Status s="wait" />
                    ) : phase === "written" && m.pick ? (
                      <Status s="written" />
                    ) : (
                      <Status s={s} text={text} />
                    )}
                  </Tr>
                );
              })}
            </Table>
          </Box>
          <MovieDetails m={MOVIES[sel]} phase={phase === "writing" || phase === "written" ? "done" : phase} p={p} />
        </>
      ) : (
        <section className="box">
          <Empty icon={<VideoClipMultipleRegular />} title="Drop videos and a dub">
            <Btn icon={<AddRegular />} onClick={() => go("movies-ready")}>Add videos</Btn>
            <Btn icon={<MusicNote2Regular />} onClick={() => go("movies-ready")}>Add dub</Btn>
          </Empty>
          {phase === "drag" && (
            <div className="dropover">
              <span className="ic"><VideoClipMultipleRegular /></span>
              Drop to add 6 videos
            </div>
          )}
        </section>
      )}
      {phase === "run" && <AutoGo to="movies-done" ms={5400} />}
      {phase === "writing" && <AutoGo to="movies-written" ms={3200} />}
    </Window>
  );
}

/* ================================ SERIES ================================ */

export type SPhase = "empty" | "pairing" | "run" | "done";

export function Series({ phase }: { phase: SPhase }) {
  const { go } = useProto();
  const p = useProgress(45, 5000);
  const upto = Math.floor((p / 100) * 16);
  const results = phase === "done";
  const e = EPISODES[10];

  const lcd: LcdProps =
    phase === "empty"
      ? { l1: "Drop an episodes folder and a dubs folder" }
      : phase === "pairing"
        ? { icon: "warn", l1: "15 of 16 episodes paired", l2: "E07 paired by list order · E08 has no dub" }
        : phase === "run"
          ? { icon: "run", l1: `Analysing ${upto} of 15`, l2: "Goblin · Season 1", pct: p, time: `${Math.max(1, Math.round((100 - p) / 16))} min left` }
          : { icon: "warn", l1: "13 matched · 1 low confidence · 1 rate change", l2: "15 analysed in 6m 12s", time: "13 selected" };

  return (
    <Window
      page="series"
      lcd={lcd}
      cols={results ? "minmax(0,1fr) 300px" : "minmax(0,1fr)"}
      tools={
        <>
          <Cmd icon={<FolderRegular />} disabled={phase === "run"}>Episodes</Cmd>
          <Cmd icon={<MusicNote2Regular />} disabled={phase === "run"}>Dubs</Cmd>
          {results && <Cmd icon={<ArrowExportRegular />} title="Export" />}
        </>
      }
      primary={
        phase === "run" ? (
          <Btn icon={<StopRegular />} kbd="Esc" onClick={() => go("series-pairing")}>Stop</Btn>
        ) : results ? (
          <>
            <Btn icon={<ArrowSyncRegular />} onClick={() => go("series-run")} />
            <Btn accent onClick={() => go("movies-writing")}>Fix 13</Btn>
          </>
        ) : (
          <Btn accent icon={<PlayRegular />} kbd="Enter" disabled={phase === "empty"} onClick={() => go("series-run")}>Analyse</Btn>
        )
      }
    >
      {phase === "empty" ? (
        <section className="box">
          <Empty icon={<FolderRegular />} title="Drop an episodes folder and a dubs folder">
            <Btn icon={<FolderRegular />} onClick={() => go("series-pairing")}>Choose episodes</Btn>
            <Btn icon={<MusicNote2Regular />} onClick={() => go("series-pairing")}>Choose dubs</Btn>
          </Empty>
        </section>
      ) : (
        <Box body={false} title="Goblin · Season 1" sub="D:\Media\Goblin (2016)  ·  Hindi dubs in \Hindi">
          {phase === "pairing" ? (
            <div style={{ position: "relative", flex: 1, minHeight: 0, display: "flex", flexDirection: "column" }}>
              <Table cols="64px minmax(0,1fr) minmax(0,1fr) 150px" head={["Episode", "Video", "Dub", "Paired by"]}>
                {EPISODES.map((x, i) => (
                  <Tr key={x.ep} on={i === 7}>
                    <span className="num t2">{x.ep}</span>
                    <Name tail={20}>{x.video}</Name>
                    {x.dub ? <Name icon={<MusicNote2Regular />} tail={14}>{x.dub}</Name> : <Combo sm placeholder value="Choose a dub" w={200} />}
                    {x.by === "List order" ? <Status s="warn" text="List order" /> : x.by ? <span className="t2">{x.by}</span> : <Status s="bad" text="Not paired" />}
                  </Tr>
                ))}
              </Table>
              <div className="fly" style={{ top: 322, left: "calc(50% + 4px)", width: 300 }}>
                <div className="mhead">Dubs not used</div>
                <div className="mi on"><span className="ic"><MusicNote2Regular /></span>Goblin.S01.Special.hin.eac3</div>
                <div className="msep" />
                <div className="mi"><span className="ic"><DismissRegular /></span>Skip this episode</div>
              </div>
            </div>
          ) : (
            <Table cols={results ? "18px 64px minmax(0,1fr) 88px 84px 140px" : "64px minmax(0,1fr) 88px 84px 140px"} head={results ? ["", "Episode", "Video", " Delay", "Confidence", "Status"] : ["Episode", "Video", " Delay", "Confidence", "Status"]}>
              {EPISODES.map((x, i) => {
                const done = results || i < upto;
                const running = !results && i >= upto && i < upto + 3 && i !== 7;
                const st: [St, string] = x.conf === 0 ? ["bad", "Not paired"] : x.note === "25 → 23.976 fps" ? ["warn", "Rate change"] : x.conf < 75 ? ["warn", "Low confidence"] : ["ok", "Matched"];
                return (
                  <Tr key={x.ep} on={results && i === 10}>
                    {results && <Chk on={st[0] === "ok"} />}
                    <span className="num t2">{x.ep}</span>
                    <Name tail={20}>{x.video}</Name>
                    <span className="r num" style={{ display: "flex" }}>{done && x.delay ? `${x.delay} ms` : <Dash />}</span>
                    {done ? <Conf c={x.conf} /> : <Dash />}
                    {done ? <Status s={st[0]} text={st[1]} /> : running ? <Status s="run" pct={[70, 35, 8][i - upto] ?? 10} /> : <Status s="wait" />}
                  </Tr>
                );
              })}
            </Table>
          )}
        </Box>
      )}
      {results && (
        <Box title={<span><span className="t2 num">{e.ep}</span>  <span>{e.dub}</span></span>}>
          <div>
            <div className="t3">Delay</div>
            <div className="big">{e.delay}<small>ms</small></div>
            <div className="warn">After slowing the dub ×0.959041</div>
          </div>
          <DL rows={[["Mastered at", "25 fps"], ["Video", "23.976 fps"], ["Confidence", `${e.conf}%`], ["Windows", "6 of 6"]]} />
          <span className="t3 sm">Made for a PAL broadcast. The fixed copy is slowed to the video's speed, then shifted.</span>
          <Links><L icon={<PlayRegular />}>Preview</L></Links>
        </Box>
      )}
      {phase === "run" && <AutoGo to="series-done" ms={5200} />}
    </Window>
  );
}

/* ============================== FIND MATCH ============================== */

export type FPhase = "empty" | "ready" | "run" | "done";

export function FindMatch({ phase }: { phase: FPhase }) {
  const { go } = useProto();
  const p = useProgress(50, 4200);
  const doneN = phase === "done" ? 9 : phase === "run" ? Math.floor((p / 100) * 9) : 0;
  const best = (i: number, j: number) => phase === "done" && MATCH.conf[i][j] >= 75 && MATCH.conf.every((r) => r[j] <= MATCH.conf[i][j]);

  const lcd: LcdProps =
    phase === "empty"
      ? { l1: "Drop up to 5 videos and 5 dubs" }
      : phase === "ready"
        ? { l1: "3 videos × 3 dubs", l2: "9 tests" }
        : phase === "run"
          ? { icon: "run", l1: `Testing ${doneN} of 9`, pct: p, time: "about 2 min" }
          : { icon: "ok", l1: "Each dub has a match", l2: "The Black & White release matches none · 9 tests in 2m 40s" };

  return (
    <Window
      page="match"
      lcd={lcd}
      cols={phase === "done" ? "minmax(0,1fr) 300px" : "minmax(0,1fr)"}
      tools={
        <>
          <Cmd icon={<AddRegular />} disabled={phase === "run"}>Add videos</Cmd>
          <Cmd icon={<MusicNote2Regular />} disabled={phase === "run"}>Add dubs</Cmd>
        </>
      }
      primary={
        phase === "run" ? (
          <Btn icon={<StopRegular />} kbd="Esc" onClick={() => go("match-ready")}>Stop</Btn>
        ) : phase === "done" ? (
          <>
            <Btn icon={<ArrowSyncRegular />} onClick={() => go("match-run")} />
            <Btn accent onClick={() => go("movies-done")}>Fix 3 matches</Btn>
          </>
        ) : (
          <Btn accent icon={<PlayRegular />} kbd="Enter" disabled={phase === "empty"} onClick={() => go("match-run")}>Find matches</Btn>
        )
      }
    >
      {phase === "empty" ? (
        <section className="box">
          <Empty icon={<VideoClipMultipleRegular />} title="Drop up to 5 videos and 5 dubs">
            <Btn icon={<AddRegular />} onClick={() => go("match-ready")}>Add videos</Btn>
            <Btn icon={<MusicNote2Regular />} onClick={() => go("match-ready")}>Add dubs</Btn>
          </Empty>
        </section>
      ) : (
        <Box body={false} title="Parasite (2019)" sub="videos down, dubs across">
          <div className="tbl" style={{ ["--cols" as string]: "minmax(0,1.2fr) repeat(3, minmax(0,1fr))" }}>
            <div className="th" style={{ height: 40 }}>
              <span />
              {MATCH.dubs.map((d) => <span key={d} className="cell" style={{ color: "var(--text)" }}><span className="fi"><MusicNote2Regular /></span>{d}</span>)}
            </div>
            <div className="tb">
              {MATCH.videos.map((v, i) => (
                <div key={v} className="tr" style={{ height: 64 }}>
                  <span className="col" style={{ minWidth: 0 }}>
                    <span className="cell"><span className="fi"><VideoClipRegular /></span><span className="truncate">{v}</span></span>
                    <span className="t3 sm" style={{ paddingLeft: 24 }}><MidText text={MATCH.vfiles[i]} tail={16} /></span>
                  </span>
                  {MATCH.conf[i].map((c, j) => {
                    const k = i * 3 + j;
                    const b = best(i, j);
                    const on = phase === "done" && i === 0 && j === 0;
                    return (
                      <span key={j} className="col" style={{ justifyContent: "center", height: 52, padding: "0 10px", borderRadius: 4, background: b ? "var(--accent-wash)" : undefined, boxShadow: on ? "inset 0 0 0 1.5px var(--accent)" : undefined }}>
                        {k < doneN ? (
                          <>
                            <span className={cx("cell num", b ? "strong" : c < 50 && "t3")} style={{ fontSize: 15 }}>
                              {b && <CheckmarkCircleFilled className="acc" style={{ fontSize: 16 }} />}
                              {c}%
                            </span>
                            <span className="t3 sm num">{MATCH.delay[i][j] || (c < 50 ? "No match" : "")}</span>
                          </>
                        ) : phase === "run" && k < doneN + 3 ? (
                          <span className="cell t3"><Ring size={14} /> Testing</span>
                        ) : (
                          <Dash />
                        )}
                      </span>
                    );
                  })}
                </div>
              ))}
              {phase === "done" && (
                <div className="tr" style={{ height: 48, borderTop: "1px solid var(--divider)", borderRadius: 0, marginTop: 6 }}>
                  <span className="strong">Timed for</span>
                  {["BluRay", "WEB-DL", "WEB-DL"].map((x, j) => <span key={j} className="strong" style={{ paddingLeft: 10 }}>{x}</span>)}
                </div>
              )}
            </div>
          </div>
        </Box>
      )}
      {phase === "done" && (
        <Box title="Hindi AC3 × BluRay">
          <div><div className="t3">Confidence</div><div className="big">94%</div><div className="t2">Best of 3 releases for this dub</div></div>
          <DL rows={[["Delay", "−317.0 ms"], ["Drift", "None"], ["Windows", "6 of 6"], ["Video", MATCH.vfiles[0]], ["Dub", "Parasite.2019.hin.dd51.ac3"]]} />
          <Links><L icon={<PlayRegular />}>Preview</L></Links>
        </Box>
      )}
      {phase === "run" && <AutoGo to="match-done" ms={4400} />}
    </Window>
  );
}

