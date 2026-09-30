import {
  CheckmarkCircleFilled,
  ClosedCaptionFilled,
  ClosedCaptionRegular,
  ColumnDoubleCompareFilled,
  ColumnDoubleCompareRegular,
  ErrorCircleFilled,
  FilmstripFilled,
  FilmstripRegular,
  HeadphonesSoundWaveFilled,
  HeadphonesSoundWaveRegular,
  HistoryRegular,
  InfoRegular,
  MoviesAndTvFilled,
  MoviesAndTvRegular,
  SettingsRegular,
  WarningFilled,
  WindowConsoleRegular,
} from "@fluentui/react-icons";
import { createContext, useContext, useEffect, useState, type CSSProperties, type ReactNode } from "react";

import { Ring, cx } from "./kit";

/* ------------------------------ prototype ------------------------------ */
export interface Proto {
  go: (id: string) => void;
  live: boolean;
  theme: "dark" | "light";
  mac: boolean;
}
export const Ctx = createContext<Proto>({ go: () => {}, live: false, theme: "dark", mac: false });
export const useProto = () => useContext(Ctx);

export function useProgress(still: number, ms: number, from = 0) {
  const { live } = useProto();
  const [p, setP] = useState(live ? from : still);
  useEffect(() => {
    if (!live) return;
    const t0 = performance.now();
    let raf = 0;
    const tick = () => {
      const k = Math.min(1, (performance.now() - t0) / ms);
      setP(from + (100 - from) * k);
      if (k < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [live, ms, from]);
  return p;
}

export function AutoGo({ to, ms }: { to: string; ms: number }) {
  const { live, go } = useProto();
  useEffect(() => {
    if (!live) return;
    const t = setTimeout(() => go(to), ms);
    return () => clearTimeout(t);
  }, [live, go, to, ms]);
  return null;
}

/* -------------------------------- window -------------------------------- */
export type Page = "movies" | "series" | "match" | "dub" | "subs";

export const SUB_TOOLS: { id: string; label: string }[] = [
  { id: "sync", label: "Sync" },
  { id: "ocr", label: "OCR" },
  { id: "translate", label: "Translate" },
  { id: "fps", label: "Frame rate" },
  { id: "generate", label: "Generate" },
  { id: "style", label: "Style" },
  { id: "hdr", label: "HDR subtitles" },
  { id: "tonemap", label: "Tone-map" },
  { id: "mux", label: "Convert & mux" },
];

const PAGES: { id: Page; label: string; icon: ReactNode; on: ReactNode; to: string }[] = [
  { id: "movies", label: "Movies", icon: <FilmstripRegular />, on: <FilmstripFilled />, to: "movies-done" },
  { id: "series", label: "Series", icon: <MoviesAndTvRegular />, on: <MoviesAndTvFilled />, to: "series-pairing" },
  { id: "match", label: "Find match", icon: <ColumnDoubleCompareRegular />, on: <ColumnDoubleCompareFilled />, to: "match-done" },
  { id: "dub", label: "Dub sync", icon: <HeadphonesSoundWaveRegular />, on: <HeadphonesSoundWaveFilled />, to: "dub-done" },
  { id: "subs", label: "Subsync", icon: <ClosedCaptionRegular />, on: <ClosedCaptionFilled />, to: "sub-sync" },
];

const MENUS = ["File", "Edit", "View", "Tools", "Help"];

function AppIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden>
      <rect width="16" height="16" rx="4" fill="var(--accent)" />
      <path d="M4.5 6.5v3M8 4v8M11.5 5.5v5" stroke="var(--on-accent)" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

export interface LcdProps {
  icon?: "run" | "ok" | "warn" | "bad" | "idle";
  l1: ReactNode;
  l2?: ReactNode;
  pct?: number;
  time?: ReactNode;
}

/** The status display in the middle of the toolbar. */
export function Lcd({ icon = "idle", l1, l2, pct, time }: LcdProps) {
  return (
    <div className="lcd">
      <span className="ic">
        {icon === "run" ? <Ring size={16} /> : icon === "ok" ? <CheckmarkCircleFilled className="ok" /> : icon === "warn" ? <WarningFilled className="warn" /> : icon === "bad" ? <ErrorCircleFilled className="bad" /> : <InfoRegular />}
      </span>
      <div className="col" style={{ minWidth: 0 }}>
        <span className="l1 truncate">{l1}</span>
        {l2 && <span className="l2 truncate">{l2}</span>}
      </div>
      {time && <span className="time">{time}</span>}
      {pct !== undefined && (
        <span className="bar"><i style={{ width: `${pct}%` }} /></span>
      )}
    </div>
  );
}

export function Window({
  page,
  tools,
  lcd,
  primary,
  strip,
  cols = "minmax(0,1fr) 300px",
  children,
  dock,
  overlay,
  menu,
  busy,
  util,
}: {
  menu?: boolean;
  page: Page;
  tools?: ReactNode;
  lcd: LcdProps;
  primary?: ReactNode;
  strip?: ReactNode;
  cols?: string;
  children: ReactNode;
  dock?: ReactNode;
  overlay?: ReactNode;
  busy?: Page;
  util?: "history" | "output";
}) {
  const { go, theme, mac } = useProto();
  const title = PAGES.find((p) => p.id === page)?.label;
  return (
    <div className={cx("win", `theme-${theme}`)}>
      <header className="titlebar">
        {mac ? (
          <span className="lights">
            <i style={{ background: "#ff5f57" }} />
            <i style={{ background: "#febc2e" }} />
            <i style={{ background: "#28c840" }} />
          </span>
        ) : (
          <>
            <AppIcon />
            <nav className="menubar">
              {MENUS.map((m) => (
                <button key={m} type="button" className={cx(menu && m === "File" && "on")} onClick={() => go(m === "File" ? "menu-file" : "prefs")}>{m}</button>
              ))}
            </nav>
          </>
        )}
        <span className="wtitle">{title} — AudioSyncMaster</span>
        {!mac && (
          <span className="caps">
            <button type="button" className="cap" aria-label="Minimize"><svg width="10" height="10"><path d="M0 5.5h10" stroke="currentColor" /></svg></button>
            <button type="button" className="cap" aria-label="Maximize"><svg width="10" height="10"><rect x=".5" y=".5" width="9" height="9" rx="1.5" fill="none" stroke="currentColor" /></svg></button>
            <button type="button" className="cap x" aria-label="Close"><svg width="10" height="10"><path d="M.5.5l9 9M9.5.5l-9 9" stroke="currentColor" /></svg></button>
          </span>
        )}
      </header>

      <div className="toolbar">
        <div className="l">{tools}</div>
        <Lcd {...lcd} />
        <div className="r">{primary}</div>
      </div>
      {strip && <div className="strip2">{strip}</div>}

      <div className="ws" style={{ gridTemplateColumns: cols }}>
        {children}
        {dock && <div className="box dock">{dock}</div>}
      </div>

      <footer className="pagebar">
        <span className="l">FFmpeg 6.1 · GPU ready</span>
        <nav className="pages" aria-label="Pages">
          {PAGES.map((p) => (
            <button key={p.id} type="button" className={cx("pg", page === p.id && "on")} onClick={() => go(p.to)} title={`${p.label} (Ctrl+${PAGES.indexOf(p) + 1})`}>
              <span className="ic">{page === p.id ? p.on : p.icon}</span>
              {p.label}
              {busy === p.id && page !== p.id && <span className="busy"><Ring size={10} /></span>}
            </button>
          ))}
        </nav>
        <span className="r">
          <button type="button" className={cx("ub", util === "history" && "on")} title="History (Ctrl+H)" onClick={() => go("history")}><span className="ic"><HistoryRegular /></span></button>
          <button type="button" className={cx("ub", util === "output" && "on")} title="Output (Ctrl+`)" onClick={() => go("dub-failed")}><span className="ic"><WindowConsoleRegular /></span></button>
          <button type="button" className="ub" title="Preferences (Ctrl+,)" onClick={() => go("prefs")}><span className="ic"><SettingsRegular /></span></button>
        </span>
      </footer>
      {menu && <FileMenu />}
      {overlay}
    </div>
  );
}

function FileMenu() {
  const { go } = useProto();
  const I = ({ children, k, to }: { children: ReactNode; k?: string; to?: string }) => (
    <div className="mi" onClick={() => to && go(to)}>
      <span className="grow">{children}</span>
      {k && <span className="t3 sm">{k}</span>}
    </div>
  );
  return (
    <>
      <div style={{ position: "absolute", inset: 0, zIndex: 35 }} onClick={() => go("movies-done")} />
      <div className="fly menu-fly">
        <I k="Ctrl+O" to="movies-ready">Add videos…</I>
        <I k="Ctrl+Shift+O">Add a dub…</I>
        <I>Add a folder…</I>
        <div className="msep" />
        <I k="Ctrl+E">Export results…</I>
        <I>Open recent</I>
        <div className="msep" />
        <I k="Ctrl+," to="prefs">Preferences…</I>
        <div className="msep" />
        <I k="Alt+F4">Exit</I>
      </div>
    </>
  );
}

/* A docked panel with a header. */
export function Box({ title, sub, end, children, style, body = true }: { title?: ReactNode; sub?: ReactNode; end?: ReactNode; children: ReactNode; style?: CSSProperties; body?: boolean }) {
  return (
    <section className="box" style={style}>
      {title !== undefined && (
        <div className="box-h">
          <span className="tt truncate">{title}</span>
          {sub && <span className="truncate t3">{sub}</span>}
          {end && <span className="end">{end}</span>}
        </div>
      )}
      {body ? <div className="box-b">{children}</div> : children}
    </section>
  );
}

export function DockTabs({ tabs, on, tools, onTab }: { tabs: string[]; on: string; tools?: ReactNode; onTab?: (t: string) => void }) {
  return (
    <div className="dtabs">
      {tabs.map((t) => (
        <button key={t} type="button" className={cx("tab", t === on && "on")} onClick={() => onTab?.(t)}>{t}</button>
      ))}
      {tools && <span className="tools">{tools}</span>}
    </div>
  );
}
