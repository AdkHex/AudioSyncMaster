import { StrictMode, useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createRoot } from "react-dom/client";

import "./styles.css";
import { FindMatch, Movies, Series } from "./screens/analyse";
import { Prefs, UpdateInstall } from "./screens/app";
import { DubSync } from "./screens/dub";
import { OcrReview, SubTool } from "./screens/subs";
import { Ctx, SUB_TOOLS, type Proto } from "./shell";

interface Screen {
  id: string;
  group: string;
  label: string;
  render: () => ReactNode;
}

const SCREENS: Screen[] = [
  { id: "movies-empty", group: "Movies", label: "Empty", render: () => <Movies phase="empty" /> },
  { id: "movies-drag", group: "Movies", label: "Dropping files", render: () => <Movies phase="drag" /> },
  { id: "movies-ready", group: "Movies", label: "Ready", render: () => <Movies phase="ready" /> },
  { id: "movies-run", group: "Movies", label: "Analysing", render: () => <Movies phase="run" /> },
  { id: "movies-done", group: "Movies", label: "Results", render: () => <Movies phase="done" /> },
  { id: "movies-drift", group: "Movies", label: "Result: drifting", render: () => <Movies phase="done" sel={2} /> },
  { id: "movies-cut", group: "Movies", label: "Result: different cut", render: () => <Movies phase="done" sel={4} /> },
  { id: "movies-writing", group: "Movies", label: "Writing fixes", render: () => <Movies phase="writing" /> },
  { id: "movies-written", group: "Movies", label: "Written", render: () => <Movies phase="written" /> },

  { id: "series-empty", group: "Series", label: "Empty", render: () => <Series phase="empty" /> },
  { id: "series-pairing", group: "Series", label: "Pairing", render: () => <Series phase="pairing" /> },
  { id: "series-run", group: "Series", label: "Analysing", render: () => <Series phase="run" /> },
  { id: "series-done", group: "Series", label: "Results", render: () => <Series phase="done" /> },

  { id: "match-empty", group: "Find match", label: "Empty", render: () => <FindMatch phase="empty" /> },
  { id: "match-ready", group: "Find match", label: "Ready", render: () => <FindMatch phase="ready" /> },
  { id: "match-run", group: "Find match", label: "Testing", render: () => <FindMatch phase="run" /> },
  { id: "match-done", group: "Find match", label: "Results", render: () => <FindMatch phase="done" /> },

  { id: "dub-empty", group: "Dub sync", label: "Empty", render: () => <DubSync phase="empty" /> },
  { id: "dub-ready", group: "Dub sync", label: "Ready", render: () => <DubSync phase="ready" /> },
  { id: "dub-run", group: "Dub sync", label: "Syncing", render: () => <DubSync phase="run" /> },
  { id: "dub-done", group: "Dub sync", label: "Synced", render: () => <DubSync phase="done" /> },
  { id: "dub-preview", group: "Dub sync", label: "Preview", render: () => <DubSync phase="preview" /> },
  { id: "dub-edit", group: "Dub sync", label: "Editing a cut", render: () => <DubSync phase="edit" /> },
  { id: "dub-season", group: "Dub sync", label: "A season", render: () => <DubSync phase="season" /> },
  { id: "dub-failed", group: "Dub sync", label: "Failure + output", render: () => <DubSync phase="failed" /> },

  { id: "sub-sync-empty", group: "Subsync", label: "Sync · empty", render: () => <SubTool tool="sync" phase="empty" /> },
  { id: "sub-sync", group: "Subsync", label: "Sync · ready", render: () => <SubTool tool="sync" /> },
  { id: "sub-sync-run", group: "Subsync", label: "Sync · running", render: () => <SubTool tool="sync" phase="run" /> },
  { id: "sub-sync-done", group: "Subsync", label: "Sync · done + cues", render: () => <SubTool tool="sync" phase="done" /> },
  { id: "sub-ocr-done", group: "Subsync", label: "OCR · review", render: () => <OcrReview /> },
  ...SUB_TOOLS.filter((t) => t.id !== "sync").map((t) => ({ id: `sub-${t.id}`, group: "Subsync", label: t.label, render: () => <SubTool tool={t.id} /> })),

  { id: "menu-file", group: "App", label: "File menu", render: () => <Movies phase="done" menu /> },
  { id: "history", group: "App", label: "History", render: () => <Movies phase="done" history /> },
  { id: "prefs", group: "App", label: "Preferences · General", render: () => <Prefs tab="general" /> },
  { id: "prefs-analysis", group: "App", label: "Preferences · Analysis", render: () => <Prefs tab="analysis" /> },
  { id: "prefs-dub", group: "App", label: "Preferences · Dub sync", render: () => <Prefs tab="dub" /> },
  { id: "prefs-subs", group: "App", label: "Preferences · Subtitles", render: () => <Prefs tab="subs" /> },
  { id: "prefs-updates", group: "App", label: "Preferences · Updates", render: () => <Prefs tab="updates" /> },
  { id: "update-install", group: "App", label: "Installing an update", render: () => <UpdateInstall /> },
];

const GROUPS = [...new Set(SCREENS.map((s) => s.group))];
const find = (id: string) => SCREENS.find((s) => s.id === id) ?? SCREENS[0];

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  return { s: h.get("s") ?? "movies-done", all: h.get("view") === "all", theme: (h.get("theme") as "dark" | "light") ?? "dark", mac: h.get("os") === "mac" };
}

function Scaled({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [k, setK] = useState(0.3);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setK(el.getBoundingClientRect().width / 1180));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return (
    <div className="ovframe" ref={ref}>
      <div style={{ transform: `scale(${k})` }}>{children}</div>
    </div>
  );
}

function App() {
  const [st, setSt] = useState(readHash);
  const [nonce, setNonce] = useState(0);
  const go = useCallback((id: string) => {
    setSt((s) => ({ ...s, s: id, all: false }));
    setNonce((n) => n + 1);
  }, []);

  useEffect(() => {
    const h = new URLSearchParams({ s: st.s });
    if (st.all) h.set("view", "all");
    if (st.theme === "light") h.set("theme", "light");
    if (st.mac) h.set("os", "mac");
    history.replaceState(null, "", `#${h}`);
  }, [st]);

  useEffect(() => {
    const onHash = () => setSt(readHash());
    const onKey = (e: KeyboardEvent) => {
      const i = SCREENS.findIndex((x) => x.id === st.s);
      if (e.key === "ArrowRight" || e.key === "ArrowDown") go(SCREENS[(i + 1) % SCREENS.length].id);
      if (e.key === "ArrowLeft" || e.key === "ArrowUp") go(SCREENS[(i - 1 + SCREENS.length) % SCREENS.length].id);
      if (e.key === "g") setSt((s) => ({ ...s, all: !s.all }));
      if (e.key === "t") setSt((s) => ({ ...s, theme: s.theme === "dark" ? "light" : "dark" }));
    };
    window.addEventListener("hashchange", onHash);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("hashchange", onHash);
      window.removeEventListener("keydown", onKey);
    };
  }, [st.s, go]);

  const desk = useRef<HTMLDivElement>(null);
  const [k, setK] = useState(1);
  useLayoutEffect(() => {
    const el = desk.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setK(Math.min(1, (r.width - 48) / 1180, (r.height - 48) / 780));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [st.all]);

  const live: Proto = { go, live: true, theme: st.theme, mac: st.mac };
  let n = 0;

  return (
    <div className={`stage ${st.theme}`}>
      <aside className="story">
        <h1>AudioSyncMaster — mockup</h1>
        <div className="opts">
          <button type="button" className={!st.all ? "on" : ""} onClick={() => setSt((s) => ({ ...s, all: false }))}>Prototype</button>
          <button type="button" className={st.all ? "on" : ""} onClick={() => setSt((s) => ({ ...s, all: true }))}>All screens</button>
          <button type="button" className={st.theme === "dark" ? "on" : ""} onClick={() => setSt((s) => ({ ...s, theme: "dark" }))}>Dark</button>
          <button type="button" className={st.theme === "light" ? "on" : ""} onClick={() => setSt((s) => ({ ...s, theme: "light" }))}>Light</button>
          <button type="button" className={!st.mac ? "on" : ""} onClick={() => setSt((s) => ({ ...s, mac: false }))}>Windows</button>
          <button type="button" className={st.mac ? "on" : ""} onClick={() => setSt((s) => ({ ...s, mac: true }))}>macOS</button>
        </div>
        {GROUPS.map((g) => (
          <div key={g}>
            <div className="g">{g}</div>
            {SCREENS.filter((x) => x.group === g).map((x) => {
              n += 1;
              return (
                <button key={x.id} type="button" className={`s${x.id === st.s && !st.all ? " on" : ""}`} onClick={() => go(x.id)}>
                  <i>{n}</i>
                  {x.label}
                </button>
              );
            })}
          </div>
        ))}
        <div className="foot">← → step · G all screens · T theme. Everything in the window is clickable; running screens finish by themselves.</div>
      </aside>

      {st.all ? (
        <div className="overview">
          <Ctx.Provider value={{ ...live, live: false }}>
            {GROUPS.map((g) => (
              <section key={g}>
                <h2>{g}</h2>
                <div className="ovgrid">
                  {SCREENS.filter((x) => x.group === g).map((x) => (
                    <div key={x.id} className="ovcell" onClick={() => go(x.id)}>
                      <Scaled>{x.render()}</Scaled>
                      <span className="ovcap"><b>{x.label}</b></span>
                    </div>
                  ))}
                </div>
              </section>
            ))}
          </Ctx.Provider>
        </div>
      ) : (
        <div className="desk" ref={desk}>
          <div style={{ position: "absolute", left: "50%", top: "50%", width: 1180, height: 780, transform: `translate(-50%, -50%) scale(${k})` }}>
            <Ctx.Provider value={live}>
              <div key={`${st.s}-${nonce}`}>{find(st.s).render()}</div>
            </Ctx.Provider>
          </div>
        </div>
      )}
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
