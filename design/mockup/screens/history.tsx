import { SearchRegular } from "@fluentui/react-icons";

import { HISTORY } from "../data";
import { Combo, Status, Table, Tr, type St } from "../kit";
import { DockTabs, useProto } from "../shell";

/** History lives in the bottom dock of every page, next to Output. */
export function HistoryDock() {
  const { go } = useProto();
  const to: Record<string, string> = { "Dub sync": "dub-done", Movies: "movies-done", Subsync: "sub-sync-done", "Find match": "match-done", Series: "series-done" };
  return (
    <div style={{ height: 280, display: "flex", flexDirection: "column" }}>
      <DockTabs
        tabs={["Output", "History"]}
        on="History"
        onTab={(t) => t === "Output" && go("dub-failed")}
        tools={
          <>
            <Combo ghost sm value="All pages" w={112} />
            <div className="tbox" style={{ width: 200, height: 28 }}><SearchRegular className="t3" /><span className="ph">Search</span></div>
          </>
        }
      />
      <Table cols="110px minmax(0,1fr) minmax(0,1.3fr) 120px" head={["Page", "Name", "Result", "When"]}>
        {HISTORY.map((h, i) => (
          <Tr key={i} on={i === 1} onClick={() => go(to[h.mode])} style={{ height: 32 }}>
            <span className="t2">{h.mode}</span>
            <span className="truncate">{h.name}</span>
            <Status s={h.tone as St} text={h.res} />
            <span className="t3 num">{h.when}</span>
          </Tr>
        ))}
      </Table>
    </div>
  );
}
