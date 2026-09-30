import { describe, expect, it } from "vitest";

import outcomeFixture from "@/lib/__fixtures__/dubsync-outcome.json";
import { nudgeOffset, splitAt } from "@/lib/dubPlanEdit";
import type { DubSyncOutcome, DubSyncPlan } from "@/lib/types";

import { HISTORY_STEPS, initialEditorState, planEditorReducer, viewOf, type EditorAction, type EditorState } from "./usePlanEditor";

/** The engine's real plan for the five-minute fixture. */
const plan = (outcomeFixture as unknown as DubSyncOutcome).plan as DubSyncPlan;

const run = (state: EditorState, ...actions: EditorAction[]) => actions.reduce(planEditorReducer, state);
const opened = (key = "job-0", p: DubSyncPlan = plan, engine: DubSyncPlan | null = null) =>
  planEditorReducer(initialEditorState, { type: "sync", key, plan: p, enginePlan: engine });

describe("planEditorReducer", () => {
  it("opens on the job's plan with nothing to write", () => {
    const view = viewOf(opened());
    expect(view.plan).toBe(plan);
    expect(view.dirty).toBe(false);
    expect(view.changes).toBe(0);
    expect(view.canUndo).toBe(false);
    expect(view.problems).toEqual([]);
    expect(view.isEngines).toBe(true);
  });

  it("counts an edit, undoes it and redoes it", () => {
    const edited = run(opened(), { type: "edit", edit: (p) => nudgeOffset(p, 1, 0.01) });
    expect(viewOf(edited)).toMatchObject({ dirty: true, changes: 1, canUndo: true, canRedo: false });
    expect(edited.plan!.segments[1].offsetS).toBeCloseTo(0.816, 6);
    const undone = run(edited, { type: "undo" });
    expect(viewOf(undone)).toMatchObject({ dirty: false, changes: 0, canUndo: false, canRedo: true });
    expect(undone.plan).toBe(plan);
    const redone = run(undone, { type: "redo" });
    expect(redone.plan).toBe(edited.plan);
    expect(viewOf(redone).changes).toBe(1);
  });

  it("applies quick edits one after the other, not both to the same plan", () => {
    const twice = run(opened(), { type: "edit", edit: (p) => nudgeOffset(p, 1, 0.01) }, { type: "edit", edit: (p) => nudgeOffset(p, 1, 0.01) });
    expect(twice.plan!.segments[1].offsetS).toBeCloseTo(0.826, 6);
    expect(viewOf(twice).changes).toBe(2);
  });

  it("ignores a change that changes nothing", () => {
    const state = opened();
    expect(run(state, { type: "commit", plan })).toBe(state);
    expect(run(state, { type: "commit", plan: { ...plan } })).toBe(state);
  });

  it("makes a dragged cut one step to undo, and none when it did not move", () => {
    const dragged = run(opened(), { type: "dragStart" }, { type: "drag", boundary: 1, timeS: 61 }, { type: "drag", boundary: 1, timeS: 62 }, { type: "dragEnd" });
    expect(dragged.plan!.segments[1].endS).toBe(62);
    expect(dragged.past).toHaveLength(1);
    expect(run(dragged, { type: "undo" }).plan).toBe(plan);
    const still = run(opened(), { type: "dragStart" }, { type: "dragEnd" });
    expect(still.past).toHaveLength(0);
    expect(viewOf(still).dirty).toBe(false);
  });

  it("draws a slip as it goes and commits it once, from where it started", () => {
    let state = run(opened(), { type: "slipStart", index: 3 }, { type: "slip", index: 3, deltaS: -0.2 }, { type: "slip", index: 3, deltaS: -0.25 });
    const view = viewOf(state);
    expect(view.slipping).toBe(true);
    expect(view.plan).toBe(plan);
    expect(view.shown!.segments[3].offsetS).toBeCloseTo(-14.444, 6);
    expect(view.dirty).toBe(false);
    state = run(state, { type: "slipEnd" });
    expect(viewOf(state)).toMatchObject({ slipping: false, dirty: true, changes: 1 });
    expect(state.plan!.segments[3].offsetS).toBeCloseTo(-14.444, 6);
    // A slip that came back to where it started is no change.
    const back = run(opened(), { type: "slipStart", index: 3 }, { type: "slip", index: 3, deltaS: 0.1 }, { type: "slip", index: 3, deltaS: 0 }, { type: "slipEnd" });
    expect(viewOf(back).dirty).toBe(false);
  });

  it("keeps a hundred steps to undo", () => {
    let state = opened();
    for (let i = 0; i < HISTORY_STEPS + 20; i++) state = run(state, { type: "edit", edit: (p) => nudgeOffset(p, 1, 0.001) });
    expect(state.past).toHaveLength(HISTORY_STEPS);
  });

  it("goes back to the engine's plan as one more step", () => {
    const edited = { ...nudgeOffset(plan, 1, 0.01) };
    // The track was written from edited cuts once; the engine's plan is kept.
    const state = opened("job-0", edited, plan);
    expect(viewOf(state)).toMatchObject({ dirty: false, isEngines: false });
    const back = run(state, { type: "commit", plan: state.engine! });
    expect(viewOf(back)).toMatchObject({ dirty: true, isEngines: true, changes: 1 });
  });

  it("says why an edited plan could not be written", () => {
    const broken: DubSyncPlan = { ...plan, segments: plan.segments.slice(1) };
    const view = viewOf(run(opened(), { type: "commit", plan: broken }));
    expect(view.problems).toContain("The first piece does not start at 0.");
  });

  it("starts afresh from a track written with the edits", () => {
    const edited = run(opened(), { type: "edit", edit: (p) => splitAt(p, 30) });
    // The engine echoes the plan it wrote: a new object, the same cuts.
    const written = { ...edited.plan!, segments: edited.plan!.segments.map((s) => ({ ...s })) };
    const after = run(edited, { type: "sync", key: "job-0", plan: written, enginePlan: plan });
    expect(viewOf(after)).toMatchObject({ dirty: false, changes: 0, canUndo: false, isEngines: false });
  });

  it("keeps the edits when the write did not go through", () => {
    const edited = run(opened(), { type: "edit", edit: (p) => nudgeOffset(p, 1, 0.01) });
    // The row goes back to the track on disk: the plan it had, as a new object.
    const kept = run(edited, { type: "sync", key: "job-0", plan: { ...plan }, enginePlan: plan });
    expect(viewOf(kept)).toMatchObject({ dirty: true, changes: 1 });
    expect(kept.plan).toBe(edited.plan);
  });

  it("follows a new plan for the job when nothing was edited", () => {
    const next = splitAt(plan, 30);
    const state = run(opened(), { type: "sync", key: "job-0", plan: next, enginePlan: next });
    expect(state.plan).toBe(next);
    expect(viewOf(state).dirty).toBe(false);
  });

  it("keeps a job's edits aside while another job is open", () => {
    const edited = run(opened("job-0"), { type: "edit", edit: (p) => nudgeOffset(p, 1, 0.01) });
    const other = run(edited, { type: "sync", key: "job-1", plan, enginePlan: plan });
    expect(viewOf(other)).toMatchObject({ key: "job-1", dirty: false });
    const back = run(other, { type: "sync", key: "job-0", plan, enginePlan: plan });
    expect(viewOf(back)).toMatchObject({ key: "job-0", dirty: true, changes: 1 });
    expect(back.plan).toBe(edited.plan);
    // Nothing is kept for a job whose track changed meanwhile.
    const stale = run(other, { type: "sync", key: "job-0", plan: splitAt(plan, 30), enginePlan: plan });
    expect(viewOf(stale).dirty).toBe(false);
  });

  it("has nothing to edit without a plan", () => {
    const view = viewOf(planEditorReducer(opened(), { type: "sync", key: null, plan: null, enginePlan: null }));
    expect(view).toMatchObject({ key: null, plan: null, shown: null, dirty: false, problems: [] });
    expect(run(planEditorReducer(initialEditorState, { type: "undo" }))).toEqual(initialEditorState);
  });
});
