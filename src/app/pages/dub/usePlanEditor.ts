/** Editing a finished job's cuts in the timeline: the plan as edited, undo
 *  and redo, a cut being dragged, a stretch being slipped, and whether
 *  there is anything to write.
 *
 *  Everything that changes the plan is a pure function in dubPlanEdit.ts;
 *  this is the state around them, as a reducer so it can be tested without
 *  a browser. The hook keeps it in step with the job it edits: a new job
 *  starts afresh, a written track becomes the new starting point, and a
 *  write that failed leaves the edits on screen for another try. Edits to
 *  a job the user moved away from are kept until they come back. */

import { useEffect, useMemo, useReducer } from "react";

import { moveBoundary, nudgeOffset, samePlan, validatePlan } from "@/lib/dubPlanEdit";
import type { DubSyncPlan } from "@/lib/types";

/** Steps kept to undo. */
export const HISTORY_STEPS = 100;

interface Snapshot {
  /** The plan the track was written from: what "changes" are counted against. */
  base: DubSyncPlan;
  /** The engine's own plan, for "Back to the engine's plan". */
  engine: DubSyncPlan;
  plan: DubSyncPlan;
  past: DubSyncPlan[];
  future: DubSyncPlan[];
}

export interface EditorState extends Partial<Snapshot> {
  key: string | null;
  /** A slip in progress: drawn, and shown in the offset box, but committed
   *  once, as one step, when the pointer lets go. */
  slip: { index: number; base: DubSyncPlan; next: DubSyncPlan } | null;
  /** A cut being dragged: its starting plan is on the undo stack already. */
  dragging: boolean;
  /** Unwritten edits of jobs the user moved away from, by key. */
  stash: Record<string, Snapshot>;
}

export const initialEditorState: EditorState = { key: null, slip: null, dragging: false, stash: {} };

export type EditorAction =
  | { type: "sync"; key: string | null; plan: DubSyncPlan | null; enginePlan: DubSyncPlan | null }
  | { type: "commit"; plan: DubSyncPlan }
  /** A change to the plan as it stands when the action lands, so two
   *  quick key presses both count. */
  | { type: "edit"; edit: (plan: DubSyncPlan) => DubSyncPlan }
  | { type: "undo" }
  | { type: "redo" }
  | { type: "dragStart" }
  | { type: "drag"; boundary: number; timeS: number }
  | { type: "dragEnd" }
  | { type: "slipStart"; index: number }
  | { type: "slip"; index: number; deltaS: number }
  | { type: "slipEnd" };

export function isDirty(state: EditorState): boolean {
  return !!state.plan && !!state.base && !samePlan(state.plan, state.base);
}

function fresh(key: string | null, plan: DubSyncPlan | null, engine: DubSyncPlan | null, stash: Record<string, Snapshot>): EditorState {
  if (!plan) return { key, slip: null, dragging: false, stash };
  return { key, base: plan, engine: engine ?? plan, plan, past: [], future: [], slip: null, dragging: false, stash };
}

function push(past: DubSyncPlan[], plan: DubSyncPlan): DubSyncPlan[] {
  return [...past.slice(-(HISTORY_STEPS - 1)), plan];
}

function commit(state: EditorState, next: DubSyncPlan): EditorState {
  const current = state.plan;
  if (!current || next === current || samePlan(next, current)) return state;
  return { ...state, plan: next, past: push(state.past ?? [], current), future: [] };
}

export function planEditorReducer(state: EditorState, action: EditorAction): EditorState {
  switch (action.type) {
    case "sync": {
      if (action.key !== state.key) {
        // Moving to another job: keep this one's unwritten edits aside.
        const stash = { ...state.stash };
        if (state.key && isDirty(state)) {
          stash[state.key] = { base: state.base!, engine: state.engine!, plan: state.plan!, past: state.past!, future: state.future! };
        }
        const kept = action.key ? stash[action.key] : undefined;
        if (action.key) delete stash[action.key];
        if (kept && action.plan && samePlan(kept.base, action.plan)) {
          return { key: action.key, ...kept, slip: null, dragging: false, stash };
        }
        return fresh(action.key, action.plan, action.enginePlan, stash);
      }
      if (!action.plan || !state.plan) return fresh(action.key, action.plan, action.enginePlan, state.stash);
      if (state.base && action.plan === state.base && (action.enginePlan ?? action.plan) === state.engine) return state;
      // The track was written from the plan as edited: that is the new start.
      if (samePlan(action.plan, state.plan)) {
        return { ...state, base: action.plan, engine: action.enginePlan ?? state.engine, past: [], future: [] };
      }
      if (!isDirty(state)) return fresh(action.key, action.plan, action.enginePlan, state.stash);
      // A write that did not go through: the track on disk is the old one,
      // and the edits stay for another try.
      return { ...state, base: action.plan, engine: action.enginePlan ?? state.engine };
    }

    case "commit":
      return commit(state, action.plan);

    case "edit":
      return state.plan ? commit(state, action.edit(state.plan)) : state;

    case "undo": {
      const past = state.past ?? [];
      if (!state.plan || past.length === 0) return state;
      return { ...state, plan: past[past.length - 1], past: past.slice(0, -1), future: [state.plan, ...(state.future ?? [])] };
    }

    case "redo": {
      const future = state.future ?? [];
      if (!state.plan || future.length === 0) return state;
      return { ...state, plan: future[0], future: future.slice(1), past: push(state.past ?? [], state.plan) };
    }

    // A drag puts the plan it started from on the undo stack once, at its
    // start; if nothing moved, the entry comes off again at its end.
    case "dragStart":
      if (!state.plan) return state;
      return { ...state, past: push(state.past ?? [], state.plan), future: [], dragging: true };

    case "drag":
      if (!state.plan) return state;
      return { ...state, plan: moveBoundary(state.plan, action.boundary, action.timeS) };

    case "dragEnd": {
      const past = state.past ?? [];
      const unmoved = state.plan && past.length > 0 && samePlan(past[past.length - 1], state.plan);
      return { ...state, dragging: false, past: unmoved ? past.slice(0, -1) : past };
    }

    // A slip is measured from where it started, so the offset follows the
    // pointer exactly and never accumulates rounding.
    case "slipStart":
      if (!state.plan) return state;
      return { ...state, slip: { index: action.index, base: state.plan, next: state.plan } };

    case "slip": {
      const slip = state.slip;
      if (!slip || slip.index !== action.index) return state;
      return { ...state, slip: { ...slip, next: nudgeOffset(slip.base, action.index, action.deltaS) } };
    }

    case "slipEnd": {
      const slip = state.slip;
      const cleared = { ...state, slip: null };
      return slip && !samePlan(slip.next, slip.base) ? commit(cleared, slip.next) : cleared;
    }

    default:
      return state;
  }
}

/** What the page reads off the editor. */
export interface EditorView {
  key: string | null;
  /** The plan as edited; null when there is nothing to edit. */
  plan: DubSyncPlan | null;
  /** What to draw: the plan, or a slip in progress. */
  shown: DubSyncPlan | null;
  dirty: boolean;
  /** Steps taken since the track was written, for "1 change". */
  changes: number;
  canUndo: boolean;
  canRedo: boolean;
  /** Why the plan as edited could not be written, as sentences. */
  problems: string[];
  /** The plan is the engine's own. */
  isEngines: boolean;
  slipping: boolean;
}

export function viewOf(state: EditorState): EditorView {
  const dirty = isDirty(state);
  const plan = state.plan ?? null;
  return {
    key: state.key,
    plan,
    shown: state.slip?.next ?? plan,
    dirty,
    changes: dirty ? Math.max(1, state.past?.length ?? 0) : 0,
    canUndo: (state.past?.length ?? 0) > 0,
    canRedo: (state.future?.length ?? 0) > 0,
    problems: plan ? validatePlan(plan) : [],
    isEngines: plan && state.engine ? samePlan(plan, state.engine) : true,
    slipping: state.slip !== null,
  };
}

export interface PlanEditor extends EditorView {
  commit: (plan: DubSyncPlan) => void;
  /** Apply a change to the plan as it stands. */
  change: (edit: (plan: DubSyncPlan) => DubSyncPlan) => void;
  undo: () => void;
  redo: () => void;
  backToEngine: () => void;
  dragStart: () => void;
  drag: (boundary: number, timeS: number) => void;
  dragEnd: () => void;
  slipStart: (index: number) => void;
  slip: (index: number, deltaS: number) => void;
  slipEnd: () => void;
}

/** The editor for the job with `key`, kept in step with its plan. `hold`
 *  freezes it while the job's track is being written from the edits, so
 *  the plan the write carries does not count as a new starting point. */
export function usePlanEditor({
  key,
  plan,
  enginePlan,
  hold = false,
}: {
  key: string | null;
  plan: DubSyncPlan | null;
  enginePlan: DubSyncPlan | null;
  hold?: boolean;
}): PlanEditor {
  const [state, dispatch] = useReducer(planEditorReducer, initialEditorState);

  useEffect(() => {
    if (!hold) dispatch({ type: "sync", key, plan, enginePlan });
  }, [key, plan, enginePlan, hold]);

  const engine = state.engine;
  const actions = useMemo(
    () => ({
      commit: (next: DubSyncPlan) => dispatch({ type: "commit", plan: next }),
      change: (edit: (plan: DubSyncPlan) => DubSyncPlan) => dispatch({ type: "edit", edit }),
      undo: () => dispatch({ type: "undo" }),
      redo: () => dispatch({ type: "redo" }),
      backToEngine: () => engine && dispatch({ type: "commit", plan: engine }),
      dragStart: () => dispatch({ type: "dragStart" }),
      drag: (boundary: number, timeS: number) => dispatch({ type: "drag", boundary, timeS }),
      dragEnd: () => dispatch({ type: "dragEnd" }),
      slipStart: (index: number) => dispatch({ type: "slipStart", index }),
      slip: (index: number, deltaS: number) => dispatch({ type: "slip", index, deltaS }),
      slipEnd: () => dispatch({ type: "slipEnd" }),
    }),
    [engine],
  );

  const view = useMemo(() => viewOf(state), [state]);
  return useMemo(() => ({ ...view, ...actions }), [view, actions]);
}
