/** Test-only helper: expand a React element into its host-element tree by
 *  calling function components directly, so tests can find a control and
 *  call its handler without a DOM. Works because the settings fields use no
 *  hooks; a component that does (the Sync reference picker) is kept as a
 *  leaf with its props intact. */

import { Fragment, isValidElement, type ReactElement, type ReactNode } from "react";

export interface HostNode {
  type: string;
  props: Record<string, unknown>;
  children: HostNode[];
  /** Components that could not be expanded (they use hooks). */
  component?: string;
}

/** Components known to use hooks; kept as leaves instead of being called. */
const STATEFUL = new Set(["ReferencePicker"]);

type AnyComponent = ((props: Record<string, unknown>) => ReactNode) & { displayName?: string };

function expandNode(node: ReactNode, out: HostNode[]): void {
  if (node === null || node === undefined || typeof node === "boolean") return;
  if (typeof node === "string" || typeof node === "number") {
    out.push({ type: "#text", props: { value: String(node) }, children: [] });
    return;
  }
  if (Array.isArray(node)) {
    for (const child of node) expandNode(child, out);
    return;
  }
  if (!isValidElement(node)) return;
  const element = node as ReactElement<Record<string, unknown>>;
  const { type, props } = element;
  if (type === Fragment) {
    expandNode(props.children as ReactNode, out);
    return;
  }
  if (typeof type === "function") {
    const fn = type as unknown as AnyComponent;
    if (STATEFUL.has(fn.displayName || fn.name)) {
      out.push({ type: "#component", props, children: [], component: fn.displayName || fn.name });
      return;
    }
    let rendered: ReactNode;
    try {
      rendered = fn(props);
    } catch {
      out.push({ type: "#component", props, children: [], component: fn.displayName || fn.name });
      return;
    }
    expandNode(rendered, out);
    return;
  }
  if (typeof type === "object" && type !== null && "render" in type) {
    const render = (type as { render: (props: unknown, ref: unknown) => ReactNode }).render;
    expandNode(render(props, null), out);
    return;
  }
  const children: HostNode[] = [];
  expandNode(props.children as ReactNode, children);
  out.push({ type: String(type), props, children });
}

export function expand(element: ReactNode): HostNode[] {
  const out: HostNode[] = [];
  expandNode(element, out);
  return out;
}

export function findAll(nodes: HostNode[], predicate: (node: HostNode) => boolean): HostNode[] {
  const found: HostNode[] = [];
  const walk = (list: HostNode[]) => {
    for (const node of list) {
      if (predicate(node)) found.push(node);
      walk(node.children);
    }
  };
  walk(nodes);
  return found;
}

export function textOf(node: HostNode): string {
  if (node.type === "#text") return String(node.props.value);
  return node.children.map(textOf).join("");
}

/** The first element with this aria-label (or radiogroup label). */
export function byLabel(nodes: HostNode[], label: string): HostNode {
  const found = findAll(nodes, (n) => n.props["aria-label"] === label);
  if (found.length === 0) throw new Error(`No element labelled "${label}"`);
  return found[0];
}

/** A radio (engine card or segment) inside the group labelled `group`, by its visible text. */
export function radio(nodes: HostNode[], group: string, text: string): HostNode {
  const radios = findAll([byLabel(nodes, group)], (n) => n.props.role === "radio");
  const match = radios.find((r) => textOf(r).includes(text));
  if (!match) throw new Error(`No radio "${text}" in "${group}"; have: ${radios.map(textOf).join(" | ")}`);
  return match;
}

export function click(node: HostNode): void {
  (node.props.onClick as () => void)();
}

export function change(node: HostNode, value: string | boolean): void {
  const target = {
    value: typeof value === "string" ? value : "",
    checked: typeof value === "boolean" ? value : false,
    setCustomValidity: () => undefined,
  };
  (node.props.onChange as (event: unknown) => void)({ target, currentTarget: target });
}

/** The checkbox whose label text contains `text`. */
export function checkbox(nodes: HostNode[], text: string): HostNode {
  const labels = findAll(nodes, (n) => n.type === "label" && textOf(n).includes(text));
  for (const label of labels) {
    const input = findAll([label], (n) => n.type === "input" && n.props.type === "checkbox")[0];
    if (input) return input;
  }
  throw new Error(`No checkbox "${text}"`);
}

/** The control inside the <label> or group whose heading is `text`. */
export function control(nodes: HostNode[], text: string, type: "select" | "input" | "textarea" = "select"): HostNode {
  const holders = findAll(
    nodes,
    (n) => (n.type === "label" || n.props.role === "group") && n.children.length > 0 && textOf(n.children[0]) === text,
  );
  for (const holder of holders) {
    const found = findAll([holder], (n) => n.type === type)[0];
    if (found) return found;
  }
  throw new Error(`No ${type} for "${text}"`);
}
