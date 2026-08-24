/** Join conditional class names. Kept out of the component modules so they
 *  export components only, which is what React Fast Refresh needs. */
export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}
