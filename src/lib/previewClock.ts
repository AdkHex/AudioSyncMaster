/** The in-app player's clock arithmetic, kept pure so it can be tested.
 *
 *  The player loads a 30 s window around the cursor -- ten seconds before
 *  it, twenty after, clamped to the film -- and everything else the
 *  transport does is arithmetic on that window: the loop region, the
 *  timecode, frame stepping and the drift check. All of it lives here
 *  rather than in the component, so the numbers are provable without a
 *  video element. Times are seconds on the video's clock; the player maps
 *  them onto the excerpt's media time by subtracting the window's start. */

export const WINDOW_BEFORE_S = 10;
export const WINDOW_AFTER_S = 20;
export const LOOP_S = 4;
/** Picture and sound are re-locked once they are this far apart: 20 ms
 *  is about where a mismatch becomes audible. */
export const DRIFT_S = 0.02;

/** The 30 s window around a cursor position, clamped to the film. */
export function windowAround(cursorS: number, durationS: number): { startS: number; endS: number } {
  const length = Math.min(WINDOW_BEFORE_S + WINDOW_AFTER_S, durationS);
  const startS = Math.min(Math.max(cursorS - WINDOW_BEFORE_S, 0), Math.max(0, durationS - length));
  return { startS, endS: startS + length };
}

/** A 4 s loop centred on `atS`, clamped inside the loaded window. */
export function loopAround(atS: number, startS: number, endS: number): { a: number; b: number } {
  const length = Math.min(LOOP_S, endS - startS);
  const a = Math.min(Math.max(atS - length / 2, startS), Math.max(startS, endS - length));
  return { a, b: a + length };
}

/** The frame playing at time `t`, counting from the film's start. The 1e-9
 *  tolerance keeps a time computed as a frame's exact position from
 *  landing on the next frame by float noise, without stealing the last
 *  frame of a second: a time 1e-9 before a boundary is still the frame
 *  before it. */
export function frameIndex(t: number, frameS: number): number {
  return Math.floor(t / frameS + 1e-9);
}

/** h:mm:ss:ff from a time on the video's clock, as editors print it: the
 *  frame number within the current second, zero-padded to two digits.
 *  The frame count is the frame index modulo the frames in a second --
 *  rounded, so 29.97 fps counts 0..29 the way an editor does. */
export function timecode(t: number, frameS: number): string {
  const ms = Math.max(0, Math.round(t * 1000));
  const hours = Math.floor(ms / 3_600_000);
  const minutes = Math.floor((ms % 3_600_000) / 60_000);
  const seconds = Math.floor((ms % 60_000) / 1000);
  const framesPerSecond = Math.max(1, Math.round(1 / frameS));
  const frame = frameIndex(t, frameS) % framesPerSecond;
  return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}:${String(frame).padStart(2, "0")}`;
}

/** The next frame back or forward from `t`, staying inside the window.
 *  Stepping past an end stays put. */
export function stepFrame(
  t: number,
  frameS: number,
  direction: -1 | 1,
  startS: number,
  endS: number,
): number {
  const target = (frameIndex(t, frameS) + direction) * frameS;
  return Math.min(Math.max(target, startS), endS);
}

/** Whether picture and sound have drifted far enough apart to need the
 *  sources restarted at the picture's time. */
export function needsResync(videoTime: number, audioTime: number): boolean {
  return Math.abs(videoTime - audioTime) > DRIFT_S;
}

/** Whether a time on the film's clock is inside the loaded window. */
export function inWindow(t: number, startS: number, endS: number): boolean {
  return t >= startS && t <= endS;
}
