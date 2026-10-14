import fs from "node:fs";
import { createRequire } from "node:module";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const { checkBundle, checkConfig, checkManifest } = require("./windows-installer.cjs") as {
  checkConfig: (base: unknown, windows: unknown) => string[];
  checkBundle: (dir: string) => { problems: string[] };
  checkManifest: (latest: unknown) => string[];
};

const read = (file: string) => JSON.parse(fs.readFileSync(path.join(__dirname, "..", "src-tauri", file), "utf8"));
const base = read("tauri.conf.json");
const windows = read("tauri.windows.conf.json");

describe("the Windows config", () => {
  it("builds only the per-user NSIS installer, and updates through it passively", () => {
    expect(checkConfig(base, windows)).toEqual([]);
  });

  it("keeps every target on macOS and Linux", () => {
    expect(base.bundle.targets).toBe("all");
  });

  it("is refused when Windows would build an MSI or install for every account", () => {
    const allTargets = { ...windows, bundle: { ...windows.bundle, targets: "all" } };
    expect(checkConfig(base, allTargets)[0]).toMatch(/MSI installs for every account/);
    const perMachine = { ...windows, bundle: { ...windows.bundle, windows: { nsis: { installMode: "perMachine" } } } };
    expect(checkConfig(base, perMachine)).toEqual(['bundle.windows.nsis.installMode is "perMachine", not "currentUser"']);
    const both = { ...windows, bundle: { ...windows.bundle, windows: { nsis: { installMode: "both" } } } };
    expect(checkConfig(base, both)).toHaveLength(1);
  });
});

describe("the published manifest", () => {
  const setup = (version: string) => ({
    url: `https://github.com/AdkHex/AudioSyncMaster/releases/download/v${version}/AudioSyncMaster_${version}_x64-setup.exe`,
    signature: "c2ln",
  });

  it("offers Windows the NSIS -setup.exe and nothing else", () => {
    expect(checkManifest({ platforms: { "windows-x86_64": setup("2.22.0"), "windows-x86_64-nsis": setup("2.22.0") } })).toEqual([]);
  });

  it("flags a manifest that still offers the MSI, as v2.21.0's did", () => {
    const old = {
      platforms: {
        "windows-x86_64": setup("2.21.0"),
        "windows-x86_64-msi": { url: ".../AudioSyncMaster_2.21.0_x64_en-US.msi", signature: "c2ln" },
      },
    };
    expect(checkManifest(old)).toEqual(["latest.json still offers an MSI (windows-x86_64-msi)"]);
    expect(checkManifest({ platforms: { "windows-x86_64": { url: "x.msi", signature: "s" } } })[0]).toMatch(/not the NSIS -setup.exe/);
    expect(checkManifest({ platforms: {} })).toEqual(["latest.json has no windows-x86_64 entry"]);
  });
});

describe("a Windows build's output", () => {
  const build = (files: Record<string, string>) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "asm-win-bundle-"));
    for (const [name, text] of Object.entries(files)) {
      fs.mkdirSync(path.dirname(path.join(dir, name)), { recursive: true });
      fs.writeFileSync(path.join(dir, name), text);
    }
    return dir;
  };
  const nsi = '!define INSTALLMODE "currentUser"\nRequestExecutionLevel user\n';

  it("passes with a per-user -setup.exe and no MSI", () => {
    const dir = build({ "bundle/nsis/AudioSyncMaster_2.22.0_x64-setup.exe": "", "nsis/x64/installer.nsi": nsi });
    expect(checkBundle(dir).problems).toEqual([]);
  });

  it("fails on an MSI, a missing installer, or a script that asks for an administrator", () => {
    const dir = build({
      "bundle/msi/AudioSyncMaster_2.22.0_x64_en-US.msi": "",
      "nsis/x64/installer.nsi": '!define INSTALLMODE "perMachine"\nRequestExecutionLevel admin\n',
    });
    expect(checkBundle(dir).problems).toHaveLength(4);
  });
});
