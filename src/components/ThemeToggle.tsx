import { Monitor, Moon, Sun } from "lucide-react";

import { cx } from "@/lib/cx";
import { useTheme } from "./ThemeProvider";

const OPTIONS = [
  { value: "light", icon: Sun, label: "Light" },
  { value: "dark", icon: Moon, label: "Dark" },
  { value: "system", icon: Monitor, label: "Auto" },
] as const;

/** Segmented theme control. Lives in Settings rather than the header, so the
 *  app chrome carries only actions people reach for during a run. */
export function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  return (
    <div
      className="flex items-center gap-0.5 rounded-lg border border-border bg-sunken p-0.5"
      role="radiogroup"
      aria-label="Theme"
    >
      {OPTIONS.map(({ value, icon: Icon, label }) => (
        <button
          key={value}
          type="button"
          role="radio"
          aria-checked={theme === value}
          onClick={() => setTheme(value)}
          className={cx(
            "flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[11.5px] transition-colors",
            theme === value
              ? "bg-card font-semibold text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          <Icon className="h-3 w-3" aria-hidden />
          {label}
        </button>
      ))}
    </div>
  );
}
