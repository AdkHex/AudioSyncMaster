import { Sun, Moon, Monitor } from "lucide-react";
import { useTheme } from "./ThemeProvider";

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  const options: { value: "light" | "dark" | "system"; icon: React.ReactNode; label: string }[] = [
    { value: "light", icon: <Sun className="w-3.5 h-3.5" />, label: "Light" },
    { value: "dark", icon: <Moon className="w-3.5 h-3.5" />, label: "Dark" },
    { value: "system", icon: <Monitor className="w-3.5 h-3.5" />, label: "System" },
  ];

  return (
    <div className="flex items-center gap-0.5 bg-secondary p-0.5 rounded-md">
      {options.map((option) => (
        <button
          key={option.value}
          onClick={() => setTheme(option.value)}
          className={`p-1.5 rounded transition-colors ${
            theme === option.value
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:text-foreground"
          }`}
          title={option.label}
        >
          {option.icon}
        </button>
      ))}
    </div>
  );
}
