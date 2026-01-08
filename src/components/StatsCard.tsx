import { ReactNode } from "react";

interface StatsCardProps {
  label: string;
  value: string | number;
  icon: ReactNode;
  trend?: "up" | "down" | "neutral";
  subtext?: string;
}

export const StatsCard = ({ label, value, icon, trend, subtext }: StatsCardProps) => {
  const trendColors = {
    up: "text-success",
    down: "text-destructive",
    neutral: "text-muted-foreground"
  };

  return (
    <div className="p-5 rounded-xl bg-card border border-border shadow-apple-sm hover:shadow-apple-md transition-all hover:-translate-y-0.5">
      <div className="flex items-start justify-between mb-3">
        <div className="p-2.5 rounded-lg bg-accent text-accent-foreground">
          {icon}
        </div>
        {trend && (
          <span className={`text-xs font-medium ${trendColors[trend]}`}>
            {trend === "up" ? "↑" : trend === "down" ? "↓" : "—"}
          </span>
        )}
      </div>
      
      <p className="text-2xl font-bold tracking-tight">{value}</p>
      <p className="text-sm text-muted-foreground mt-1">{label}</p>
      
      {subtext && (
        <p className="text-xs text-muted-foreground mt-2 pt-2 border-t border-border">
          {subtext}
        </p>
      )}
    </div>
  );
};
