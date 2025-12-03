type StatCardProps = {
  label: string;
  value: string;
  sublabel?: string;
  tone?: "positive" | "neutral" | "negative";
};

const toneToColor: Record<NonNullable<StatCardProps["tone"]>, string> = {
  positive: "text-emerald-400",
  neutral: "text-slate-200",
  negative: "text-rose-400",
};

export function StatCard({
  label,
  value,
  sublabel,
  tone = "neutral",
}: StatCardProps) {
  return (
    <div className="finlove-card p-4">
      <p className="text-xs font-medium uppercase tracking-[0.08em] text-slate-400">
        {label}
      </p>
      <p className={`mt-2 text-xl font-semibold ${toneToColor[tone]}`}>
        {value}
      </p>
      {sublabel && (
        <p className="mt-1 text-[11px] text-slate-400">{sublabel}</p>
      )}
    </div>
  );
}


