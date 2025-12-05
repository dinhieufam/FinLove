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
    <div className="finlove-card p-5">
      <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
        {label}
      </p>
      <p className={`mt-3 text-2xl font-bold ${toneToColor[tone]}`}>
        {value}
      </p>
      {sublabel && (
        <p className="mt-2 text-xs text-slate-400">{sublabel}</p>
      )}
    </div>
  );
}


