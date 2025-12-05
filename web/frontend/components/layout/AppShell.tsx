import Link from "next/link";
import type { ReactNode } from "react";

type Props = {
  children: ReactNode;
  sidebarContent?: ReactNode;
};

export function AppShell({ children, sidebarContent }: Props) {
  const sidebarWidth = sidebarContent ? "w-72" : "w-56";

  return (
    <div className="finlove-gradient-bg min-h-screen">
      <div className="mx-auto flex min-h-screen max-w-6xl gap-6 px-4 py-5 sm:px-6 lg:px-8">
        {/* Sidebar */}
        <aside
          className={`hidden ${sidebarWidth} shrink-0 flex-col justify-between rounded-2xl border border-slate-800/80 bg-slate-950/80 px-4 py-5 shadow-soft-elevated md:flex`}
        >
          <div className="space-y-6">
            <Link href="/" className="flex items-center gap-2">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-500/20 ring-1 ring-emerald-400/40">
                <span className="text-lg font-semibold text-emerald-400">ƒ</span>
              </div>
              <div>
                <p className="text-sm font-semibold tracking-tight">FinLove</p>
                <p className="text-[11px] text-slate-400">
                  Risk-Aware Portfolio Lab
                </p>
              </div>
            </Link>
            {sidebarContent ? (
              <div className="space-y-5">{sidebarContent}</div>
            ) : (
              <nav className="space-y-1 text-sm">
                {navItems.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className="flex items-center justify-between rounded-xl px-3 py-2 text-slate-300 transition hover:bg-slate-800/70 hover:text-slate-50"
                  >
                    <span>{item.label}</span>
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-400/70" />
                  </Link>
                ))}
              </nav>
            )}
          </div>
          <div className="space-y-2 border-t border-slate-800/70 pt-3 text-[11px] text-slate-400">
            <p className="font-medium text-slate-300">Session profile</p>
            <p>Configure portfolios, run walk-forward tests, and export results.</p>
          </div>
        </aside>

        {/* Main content */}
        <div className="flex min-h-screen flex-1 flex-col">
          {/* Top bar */}
          <header className="mb-5 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2 md:hidden">
              <Link href="/" className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-emerald-500/20 ring-1 ring-emerald-400/40">
                  <span className="text-base font-semibold text-emerald-400">
                    Ƒ
                  </span>
                </div>
                <span className="text-sm font-semibold tracking-tight">
                  FinLove
                </span>
              </Link>
            </div>
            <div className="flex flex-1 items-center justify-end gap-3">
              <span className="hidden text-xs text-slate-400 sm:inline">
                Data from Yahoo Finance · Powered by FinLove engine
              </span>
              <div className="flex items-center gap-2 rounded-full border border-slate-700/80 bg-slate-900/80 px-3 py-1.5">
                <div className="h-2 w-2 rounded-full bg-emerald-400" />
                <span className="text-[11px] font-medium text-slate-200">
                  Market Open (demo)
                </span>
              </div>
            </div>
          </header>

          <main className="flex-1 pb-8">{children}</main>
        </div>
      </div>
    </div>
  );
}


