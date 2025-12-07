"use client";

import Link from "next/link";
import { Marquee } from "@components/ui/marquee";

export default function LandingPage() {
  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="finlove-gradient-bg min-h-screen">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 border-b border-slate-800/70 bg-slate-950/80 backdrop-blur-sm">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
          <Link href="/" className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-500/20 ring-1 ring-emerald-400/40">
              <span className="text-lg font-semibold text-emerald-400">ƒ</span>
            </div>
            <div>
              <p className="text-sm font-semibold tracking-tight">FinLove</p>
              <p className="text-[11px] text-slate-400">Smart Portfolio Lab</p>
            </div>
          </Link>
          <div className="hidden items-center gap-6 md:flex">
            <button
              onClick={() => scrollToSection("home")}
              className="text-sm text-slate-300 transition hover:text-emerald-400"
            >
              Home
            </button>
            <button
              onClick={() => scrollToSection("about")}
              className="text-sm text-slate-300 transition hover:text-emerald-400"
            >
              About
            </button>
            <button
              onClick={() => scrollToSection("features")}
              className="text-sm text-slate-300 transition hover:text-emerald-400"
            >
              Features
            </button>
            <button
              onClick={() => scrollToSection("team")}
              className="text-sm text-slate-300 transition hover:text-emerald-400"
            >
              Our Team
            </button>
            <button
              onClick={() => scrollToSection("contact")}
              className="text-sm text-slate-300 transition hover:text-emerald-400"
            >
              Contact
            </button>
            <Link
              href="/dashboard"
              className="rounded-xl bg-emerald-500 px-4 py-2 text-sm font-medium text-slate-950 transition hover:bg-emerald-400"
            >
              Launch Dashboard
            </Link>
          </div>
        </div>
      </nav>

      {/* Home/Hero Section */}
      <section id="home" className="px-4 py-20 sm:px-6 lg:px-8">
        <div className="mx-auto flex max-w-6xl flex-col gap-10 lg:flex-row lg:items-center">
          {/* Left: hero content */}
          <div className="max-w-xl space-y-6">
            <span className="finlove-pill">Smart Investing Made Simple</span>
            <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
              FinLove:
              <span className="inline-block bg-gradient-to-r from-emerald-300 to-cyan-300 bg-clip-text text-transparent">
                {" "}
                invest with confidence.
              </span>
            </h1>
            <p className="finlove-subtle">
              Build stable, safer portfolios without needing a PhD in finance.
              We use advanced technology to help you balance risk and return,
              so you can sleep easier at night.
            </p>
            <div className="flex flex-wrap items-center gap-3">
              <Link
                href="/dashboard"
                className="inline-flex items-center justify-center rounded-xl bg-emerald-500 px-6 py-2.5 text-sm font-medium text-slate-950 shadow-md transition hover:bg-emerald-400"
              >
                Start Building
              </Link>
              <a
                href="https://github.com/dinhieufam/FinLove"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center justify-center rounded-xl border border-slate-700 bg-slate-900/60 px-6 py-2.5 text-sm font-medium text-slate-100 transition hover:border-slate-500 hover:bg-slate-900"
              >
                View on GitHub
              </a>
            </div>
            <div className="mt-4 grid gap-3 text-xs text-slate-300 sm:grid-cols-3">
              <div>
                <p className="font-semibold text-slate-100">Smart Analysis</p>
                <p>Filters out market noise to find true trends.</p>
              </div>
              <div>
                <p className="font-semibold text-slate-100">Portfolio Balance</p>
                <p>Optimizes for safety, growth, or efficiency.</p>
              </div>
              <div>
                <p className="font-semibold text-slate-100">History Testing</p>
                <p>See how your strategy would have performed.</p>
              </div>
            </div>
          </div>

          {/* Right: feature / metrics panel */}
          <div className="flex-1">
            <div className="finlove-card relative overflow-hidden p-5 sm:p-6">
              <div className="pointer-events-none absolute inset-x-0 -top-8 h-32 bg-gradient-to-b from-emerald-400/20 to-transparent blur-3xl" />
              <div className="relative flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <p className="text-xs font-medium uppercase tracking-[0.14em] text-emerald-300">
                    Strategy snapshot
                  </p>
                  <p className="mt-1 text-sm text-slate-300">
                    Example performance vs. standard market benchmark.
                  </p>
                </div>
                <div className="text-right text-xs text-slate-400">
                  <p>Universe: Sector ETFs</p>
                  <p>Goal: Max Efficiency</p>
                  <p>Risk Level: Balanced</p>
                </div>
              </div>

              <div className="mt-5 grid gap-4 sm:grid-cols-3">
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-slate-400">
                    Total Return
                  </p>
                  <p className="mt-1 text-xl font-semibold text-emerald-400">
                    +142.7%
                  </p>
                  <p className="mt-1 text-[11px] text-slate-400">
                    Benchmark: +98.3%
                  </p>
                </div>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-slate-400">
                    Worst Drop
                  </p>
                  <p className="mt-1 text-xl font-semibold text-emerald-300">
                    -16.4%
                  </p>
                  <p className="mt-1 text-[11px] text-slate-400">
                    Benchmark: -24.1%
                  </p>
                </div>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-slate-400">
                    Efficiency Score
                  </p>
                  <p className="mt-1 text-xl font-semibold text-emerald-400">
                    1.46
                  </p>
                  <p className="mt-1 text-[11px] text-slate-400">
                    Higher is better
                  </p>
                </div>
              </div>

              <div className="mt-5 flex items-center justify-between border-t border-slate-800/70 pt-3 text-[11px] text-slate-400">
                <p>
                  Daily data from trusted sources.
                </p>
                <p>Built for everyone.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="border-t border-slate-800/70 px-4 py-20 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-6xl">
          <div className="mb-12 text-center">
            <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              About <span className="inline-block bg-gradient-to-r from-emerald-300 to-cyan-300 bg-clip-text text-transparent">FinLove</span>
            </h2>
            <p className="mt-4 text-slate-400 max-w-2xl mx-auto">
              A powerful tool designed to help you build smarter, safer investment portfolios.
            </p>
          </div>
          <div className="grid gap-6 md:grid-cols-2">
            <div className="finlove-card p-6">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-500/20 ring-1 ring-emerald-400/40">
                <span className="text-2xl">🎯</span>
              </div>
              <h3 className="mb-2 text-lg font-semibold text-slate-100">Our Mission</h3>
              <p className="text-sm text-slate-300">
                We believe professional-grade investment tools shouldn't be reserved for
                Wall Street. FinLove combines advanced technology with a simple interface
                to help anyone create stable, risk-aware portfolios.
              </p>
            </div>
            <div className="finlove-card p-6">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-500/20 ring-1 ring-emerald-400/40">
                <span className="text-2xl">🎓</span>
              </div>
              <h3 className="mb-2 text-lg font-semibold text-slate-100">Learn Investing</h3>
              <p className="text-sm text-slate-300">
                Whether you're a student or just starting out, FinLove provides a
                sandbox to explore different strategies. See how different choices
                affect your risk and return in a safe environment.
              </p>
            </div>
            <div className="finlove-card p-6">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-500/20 ring-1 ring-emerald-400/40">
                <span className="text-2xl">⚡</span>
              </div>
              <h3 className="mb-2 text-lg font-semibold text-slate-100">Fast & Easy</h3>
              <p className="text-sm text-slate-300">
                No coding required. Our platform handles all the complex calculations
                instantly, so you can focus on making the right decisions for your money.
              </p>
            </div>
            <div className="finlove-card p-6">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-500/20 ring-1 ring-emerald-400/40">
                <span className="text-2xl">📊</span>
              </div>
              <h3 className="mb-2 text-lg font-semibold text-slate-100">Interactive Dashboard</h3>
              <p className="text-sm text-slate-300">
                Visualize your portfolio's potential. Our intuitive dashboard shows you
                exactly what's happening with clear charts and simple metrics.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="border-t border-slate-800/70 px-4 py-20 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-6xl">
          <div className="mb-12 text-center">
            <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              Powerful <span className="inline-block bg-gradient-to-r from-emerald-300 to-cyan-300 bg-clip-text text-transparent">Features</span>
            </h2>
            <p className="mt-4 text-slate-400 max-w-2xl mx-auto">
              Everything you need to build a portfolio you can trust
            </p>
          </div>

          {/* Risk Models */}
          <div className="mb-16">
            <h3 className="mb-6 text-xl font-semibold text-emerald-400">Smart Analysis</h3>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">Stable Estimates</h4>
                <p className="text-xs text-slate-400">
                  Uses advanced math to filter out short-term market noise and find
                  reliable long-term trends.
                </p>
              </div>
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">Precision Tracking</h4>
                <p className="text-xs text-slate-400">
                  Identifies hidden relationships between different assets to
                  better diversify your portfolio.
                </p>
              </div>
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">Volatility Protection</h4>
                <p className="text-xs text-slate-400">
                  Adapts to changing market conditions. When markets get bumpy,
                  our models adjust to keep you safe.
                </p>
              </div>
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">Dynamic Correlations</h4>
                <p className="text-xs text-slate-400">
                  Understands that assets move differently in crashes vs. booms,
                  helping protect you when it matters most.
                </p>
              </div>
            </div>
          </div>

          {/* Optimization Methods */}
          <div className="mb-16">
            <h3 className="mb-6 text-xl font-semibold text-emerald-400">Portfolio Balance</h3>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">Classic Balance</h4>
                <p className="text-xs text-slate-400">
                  The Nobel Prize-winning approach. Finds the best possible return
                  for your comfort level with risk.
                </p>
              </div>
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">Safety First</h4>
                <p className="text-xs text-slate-400">
                  Constructs the most stable portfolio possible. Ideal for preserving
                  wealth and avoiding sleepless nights.
                </p>
              </div>
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">Max Efficiency</h4>
                <p className="text-xs text-slate-400">
                  Targets the highest "bang for your buck" (return per unit of risk).
                  Great for long-term growth.
                </p>
              </div>
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">Market Views</h4>
                <p className="text-xs text-slate-400">
                  Combines market data with your own intuition. A flexible way to
                  express your investment ideas.
                </p>
              </div>
              <div className="finlove-card p-5">
                <h4 className="mb-2 font-semibold text-slate-100">Crash Protection</h4>
                <p className="text-xs text-slate-400">
                  Specifically designed to minimize losses during extreme market
                  crashes and "black swan" events.
                </p>
              </div>
            </div>
          </div>

          {/* Backtesting & Dashboard */}
          <div className="grid gap-6 md:grid-cols-2">
            <div className="finlove-card p-6">
              <h3 className="mb-4 text-xl font-semibold text-emerald-400">History Testing</h3>
              <ul className="space-y-3 text-sm text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Simple Test:</strong> See how your strategy would have done in the past</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Realistic Simulation:</strong> "Walk-forward" testing mimics real-life investing</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Cost Analysis:</strong> Accounts for trading fees so you see net returns</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Smart Rebalancing:</strong> Keeps your portfolio on track automatically</span>
                </li>
              </ul>
            </div>
            <div className="finlove-card p-6">
              <h3 className="mb-4 text-xl font-semibold text-emerald-400">Dashboard Features</h3>
              <ul className="space-y-3 text-sm text-slate-300">
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Easy Interface:</strong> Simple web app, no installation needed</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Clear Charts:</strong> Visualize growth, drawdowns, and asset weights</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Key Metrics:</strong> All the important numbers in one place</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-emerald-400">✓</span>
                  <span><strong>Company Info:</strong> Detailed data for every stock in your portfolio</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>



      {/* Our Team Section */}
      <section id="team" className="border-t border-slate-800/70 py-20">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8">
          <div className="mb-12 text-center">
            <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              Our <span className="inline-block bg-gradient-to-r from-emerald-300 to-cyan-300 bg-clip-text text-transparent">Team</span>
            </h2>
            <p className="mt-4 text-slate-400 max-w-2xl mx-auto">
              A collaborative effort by passionate students and researchers
            </p>
          </div>
        </div>

        <div className="relative flex w-full flex-col items-center justify-center overflow-hidden py-8">
          <Marquee pauseOnHover className="[--duration:20s]">
            {[
              { name: "Nguyen Van Duy Anh", role: "Team Member", img: "/images/duyanh.jpg" },
              { name: "Pham Dinh Hieu", role: "Team Member", img: "/images/dinhhieu.jpg" },
              { name: "Cao Pham Minh Dang", role: "Team Member", img: "/images/minhdang.png" },
              { name: "Tran Anh Chuong", role: "Team Member", img: "/images/thanhchuong.jpg" },
              { name: "Ngo Dinh Khanh", role: "Team Member", img: "/images/dinhkhanh.jpg" },
              { name: "Nguyen Huy Hung", role: "Project Advisor", img: "/images/huyhung.jpg" },
              { name: "Dr. Mo El-Haj", role: "Project Advisor", img: "/images/mo_elhaj.jpg" },
            ].map((member, i) => (
              <div
                key={i}
                className="finlove-card mx-4 flex w-64 shrink-0 flex-col items-center p-6 text-center"
              >
                {member.img ? (
                  <div className="mb-4 h-24 w-24 overflow-hidden rounded-full ring-1 ring-emerald-400/40">
                    <img src={member.img} alt={member.name} className="h-full w-full object-cover" />
                  </div>
                ) : (
                  <div className="mb-4 flex h-24 w-24 items-center justify-center rounded-full bg-emerald-500/20 ring-1 ring-emerald-400/40">
                    <span className="text-2xl font-semibold text-emerald-400">{member.initial}</span>
                  </div>
                )}
                <h3 className="mb-1 text-lg font-semibold text-slate-100">{member.name}</h3>
                <p className="text-sm text-slate-400">{member.role}</p>
              </div>
            ))}
          </Marquee>
          <div className="pointer-events-none absolute inset-y-0 left-0 w-1/3 bg-gradient-to-r from-slate-950 dark:from-background"></div>
          <div className="pointer-events-none absolute inset-y-0 right-0 w-1/3 bg-gradient-to-l from-slate-950 dark:from-background"></div>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="border-t border-slate-800/70 px-4 py-20 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-4xl">
          <div className="mb-12 text-center">
            <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              Get in <span className="inline-block bg-gradient-to-r from-emerald-300 to-cyan-300 bg-clip-text text-transparent">Touch</span>
            </h2>
            <p className="mt-4 text-slate-400 max-w-2xl mx-auto">
              Have questions, suggestions, or want to contribute? We'd love to hear from you!
            </p>
          </div>
          <div className="finlove-card p-8">
            <div className="grid gap-8 md:grid-cols-2">
              <div>
                <h3 className="mb-4 text-lg font-semibold text-slate-100">Project Information</h3>
                <div className="space-y-4 text-sm text-slate-300">
                  <div>
                    <p className="mb-1 font-medium text-slate-200">GitHub Repository</p>
                    <a
                      href="https://github.com/dinhieufam/FinLove"
                      target="_blank"
                      rel="noreferrer"
                      className="text-emerald-400 hover:text-emerald-300 transition"
                    >
                      github.com/dinhieufam/FinLove
                    </a>
                  </div>
                  <div>
                    <p className="mb-1 font-medium text-slate-200">License</p>
                    <p className="text-slate-400">See LICENSE file for details</p>
                  </div>
                  <div>
                    <p className="mb-1 font-medium text-slate-200">Documentation</p>
                    <p className="text-slate-400">
                      Comprehensive guides available in the repository, including:
                    </p>
                    <ul className="mt-2 ml-4 list-disc space-y-1 text-slate-400">
                      <li>README.md - Main documentation</li>
                      <li>DATA.md - Data download and caching</li>
                      <li>QUICK_START.md - Quick start guide</li>
                    </ul>
                  </div>
                </div>
              </div>
              <div>
                <h3 className="mb-4 text-lg font-semibold text-slate-100">Support & Resources</h3>
                <div className="space-y-4 text-sm text-slate-300">
                  <div>
                    <p className="mb-1 font-medium text-slate-200">Issues & Bug Reports</p>
                    <p className="text-slate-400">
                      Please report issues on our GitHub repository's Issues page
                    </p>
                  </div>
                  <div>
                    <p className="mb-1 font-medium text-slate-200">Data Questions</p>
                    <p className="text-slate-400">
                      Refer to DATA.md for data-related questions and caching information
                    </p>
                  </div>
                  <div>
                    <p className="mb-1 font-medium text-slate-200">Usage Help</p>
                    <p className="text-slate-400">
                      Check the README.md and QUICK_START.md for detailed usage instructions
                    </p>
                  </div>
                </div>
              </div>
            </div>
            <div className="mt-8 flex flex-wrap items-center justify-center gap-4 border-t border-slate-800/70 pt-6">
              <Link
                href="/dashboard"
                className="inline-flex items-center justify-center rounded-xl bg-emerald-500 px-6 py-2.5 text-sm font-medium text-slate-950 shadow-md transition hover:bg-emerald-400"
              >
                Launch Dashboard
              </Link>
              <a
                href="https://github.com/dinhieufam/FinLove"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center justify-center rounded-xl border border-slate-700 bg-slate-900/60 px-6 py-2.5 text-sm font-medium text-slate-100 transition hover:border-slate-500 hover:bg-slate-900"
              >
                View on GitHub
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-800/70 px-4 py-8 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-6xl">
          <div className="flex flex-col items-center justify-between gap-4 md:flex-row">
            <div className="flex items-center gap-2">
              <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-emerald-500/20 ring-1 ring-emerald-400/40">
                <span className="text-sm font-semibold text-emerald-400">ƒ</span>
              </div>
              <p className="text-sm text-slate-400">
                FinLove — Smart Portfolio Construction
              </p>
            </div>
            <p className="text-xs text-slate-500">
              Built for everyone · Data from Yahoo Finance
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}


