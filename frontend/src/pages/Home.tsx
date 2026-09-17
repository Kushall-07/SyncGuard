import type { ReactNode } from "react";
import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div>
      <Hero />
      <FeatureGrid />
      <SignalDiagram />
      <ClosingCta />
    </div>
  );
}

function Hero() {
  return (
    <section className="container-page pt-16 pb-20 md:pt-24 md:pb-28">
      <div className="grid lg:grid-cols-[1.1fr_0.9fr] gap-14 lg:gap-10 items-center">
        <div>
          <p className="text-xs uppercase tracking-[0.3em] opacity-60">
            Multimodal Media Verification
          </p>

          <h1 className="font-display text-[2.6rem] leading-[1.08] sm:text-6xl sm:leading-[1.06] md:text-7xl md:leading-[1.04] mt-6">
            Verify
            <br />
            what you hear.
            <br />
            Verify
            <br />
            what you see.
          </h1>

          <p className="mt-8 max-w-md text-base sm:text-lg opacity-70 leading-relaxed">
            Detect synthetic speech and analyze audio-visual synchronization
            using multimodal deep learning.
          </p>

          <div className="mt-10 flex flex-wrap items-center gap-4">
            <LinkButton to="/analyze" variant="primary">
              Start Analysis →
            </LinkButton>
            <LinkButton to="/technology" variant="ghost">
              Explore Technology →
            </LinkButton>
          </div>
        </div>

        <HeroVisual />
      </div>
    </section>
  );
}

function HeroVisual() {
  return (
    <div
      className="relative border aspect-[4/5] sm:aspect-[5/4] lg:aspect-[4/5] overflow-hidden fade-in"
      style={{ borderColor: "var(--page-border)", backgroundColor: "var(--page-surface)" }}
    >
      <svg
        viewBox="0 0 400 480"
        className="absolute inset-0 h-full w-full"
        preserveAspectRatio="xMidYMid slice"
      >
        {/* abstract face silhouette outline */}
        <path
          d="M200 60
             C 130 60 90 120 90 190
             C 90 250 110 300 140 340
             C 160 366 180 380 200 380
             C 220 380 240 366 260 340
             C 290 300 310 250 310 190
             C 310 120 270 60 200 60 Z"
          fill="none"
          stroke="var(--page-fg)"
          strokeOpacity="0.16"
          strokeWidth="1.25"
        />
        <line x1="200" y1="380" x2="200" y2="430" stroke="var(--page-fg)" strokeOpacity="0.12" strokeWidth="1.25" />
        <path
          d="M150 420 C 165 440 235 440 250 420"
          fill="none"
          stroke="var(--page-fg)"
          strokeOpacity="0.12"
          strokeWidth="1.25"
        />

        {/* overlapping waveform strokes crossing the silhouette */}
        <path
          d="M0 250 C 40 230, 70 270, 100 250 S 160 220, 190 250 S 250 280, 280 250 S 340 220, 400 250"
          fill="none"
          stroke="#7c9478"
          strokeWidth="2"
          strokeOpacity="0.85"
        />
        <path
          d="M0 280 C 30 300, 60 250, 100 280 S 170 320, 210 280 S 270 240, 310 280 S 370 310, 400 280"
          fill="none"
          stroke="#ab7440"
          strokeWidth="1.5"
          strokeOpacity="0.7"
        />
        <path
          d="M0 215 C 50 200, 80 235, 120 215 S 190 190, 230 215 S 300 245, 340 215 S 380 195, 400 215"
          fill="none"
          stroke="#f6f1e7"
          strokeWidth="1"
          strokeOpacity="0.25"
        />
      </svg>

      <div className="absolute bottom-0 left-0 right-0 flex items-end justify-center gap-[3px] px-8 pb-8 h-24">
        {BARS.map((h, i) => (
          <span key={i} className="w-[3px] bg-current opacity-30" style={{ height: `${h}%` }} />
        ))}
      </div>
    </div>
  );
}

const BARS = [20, 45, 30, 60, 35, 80, 50, 65, 40, 90, 55, 30, 70, 45, 25, 60, 40, 75, 50, 35, 55, 30, 65, 40];

function FeatureGrid() {
  const features = [
    {
      n: "01",
      title: "Audio Spoof Detection",
      desc: "Detect synthetic or spoofed speech.",
    },
    {
      n: "02",
      title: "Audio-Visual Synchronization",
      desc: "Analyze temporal consistency between audio and visual streams.",
    },
    {
      n: "03",
      title: "Research Driven",
      desc: "Transparent, reproducible multimodal analysis.",
    },
  ];

  return (
    <section className="border-t" style={{ borderColor: "var(--page-border)" }}>
      <div className="container-page py-16 md:py-24">
        <div className="grid md:grid-cols-3 gap-10 md:gap-8">
          {features.map((f) => (
            <div key={f.n} className="pt-6 border-t md:border-t-0" style={{ borderColor: "var(--page-border)" }}>
              <span className="font-display text-sm tracking-widest text-sage-light opacity-90">{f.n}</span>
              <h3 className="font-display text-xl md:text-2xl mt-4">{f.title}</h3>
              <p className="mt-3 text-sm opacity-70 leading-relaxed max-w-xs">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function SignalDiagram() {
  return (
    <section className="border-t" style={{ borderColor: "var(--page-border)" }}>
      <div className="container-page py-16 md:py-28">
        <p className="font-display text-2xl sm:text-3xl md:text-4xl leading-snug max-w-2xl">
          Two signals. One question:
          <br />
          do they belong together?
        </p>

        <div className="mt-16 md:mt-20 flex flex-col sm:flex-row items-center justify-center gap-6 sm:gap-0">
          <DiagramNode label="Audio" />
          <Connector />
          <DiagramNode label="Temporal Analysis" accent />
          <Connector flip />
          <DiagramNode label="Visual" />
        </div>
      </div>
    </section>
  );
}

function DiagramNode({ label, accent = false }: { label: string; accent?: boolean }) {
  return (
    <div
      className={`shrink-0 border px-6 py-5 sm:px-8 sm:py-6 text-center ${accent ? "text-sage-light" : ""}`}
      style={{ borderColor: accent ? "#a3b89e" : "var(--page-border)" }}
    >
      <span className="text-xs sm:text-sm uppercase tracking-[0.2em]">{label}</span>
    </div>
  );
}

function Connector({ flip = false }: { flip?: boolean }) {
  return (
    <div className="flex items-center justify-center w-8 h-10 sm:w-20 sm:h-auto rotate-90 sm:rotate-0">
      <svg width="100%" height="2" viewBox="0 0 80 2" preserveAspectRatio="none">
        <line
          x1={flip ? 80 : 0}
          y1="1"
          x2={flip ? 0 : 80}
          y2="1"
          stroke="var(--page-fg)"
          strokeOpacity="0.3"
          strokeWidth="1"
          strokeDasharray="4 4"
        />
      </svg>
    </div>
  );
}

function ClosingCta() {
  return (
    <section className="border-t" style={{ borderColor: "var(--page-border)" }}>
      <div className="container-page py-20 md:py-32 flex flex-col items-center text-center">
        <h2 className="font-display text-3xl sm:text-4xl md:text-5xl max-w-xl leading-tight">
          Ready to analyze a piece of media?
        </h2>
        <div className="mt-10">
          <LinkButton to="/analyze" variant="primary">
            Start Analysis →
          </LinkButton>
        </div>
      </div>
    </section>
  );
}

function LinkButton({
  to,
  variant = "primary",
  children,
}: {
  to: string;
  variant?: "primary" | "ghost";
  children: ReactNode;
}) {
  const base =
    "inline-flex items-center gap-2 px-6 py-3 text-sm tracking-wide transition-all duration-300 ease-editorial";
  const variants: Record<string, string> = {
    primary: "bg-sage text-ink hover:bg-sage-light active:scale-[0.98]",
    ghost: "border hover:border-sage hover:text-sage-light active:scale-[0.98]",
  };
  const style = variant === "ghost" ? { borderColor: "var(--page-border)" } : undefined;

  return (
    <Link to={to} className={`${base} ${variants[variant]}`} style={style}>
      {children}
    </Link>
  );
}
