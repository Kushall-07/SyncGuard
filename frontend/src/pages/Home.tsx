import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import Reveal from "../components/scroll/Reveal";
import ForensicScene from "../components/scene/ForensicScene";

export default function Home() {
  return (
    <div className="relative bg-grain">
      <ForensicScene />
      <ProductPreview />
      <FinalCta />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Product preview — bridge from the scene to the real application
// ---------------------------------------------------------------------------

function ProductPreview() {
  const steps = ["Upload", "Processing", "Analysis", "Evidence", "Result"];
  return (
    <section className="border-t" style={{ borderColor: "var(--page-border)" }}>
      <div className="container-page py-20 md:py-28">
        <Reveal>
          <p className="font-mono text-xs uppercase tracking-[0.3em] text-signal/80">Workspace</p>
          <h2 className="font-display text-4xl sm:text-5xl mt-4 max-w-xl">A real analysis workstation.</h2>
          <p className="mt-5 max-w-lg text-base sm:text-lg text-ink-soft leading-relaxed">
            Not a marketing mockup — this is the actual workflow the Analyze page runs,
            backed by the real inference pipeline. Run audio alone for spoof detection,
            or audio and video together for synchronization analysis.
          </p>
        </Reveal>

        <Reveal stagger={60} className="mt-12 flex flex-wrap items-stretch gap-3">
          {steps.map((step, i) => (
            <div key={step} className="flex items-center gap-3">
              <div
                className="border px-5 py-4 font-mono text-sm uppercase tracking-[0.15em] text-ink-soft"
                style={{ borderColor: "var(--page-border)" }}
              >
                {step}
              </div>
              {i < steps.length - 1 && <span className="text-ink-faint">→</span>}
            </div>
          ))}
        </Reveal>

        <div className="mt-12">
          <LinkButton to="/analyze" variant="primary">
            Analyze Your Media
          </LinkButton>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Final CTA — quiet, the scene has already made its case
// ---------------------------------------------------------------------------

function FinalCta() {
  return (
    <section className="border-t" style={{ borderColor: "var(--page-border)" }}>
      <div className="container-page py-24 md:py-36 flex flex-col items-center text-center">
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-signal/80">SyncGuard</p>
        <h2 className="font-display text-4xl sm:text-5xl md:text-6xl max-w-xl leading-tight mt-5">
          Verify the signal.
          <br />
          Understand the media.
        </h2>
        <p className="mt-5 max-w-md text-base sm:text-lg text-ink-soft leading-relaxed">
          Every result ships with the evidence behind it, and an honest account
          of what it doesn't prove.
        </p>
        <div className="mt-10">
          <LinkButton to="/analyze" variant="primary">
            Analyze Media
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
    "inline-flex items-center gap-2 px-7 py-3.5 text-base tracking-wide transition-all duration-300 ease-editorial";
  const variants: Record<string, string> = {
    primary: "bg-signal text-[#061012] hover:bg-signal-light active:scale-[0.98]",
    ghost: "border hover:border-signal hover:text-signal active:scale-[0.98]",
  };
  const style = variant === "ghost" ? { borderColor: "var(--page-border)" } : undefined;

  return (
    <Link to={to} className={`${base} ${variants[variant]}`} style={style}>
      {children}
    </Link>
  );
}
