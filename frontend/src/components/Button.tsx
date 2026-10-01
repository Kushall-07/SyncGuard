import type { ButtonHTMLAttributes, ReactNode } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "ghost";
  children: ReactNode;
}

export default function Button({ variant = "primary", className = "", children, ...rest }: ButtonProps) {
  // No blanket `disabled:opacity-40` here: the primary variant's text is a
  // near-black (#061012) tuned for contrast against the bright cyan fill.
  // Dropping the whole element's opacity composites that near-black text
  // over the page's near-black canvas, which makes the label disappear
  // entirely while the (brighter) fill and shadow stay faintly visible —
  // exactly the "empty glowing box, no text" bug this replaces. Disabled
  // state gets its own explicit, legible colors instead.
  const base =
    "inline-flex items-center gap-2 px-7 py-3.5 text-base font-medium tracking-wide transition-all duration-300 ease-editorial disabled:cursor-not-allowed focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal";
  const variants: Record<string, string> = {
    primary:
      "bg-signal text-[#061012] shadow-[0_0_0_1px_rgba(34,211,238,0.5),0_8px_24px_-8px_rgba(34,211,238,0.55)] hover:bg-signal-light hover:shadow-[0_0_0_1px_rgba(103,232,249,0.6),0_10px_28px_-6px_rgba(34,211,238,0.7)] active:scale-[0.98] disabled:bg-canvas-elevated disabled:text-ink-faint disabled:shadow-none disabled:border disabled:border-[color:var(--page-border)]",
    ghost: "border hover:border-signal hover:text-signal active:scale-[0.98] disabled:opacity-40",
  };
  const style =
    variant === "ghost" ? { borderColor: "var(--page-border)" } : undefined;

  return (
    <button className={`${base} ${variants[variant]} ${className}`} style={style} {...rest}>
      {children}
    </button>
  );
}
