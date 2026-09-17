import type { ButtonHTMLAttributes, ReactNode } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "ghost";
  children: ReactNode;
}

export default function Button({ variant = "primary", className = "", children, ...rest }: ButtonProps) {
  const base =
    "inline-flex items-center gap-2 px-6 py-3 text-sm tracking-wide transition-all duration-300 ease-editorial disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sage";
  const variants: Record<string, string> = {
    primary: "bg-sage text-ink hover:bg-sage-light active:scale-[0.98]",
    ghost:
      "border hover:border-sage hover:text-sage-light active:scale-[0.98]",
  };
  const style =
    variant === "ghost" ? { borderColor: "var(--page-border)" } : undefined;

  return (
    <button className={`${base} ${variants[variant]} ${className}`} style={style} {...rest}>
      {children}
    </button>
  );
}
