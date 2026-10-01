import { Children, cloneElement, isValidElement } from "react";
import type { ReactNode } from "react";
import { useInView } from "../../lib/scroll";

interface RevealProps {
  children: ReactNode;
  className?: string;
  /** Stagger between direct children, in ms. 0 disables staggering. */
  stagger?: number;
  as?: "div" | "section";
}

// The "flow + in" device from scroll-craft's taste rules: opacity 0 -> 1 and a
// 14px rise over ~620ms, triggered once when the block is mostly in view,
// children staggered in reading order. Never re-hides on scroll-back.
export default function Reveal({ children, className = "", stagger = 0, as = "div" }: RevealProps) {
  const [ref, inView] = useInView<HTMLDivElement>();
  const Tag = as;

  if (!stagger) {
    return (
      <Tag
        ref={ref as never}
        className={`${className} transition-[opacity,transform] duration-[620ms] ease-editorial ${
          inView ? "opacity-100 translate-y-0" : "opacity-0 translate-y-[14px]"
        }`}
      >
        {children}
      </Tag>
    );
  }

  const items = Children.toArray(children);
  return (
    <Tag ref={ref as never} className={className}>
      {items.map((child, i) =>
        isValidElement<{ className?: string; style?: React.CSSProperties }>(child) ? (
          cloneElement(child, {
            key: child.key ?? i,
            className: `${child.props.className ?? ""} transition-[opacity,transform] duration-[620ms] ease-editorial ${
              inView ? "opacity-100 translate-y-0" : "opacity-0 translate-y-[14px]"
            }`,
            style: { ...child.props.style, transitionDelay: inView ? `${i * stagger}ms` : "0ms" },
          })
        ) : (
          <span key={i}>{child}</span>
        ),
      )}
    </Tag>
  );
}
