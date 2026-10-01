import { useEffect, useRef, useState } from "react";
import type { CSSProperties, RefObject } from "react";
import { usePrefersReducedMotion } from "./scroll";

function supportsFinePointer(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(hover: hover) and (pointer: fine)").matches
  );
}

export interface PointerPos {
  /** px relative to the element's own box */
  x: number;
  y: number;
  /** -1..1, centered, for tilt/parallax-style math */
  nx: number;
  ny: number;
}

// Pointer devices (scroll-craft devices.md §9): tracked only on
// (hover: hover) and (pointer: fine) so touch never fires a false hover, and
// disabled entirely under reduced motion. null means "not currently tracked".
export function usePointerInElement<T extends HTMLElement>(): [RefObject<T | null>, PointerPos | null] {
  const ref = useRef<T>(null);
  const [pos, setPos] = useState<PointerPos | null>(null);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    const el = ref.current;
    if (!el || reduced || !supportsFinePointer()) return;

    const onMove = (e: PointerEvent) => {
      const rect = el.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      setPos({
        x,
        y,
        nx: rect.width ? (x / rect.width) * 2 - 1 : 0,
        ny: rect.height ? (y / rect.height) * 2 - 1 : 0,
      });
    };
    const onLeave = () => setPos(null);

    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerleave", onLeave);
    return () => {
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerleave", onLeave);
    };
  }, [reduced]);

  return [ref, pos];
}

// `tilt` device: 3D rotation toward the pointer, capped well under the
// "starts reading as a toy" line (devices.md §9: 5-9deg, past 12 is a toy).
export function useTilt<T extends HTMLElement>(maxDeg = 6): [RefObject<T | null>, CSSProperties] {
  const [ref, pos] = usePointerInElement<T>();
  const style: CSSProperties = pos
    ? {
        transform: `perspective(900px) rotateX(${(-pos.ny * maxDeg).toFixed(2)}deg) rotateY(${(pos.nx * maxDeg).toFixed(2)}deg)`,
        transition: "transform 120ms ease-out",
      }
    : { transform: "perspective(900px) rotateX(0deg) rotateY(0deg)", transition: "transform 300ms ease-out" };
  return [ref, style];
}

// `magnet` device: the element drifts toward the pointer within its own
// hit area. Primary CTA only (devices.md §9: "a page of magnetic elements
// is unusable") — strength 0.2-0.35.
export function useMagnetic<T extends HTMLElement>(strength = 0.3): [RefObject<T | null>, CSSProperties] {
  const [ref, pos] = usePointerInElement<T>();
  const style: CSSProperties = pos
    ? {
        transform: `translate(${(pos.nx * 10 * strength).toFixed(1)}px, ${(pos.ny * 10 * strength).toFixed(1)}px)`,
        transition: "transform 120ms ease-out",
      }
    : { transform: "translate(0,0)", transition: "transform 250ms ease-out" };
  return [ref, style];
}
