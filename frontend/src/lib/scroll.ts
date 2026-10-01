import { useEffect, useRef, useState } from "react";
import type { RefObject } from "react";

function supportsMatchMedia(): boolean {
  return typeof window !== "undefined" && typeof window.matchMedia === "function";
}

export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => supportsMatchMedia() && window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    if (!supportsMatchMedia()) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduced(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  return reduced;
}

// Fires once, true, the first time the element is substantially in view.
// Mirrors the scroll-craft `data-sc-in` contract: enter once, never re-hide.
export function useInView<T extends HTMLElement>(threshold = 0.15): [RefObject<T | null>, boolean] {
  const ref = useRef<T>(null);
  const [inView, setInView] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") {
      setInView(true);
      return;
    }
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setInView(true);
          observer.disconnect();
        }
      },
      { threshold, rootMargin: "0px 0px -12% 0px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [threshold]);

  return [ref, inView];
}

// 0..1 progress of an element through the viewport, for pin-style /
// sync-drift style visualizations. 0 = element's top just entered the
// bottom of the viewport, 1 = element's bottom has left the top.
export function useScrollProgress<T extends HTMLElement>(): [RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [progress, setProgress] = useState(0);
  const reduced = usePrefersReducedMotion();

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    if (reduced) {
      setProgress(1);
      return;
    }

    let raf = 0;
    const update = () => {
      const rect = el.getBoundingClientRect();
      const vh = window.innerHeight || 1;
      const total = rect.height + vh;
      const traveled = vh - rect.top;
      const p = Math.min(1, Math.max(0, traveled / total));
      setProgress(p);
      raf = 0;
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, [reduced]);

  return [ref, progress];
}

// 0..1 progress through a tall container's *own* pin travel, matching
// scroll-craft's `pin` device: the container is `travel = height - viewport`
// tall, an inner stage stays `position: sticky; top: 0`, and progress is how
// far the container has scrolled through that travel. 0 at the moment the
// sticky stage locks, 1 the instant it's about to unlock. Used to drive a
// single scene through multiple states rather than revealing independent
// sections. Reduced motion still advances (callers decide what "progress"
// means without position:sticky; this hook does not fall back to a frozen 1).
export function usePinnedProgress<T extends HTMLElement>(): [RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    let raf = 0;
    const update = () => {
      const rect = el.getBoundingClientRect();
      const vh = window.innerHeight || 1;
      const travel = Math.max(rect.height - vh, 1);
      const traveled = Math.min(Math.max(-rect.top, 0), travel);
      setProgress(traveled / travel);
      raf = 0;
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  return [ref, progress];
}

// The cue contract from scroll-craft's devices.md: "from [to [rampIn [rampOut]]]".
// inStart === inEnd means "greet" (already at full opacity at p = 0).
// outStart === outEnd means "hold" (never fades once reached — for the last state).
export type Cue = [inStart: number, inEnd: number, outStart: number, outEnd: number];

export function cueValue(p: number, [inStart, inEnd, outStart, outEnd]: Cue): number {
  const greet = inStart === inEnd;
  const hold = outStart === outEnd;
  if (!greet) {
    if (p <= inStart) return 0;
    if (p < inEnd) return (p - inStart) / (inEnd - inStart);
  } else if (p >= outStart && hold === false) {
    // past the greet window, fall through to the out-ramp check below
  }
  if (p <= outStart) return 1;
  if (hold) return 1;
  if (p < outEnd) return 1 - (p - outStart) / (outEnd - outStart);
  return 0;
}

// True when scroll has moved past a small threshold from the top of the
// document — used for the nav's transparent -> glass transition.
export function useScrolledPast(thresholdPx = 24): boolean {
  const [past, setPast] = useState(() => typeof window !== "undefined" && window.scrollY > thresholdPx);
  useEffect(() => {
    let raf = 0;
    const update = () => {
      setPast(window.scrollY > thresholdPx);
      raf = 0;
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, [thresholdPx]);
  return past;
}
