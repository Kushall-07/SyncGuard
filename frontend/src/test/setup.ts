import "@testing-library/jest-dom/vitest";

// jsdom doesn't implement ResizeObserver, which Recharts' ResponsiveContainer
// needs to measure its parent. A no-op stub is enough for components to mount
// without crashing in tests that don't assert on rendered chart geometry.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = globalThis.ResizeObserver ?? (ResizeObserverStub as unknown as typeof ResizeObserver);
