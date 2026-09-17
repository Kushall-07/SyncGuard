import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import MainLayout from "./layouts/MainLayout";
import Home from "./pages/Home";

// Home renders eagerly (first paint); every other route is code-split so the
// initial bundle doesn't pay for Recharts-heavy pages (Research, Results, Lab)
// until they're actually visited.
const Analyze = lazy(() => import("./pages/Analyze"));
const Technology = lazy(() => import("./pages/Technology"));
const Research = lazy(() => import("./pages/Research"));
const About = lazy(() => import("./pages/About"));
const Results = lazy(() => import("./pages/Results"));
const SynchronizationLab = lazy(() => import("./pages/SynchronizationLab"));
const Reproducibility = lazy(() => import("./pages/Reproducibility"));

function RouteFallback() {
  return (
    <div className="container-page py-24 flex justify-center">
      <div className="h-8 w-8 rounded-full border-2 border-sage border-t-transparent animate-spin" aria-label="Loading page" />
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<MainLayout />}>
        <Route path="/" element={<Home />} />
        <Route
          path="/analyze"
          element={
            <Suspense fallback={<RouteFallback />}>
              <Analyze />
            </Suspense>
          }
        />
        <Route
          path="/technology"
          element={
            <Suspense fallback={<RouteFallback />}>
              <Technology />
            </Suspense>
          }
        />
        <Route
          path="/research"
          element={
            <Suspense fallback={<RouteFallback />}>
              <Research />
            </Suspense>
          }
        />
        <Route
          path="/about"
          element={
            <Suspense fallback={<RouteFallback />}>
              <About />
            </Suspense>
          }
        />
        <Route
          path="/results"
          element={
            <Suspense fallback={<RouteFallback />}>
              <Results />
            </Suspense>
          }
        />
        <Route
          path="/lab"
          element={
            <Suspense fallback={<RouteFallback />}>
              <SynchronizationLab />
            </Suspense>
          }
        />
        <Route
          path="/reproducibility"
          element={
            <Suspense fallback={<RouteFallback />}>
              <Reproducibility />
            </Suspense>
          }
        />
      </Route>
    </Routes>
  );
}
