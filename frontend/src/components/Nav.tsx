import { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { checkHealth } from "../services/api";

const LINKS = [
  { to: "/", label: "Home" },
  { to: "/analyze", label: "Analyze" },
  { to: "/lab", label: "Lab" },
  { to: "/technology", label: "Technology" },
  { to: "/research", label: "Research" },
  { to: "/about", label: "About" },
];

export default function Nav() {
  const [gpuReady, setGpuReady] = useState<boolean | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    let cancelled = false;
    checkHealth()
      .then((h) => {
        if (!cancelled) setGpuReady(h.gpu_ready);
      })
      .catch(() => {
        if (!cancelled) setGpuReady(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setMenuOpen(false);
  }, [location.pathname]);

  return (
    <header
      className="sticky top-0 z-50 border-b backdrop-blur-md"
      style={{
        borderColor: "var(--page-border)",
        backgroundColor: "color-mix(in srgb, var(--page-bg) 82%, transparent)",
      }}
    >
      <div className="container-page flex h-16 items-center justify-between">
        <NavLink to="/" className="font-display text-lg tracking-tight">
          SyncGuard
        </NavLink>

        <nav className="hidden md:flex items-center gap-8 text-sm">
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === "/"}
              className={({ isActive }) =>
                `relative py-2 transition-colors duration-200 ${
                  isActive ? "text-sage-light" : "opacity-75 hover:opacity-100"
                }`
              }
            >
              {({ isActive }) => (
                <>
                  {link.label}
                  <span
                    className={`absolute -bottom-[1px] left-0 h-[1.5px] w-full bg-sage transition-transform duration-300 origin-left ${
                      isActive ? "scale-x-100" : "scale-x-0"
                    }`}
                  />
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center gap-4 sm:gap-5 text-sm">
          <span className="hidden sm:flex items-center gap-2 opacity-80">
            <span className={`h-1.5 w-1.5 rounded-full ${gpuReady ? "bg-sage" : "bg-copper"}`} />
            {gpuReady === null ? "Checking…" : gpuReady ? "GPU Ready" : "CPU Mode"}
          </span>
          <a
            href="https://github.com/Kushall-07/SyncGuard"
            target="_blank"
            rel="noreferrer"
            className="hidden sm:inline opacity-80 hover:opacity-100 transition-opacity"
          >
            GitHub
          </a>
          <button
            onClick={() => setMenuOpen((v) => !v)}
            aria-label="Toggle menu"
            aria-expanded={menuOpen}
            className="md:hidden flex flex-col justify-center gap-1.5 h-8 w-8 -mr-1"
          >
            <span
              className="block h-px w-5 bg-current transition-transform duration-300"
              style={{ transform: menuOpen ? "translateY(4.5px) rotate(45deg)" : "none" }}
            />
            <span
              className="block h-px w-5 bg-current transition-transform duration-300"
              style={{ transform: menuOpen ? "translateY(-4.5px) rotate(-45deg)" : "none" }}
            />
          </button>
        </div>
      </div>

      <div
        className={`md:hidden overflow-hidden border-t transition-[max-height,opacity] duration-300 ease-editorial ${
          menuOpen ? "max-h-72 opacity-100" : "max-h-0 opacity-0"
        }`}
        style={{ borderColor: "var(--page-border)" }}
      >
        <nav className="container-page flex flex-col py-4 text-sm">
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === "/"}
              className={({ isActive }) =>
                `py-3 border-b last:border-b-0 transition-colors duration-200 ${
                  isActive ? "text-sage-light" : "opacity-75"
                }`
              }
              style={{ borderColor: "var(--page-border)" }}
            >
              {link.label}
            </NavLink>
          ))}
          <a
            href="https://github.com/Kushall-07/SyncGuard"
            target="_blank"
            rel="noreferrer"
            className="py-3 opacity-75"
          >
            GitHub ↗
          </a>
        </nav>
      </div>
    </header>
  );
}
