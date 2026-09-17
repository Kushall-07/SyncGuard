import { useEffect } from "react";
import { Outlet, useLocation } from "react-router-dom";
import Nav from "../components/Nav";
import Footer from "../components/Footer";

const DARK_ROUTES = new Set(["/", "/about", "/results"]);

export default function MainLayout() {
  const location = useLocation();
  const isDark = DARK_ROUTES.has(location.pathname);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
  }, [location.pathname]);

  return (
    <div
      data-theme={isDark ? "dark" : "light"}
      className="min-h-screen flex flex-col font-body"
      style={{ backgroundColor: "var(--page-bg)", color: "var(--page-fg)" }}
    >
      <Nav />
      <main key={location.pathname} className="flex-1 fade-in">
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}
