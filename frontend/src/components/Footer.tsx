import { Link } from "react-router-dom";

export default function Footer() {
  return (
    <footer className="border-t mt-auto" style={{ borderColor: "var(--page-border)" }}>
      <div className="container-page py-10 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 text-sm opacity-70">
        <span className="font-display text-base">SyncGuard</span>
        <p className="max-w-xl">
          Research demonstration for multimodal media verification. Results should not be
          used for security-critical decisions without further validation.
        </p>
        <div className="flex items-center gap-6">
          <Link to="/reproducibility" className="hover:opacity-100 transition-opacity">
            Reproducibility
          </Link>
          <a
            href="https://github.com/Kushall-07/SyncGuard"
            target="_blank"
            rel="noreferrer"
            className="hover:opacity-100 transition-opacity"
          >
            GitHub ↗
          </a>
        </div>
      </div>
    </footer>
  );
}
