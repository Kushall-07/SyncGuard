export default function About() {
  return (
    <div className="container-page py-16">
      <p className="text-xs uppercase tracking-[0.2em] opacity-60">About</p>
      <h1 className="font-display text-4xl md:text-5xl mt-3 max-w-2xl leading-tight">
        Building Trust in Digital Media
      </h1>
      <p className="mt-4 max-w-xl opacity-70">
        SyncGuard is a research project exploring how audio and visual analysis can be combined
        to examine synthetic speech and audio-visual synchronization.
      </p>

      <div className="mt-16 grid sm:grid-cols-3 gap-10">
        <AboutPoint
          title="Research Driven"
          body="SyncGuard grew out of coursework and independent research into speech spoof detection and multimodal audio-visual analysis. It is a research artifact, not a commercial product."
        />
        <AboutPoint
          title="Open & Transparent"
          body="The project documents what its models were trained and evaluated on, what the results were, and where the current approach falls short, rather than presenting only favorable numbers."
        />
        <AboutPoint
          title="Multimodal Analysis"
          body="Audio and visual signals are analyzed together, using landmark-based visual representations and cross-attention, to examine whether they remain temporally consistent."
        />
      </div>

      <div className="mt-20 border-t border-b py-14" style={{ borderColor: "var(--page-border)" }}>
        <p className="font-display italic text-2xl md:text-3xl leading-relaxed max-w-2xl mx-auto text-center opacity-90">
          AI reasons.
          <br />
          Evidence supports the result.
          <br />
          The user makes the final decision.
        </p>
      </div>

      <div className="mt-16 flex flex-wrap items-center gap-x-10 gap-y-4 text-sm">
        <a
          href="https://github.com/Kushall-07/SyncGuard"
          target="_blank"
          rel="noopener noreferrer"
          className="underline opacity-80 hover:opacity-100 transition-opacity duration-200"
        >
          GitHub
        </a>
        <span className="opacity-40 cursor-not-allowed" aria-disabled="true">
          Documentation <span className="text-xs">(coming soon)</span>
        </span>
        <span className="opacity-40 cursor-not-allowed" aria-disabled="true">
          Research / Paper <span className="text-xs">(coming soon)</span>
        </span>
      </div>
    </div>
  );
}

function AboutPoint({ title, body }: { title: string; body: string }) {
  return (
    <div className="border-t pt-6" style={{ borderColor: "var(--page-border)" }}>
      <h2 className="font-display text-xl">{title}</h2>
      <p className="mt-3 text-sm opacity-70 leading-relaxed">{body}</p>
    </div>
  );
}
