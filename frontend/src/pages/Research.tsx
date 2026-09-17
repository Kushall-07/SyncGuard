import type { ReactNode } from "react";

export default function Research() {
  return (
    <div className="container-page py-16">
      <p className="text-xs uppercase tracking-[0.2em] opacity-60">Research</p>
      <h1 className="font-display text-4xl md:text-5xl mt-3">Research &amp; Evaluation</h1>
      <p className="mt-4 max-w-xl opacity-70">
        What SyncGuard was trained and evaluated on, how it was measured, and where those
        measurements do and do not generalize.
      </p>

      <div className="mt-16 space-y-20">
        <Section title="Datasets">
          <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-6">
            <DatasetCard
              name="ASVspoof 2019 LA"
              usedFor="Audio spoof detection"
              description="Logical-access spoofed and bonafide speech used to train and evaluate the audio model."
            />
            <DatasetCard
              name="Celeb-DF v2"
              usedFor="Visual landmark representation development"
              description="Face video used during development of the MediaPipe-based landmark representation. It does not provide audio-visual synchronization ground truth."
            />
            <DatasetCard
              name="LAV-DF"
              usedFor="Audio-visual synchronization evaluation"
              description="Video with controlled temporal audio-visual shifts, used to evaluate the synchronization head."
            />
          </div>
        </Section>

        <Section title="Experiments">
          <p className="max-w-2xl opacity-70 leading-relaxed">
            Evaluation covered two tasks: audio spoof classification on held-out ASVspoof 2019 LA
            utterances, and audio-visual synchronization detection on LAV-DF clips with controlled
            temporal shifts introduced between the audio and visual streams. Both tasks were
            evaluated independently, reflecting the two operating modes described under
            Technology.
          </p>
        </Section>

        <Section title="Metrics">
          <div className="grid sm:grid-cols-2 gap-10">
            <MetricGroup title="Audio" items={["ROC-AUC", "EER", "Accuracy", "Precision", "Recall", "F1"]} />
            <MetricGroup
              title="Audio-Visual"
              items={["Synchronization score", "Per-window synchronization analysis", "Video-level synchronization AUC"]}
            />
          </div>

          <div className="mt-10">
            <p className="text-xs uppercase tracking-widest opacity-60 mb-4">
              Validated audio spoof-detection results
            </p>
            <div className="overflow-x-auto border" style={{ borderColor: "var(--page-border)" }}>
              <table className="w-full text-sm min-w-[420px]">
                <thead>
                  <tr className="border-b" style={{ borderColor: "var(--page-border)" }}>
                    <th className="text-left font-normal opacity-60 uppercase text-xs tracking-widest px-5 py-3">
                      Model
                    </th>
                    <th className="text-left font-normal opacity-60 uppercase text-xs tracking-widest px-5 py-3">
                      AUC
                    </th>
                    <th className="text-left font-normal opacity-60 uppercase text-xs tracking-widest px-5 py-3">
                      EER
                    </th>
                  </tr>
                </thead>
                <tbody>
                  <ResultRow model="CNN" auc="0.988" eer="4.98%" />
                  <ResultRow model="Transformer" auc="0.993" eer="3.74%" />
                  <ResultRow model="CNN + Transformer ensemble" auc="0.998" eer="1.93%" emphasis />
                </tbody>
              </table>
            </div>
          </div>
        </Section>

        <Section title="Visual Results">
          <p className="max-w-2xl text-sm opacity-70 leading-relaxed">
            The visual branch is landmark-based, not image-based: MediaPipe face and mouth landmarks are encoded
            by a transformer and combined with the audio branch through cross-attention. It was retained as an
            auxiliary multimodal representation that feeds the synchronization head, not as a standalone spoof
            or deepfake detector. No standalone visual-only accuracy figure is reported, because the visual
            branch was never evaluated as an independent classifier.
          </p>
        </Section>

        <Section title="AV Results">
          <p className="max-w-2xl text-sm opacity-70 leading-relaxed">
            The sync head was evaluated on LAV-DF with controlled audio-video temporal shifts: a zero-shift
            (naturally aligned) pair is treated as positive, and every non-zero shift is treated as negative.
            Accuracy is therefore a measure of how well the model separates aligned pairs from artificially
            shifted ones — not a measure of real-world deepfake detection accuracy. Natural, unmodified clips do
            not provide independent synchronization ground truth in this setup, since each clip's own audio is
            the only known-positive pairing available.
          </p>
          <div className="mt-8 overflow-x-auto border" style={{ borderColor: "var(--page-border)" }}>
            <table className="w-full text-sm min-w-[480px]">
              <thead>
                <tr className="border-b" style={{ borderColor: "var(--page-border)" }}>
                  <th className="text-left font-normal opacity-60 uppercase text-xs tracking-widest px-5 py-3">
                    Audio Shift
                  </th>
                  <th className="text-left font-normal opacity-60 uppercase text-xs tracking-widest px-5 py-3">
                    Window Accuracy
                  </th>
                  <th className="text-left font-normal opacity-60 uppercase text-xs tracking-widest px-5 py-3">
                    Video-Level Accuracy
                  </th>
                </tr>
              </thead>
              <tbody>
                <DataRow col1="0.0s (aligned)" col2="100.0%" col3="100.0%" />
                <DataRow col1="+0.5s" col2="82.9%" col3="86.7%" />
                <DataRow col1="+1.0s" col2="94.0%" col3="96.0%" />
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs opacity-50 max-w-2xl">
            Phase 11 controlled shift-sensitivity evaluation, LAV-DF dev split, 1,000 clips. Negative shifts are
            not evaluable under this alignment scheme (zero valid overlapping windows) and are excluded from the
            Synchronization Lab for the same reason.
          </p>
        </Section>

        <Section title="Contrastive Ablation">
          <p className="max-w-2xl text-sm opacity-70 leading-relaxed">
            Phase 12 adds an optional InfoNCE contrastive objective on top of the Phase 11 sync-only baseline,
            weighted by λ. Both configurations below were trained for 120 epochs on the same controlled-shift
            training setup and monitored on video-level synchronization AUC.
          </p>
          <div className="mt-8 overflow-x-auto border" style={{ borderColor: "var(--page-border)" }}>
            <table className="w-full text-sm min-w-[420px]">
              <thead>
                <tr className="border-b" style={{ borderColor: "var(--page-border)" }}>
                  <th className="text-left font-normal opacity-60 uppercase text-xs tracking-widest px-5 py-3">
                    Configuration
                  </th>
                  <th className="text-left font-normal opacity-60 uppercase text-xs tracking-widest px-5 py-3">
                    Best Val. Video Sync AUC
                  </th>
                  <th className="text-left font-normal opacity-60 uppercase text-xs tracking-widest px-5 py-3">
                    Best Epoch
                  </th>
                </tr>
              </thead>
              <tbody>
                <DataRow col1="λ = 0 (baseline)" col2="1.0000" col3="7 / 120" />
                <DataRow col1="λ = 0.1 (contrastive)" col2="1.0000" col3="7 / 120" />
              </tbody>
            </table>
          </div>
          <p className="mt-4 max-w-2xl text-sm opacity-80 leading-relaxed border-l-2 pl-4" style={{ borderColor: "var(--copper)" }}>
            Both settings saturate at the same ceiling video-level AUC on the controlled-shift validation split.
            This experiment does not establish a measurable improvement on the controlled-shift validation
            setup — the comparison is inconclusive at this ceiling, not a demonstrated benefit of contrastive
            learning.
          </p>
        </Section>

        <Section title="Limitations">
          <div className="border-l-2 border-copper pl-6 py-1 max-w-2xl">
            <ul className="space-y-4 text-sm leading-relaxed opacity-80">
              <li>AV synchronization analysis is not equivalent to universal deepfake detection.</li>
              <li>Manipulated content can remain synchronized.</li>
              <li>
                LAV-DF controlled shift evaluation does not establish natural real-world
                synchronization ground truth.
              </li>
              <li>Raw model scores are not calibrated probabilities.</li>
              <li>
                Visual landmark representation is an auxiliary multimodal representation rather
                than a standalone deepfake detector.
              </li>
              <li>Performance depends on input quality and preprocessing.</li>
              <li>No claim of frame-level manipulation localization is made.</li>
            </ul>
          </div>
        </Section>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <h2 className="font-display text-2xl md:text-3xl">{title}</h2>
      <div className="mt-6">{children}</div>
    </section>
  );
}

function DatasetCard({ name, usedFor, description }: { name: string; usedFor: string; description: string }) {
  return (
    <div className="border p-6" style={{ borderColor: "var(--page-border)" }}>
      <h3 className="font-display text-lg">{name}</h3>
      <p className="mt-3 text-xs uppercase tracking-widest opacity-50">Used for</p>
      <p className="mt-1 text-sm opacity-90">{usedFor}</p>
      <p className="mt-4 text-sm opacity-60 leading-relaxed">{description}</p>
    </div>
  );
}

function MetricGroup({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-widest opacity-60">{title}</p>
      <ul className="mt-3 space-y-2 text-sm opacity-80">
        {items.map((item) => (
          <li key={item} className="border-b pb-2" style={{ borderColor: "var(--page-border)" }}>
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ResultRow({ model, auc, eer, emphasis = false }: { model: string; auc: string; eer: string; emphasis?: boolean }) {
  return (
    <tr className="border-b last:border-b-0" style={{ borderColor: "var(--page-border)" }}>
      <td className={`px-5 py-3 ${emphasis ? "font-medium" : ""}`}>{model}</td>
      <td className={`px-5 py-3 font-display ${emphasis ? "text-sage-dark" : ""}`}>≈ {auc}</td>
      <td className={`px-5 py-3 font-display ${emphasis ? "text-sage-dark" : ""}`}>≈ {eer}</td>
    </tr>
  );
}

function DataRow({ col1, col2, col3 }: { col1: string; col2: string; col3: string }) {
  return (
    <tr className="border-b last:border-b-0" style={{ borderColor: "var(--page-border)" }}>
      <td className="px-5 py-3">{col1}</td>
      <td className="px-5 py-3 font-display">{col2}</td>
      <td className="px-5 py-3 font-display">{col3}</td>
    </tr>
  );
}
