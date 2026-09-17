import type { ReactNode } from "react";

export default function Reproducibility() {
  return (
    <div className="container-page py-16">
      <p className="text-xs uppercase tracking-[0.2em] opacity-60">Reproducibility</p>
      <h1 className="font-display text-4xl md:text-5xl mt-3 max-w-2xl leading-tight">System &amp; Reproducibility</h1>
      <p className="mt-4 max-w-2xl opacity-70 leading-relaxed">
        What SyncGuard's models were built from and what the running system consists of, so results can be
        traced back to their source.
      </p>

      <div className="mt-16 space-y-16">
        <Section title="Datasets">
          <Grid>
            <Item label="ASVspoof 2019 LA" value="Audio spoof detection training and evaluation" />
            <Item label="Celeb-DF v2" value="Visual landmark representation development" />
            <Item label="LAV-DF" value="Audio-visual synchronization evaluation with controlled temporal shifts" />
          </Grid>
        </Section>

        <Section title="Models">
          <Grid>
            <Item label="Audio CNN" value="Convolutional spoof-detection baseline" />
            <Item label="Audio Transformer" value="Transformer spoof-detection model" />
            <Item label="Audio Ensemble" value="Mean of CNN + Transformer probabilities" />
            <Item label="Visual Landmark Transformer" value="Encodes MediaPipe face/mouth landmark sequences" />
            <Item label="Sync Head" value="Per-window synchronization classifier over fused representations" />
            <Item label="Contrastive Head" value="Phase 12 InfoNCE adapters + projection heads (λ configurable)" />
          </Grid>
        </Section>

        <Section title="Preprocessing">
          <Grid>
            <Item label="Audio" value="16 kHz mono, peak-normalized, log-Mel spectrogram" />
            <Item label="Landmarks" value="MediaPipe FaceLandmarker, 478-point face mesh" />
            <Item label="Landmark region" value="Face + mouth subset (142 of 478 points)" />
            <Item label="Landmark normalization" value="Interocular scale normalization with rotation alignment" />
            <Item label="Temporal alignment" value="Deterministic time-aware audio-to-video token correspondence" />
          </Grid>
        </Section>

        <Section title="Training">
          <p className="max-w-2xl text-sm opacity-70 leading-relaxed mb-6">
            Audio and visual encoders are trained and frozen independently, then reused unmodified for
            audio-visual synchronization training. Only the cross-attention, sync head, and (when enabled)
            Phase 12 contrastive adapters are trained in the synchronization stage.
          </p>
          <Grid>
            <Item label="Encoder freezing" value="Audio and visual encoders frozen during sync training" />
            <Item label="Sync objective" value="Masked binary cross-entropy on shifted vs. aligned pairs" />
            <Item label="Contrastive objective" value="Optional symmetric InfoNCE, λ ∈ {0, 0.1} evaluated" />
            <Item label="Negative pairs" value="Deterministic controlled temporal shifts (±0.5s, ±1.0s, ±2.0s)" />
          </Grid>
        </Section>

        <Section title="Evaluation">
          <Grid>
            <Item label="Audio metrics" value="ROC-AUC, EER on held-out ASVspoof 2019 LA" />
            <Item label="AV metrics" value="Window-level and video-level synchronization AUC" />
            <Item label="AV setup" value="LAV-DF controlled-shift evaluation (shift = 0 → positive, shift ≠ 0 → negative)" />
          </Grid>
        </Section>

        <Section title="System">
          <Grid>
            <Item label="Language" value="Python 3.11" />
            <Item label="Deep learning" value="PyTorch 2.13 (CUDA 13.0 build)" />
            <Item label="Landmark extraction" value="MediaPipe 0.10" />
            <Item label="Numerical" value="NumPy 2.4" />
            <Item label="Backend API" value="FastAPI 0.141 + Uvicorn" />
            <Item label="Frontend" value="React 19 + TypeScript" />
            <Item label="Build tool" value="Vite" />
            <Item label="Styling" value="Tailwind CSS 3.4" />
            <Item label="Charting" value="Recharts 3.10" />
            <Item label="Acceleration" value="CUDA GPU when available, CPU fallback" />
          </Grid>
        </Section>

        <Section title="Source">
          <a
            href="https://github.com/Kushall-07/SyncGuard"
            target="_blank"
            rel="noopener noreferrer"
            className="underline opacity-80 hover:opacity-100 transition-opacity duration-200 text-sm"
          >
            GitHub repository ↗
          </a>
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

function Grid({ children }: { children: ReactNode }) {
  return <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-6">{children}</div>;
}

function Item({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-l-2 pl-4 py-0.5" style={{ borderColor: "var(--page-border)" }}>
      <p className="text-xs uppercase tracking-widest opacity-50">{label}</p>
      <p className="mt-1 text-sm opacity-90 leading-relaxed">{value}</p>
    </div>
  );
}
