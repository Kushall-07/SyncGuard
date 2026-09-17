import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import SyncTimeline from "./SyncTimeline";

describe("SyncTimeline", () => {
  it("shows a graceful empty state when there are no per-window scores", () => {
    render(<SyncTimeline scores={[]} metadata={null} />);
    expect(screen.getByText(/no per-window synchronization data/i)).toBeInTheDocument();
  });

  it("renders the threshold legend", () => {
    render(<SyncTimeline scores={[0.9, 0.8, 0.7]} metadata={{ fps: 25 }} />);
    expect(screen.getByText(/at\/above threshold \(0\.5\)/i)).toBeInTheDocument();
    expect(screen.getByText(/below threshold/i)).toBeInTheDocument();
  });

  it("surfaces a neutral lower-synchronization summary without claiming manipulation", () => {
    render(<SyncTimeline scores={[0.9, 0.8, 0.2, 0.7]} metadata={{ fps: 25 }} />);
    expect(screen.getByText(/lower synchronization region/i)).toBeInTheDocument();
    // The lowest score is at index 2 -> displayed as "Window 3".
    expect(screen.getByText(/window 3 at/i)).toBeInTheDocument();
    expect(screen.queryByText(/deepfake detected/i)).not.toBeInTheDocument();
  });

  it("does not show a lower-synchronization summary when every window is above threshold", () => {
    render(<SyncTimeline scores={[0.9, 0.8, 0.7]} metadata={{ fps: 25 }} />);
    expect(screen.queryByText(/lower synchronization region/i)).not.toBeInTheDocument();
  });

  it("warns when fps metadata is missing instead of silently mis-scaling time", () => {
    render(<SyncTimeline scores={[0.5, 0.6]} metadata={{}} />);
    expect(screen.getByText(/frame rate metadata was unavailable/i)).toBeInTheDocument();
  });

  it("invites the user to click when a seek handler is provided", () => {
    render(<SyncTimeline scores={[0.5, 0.6]} metadata={{ fps: 25 }} onSeek={() => {}} />);
    expect(screen.getByText(/click a bar to seek/i)).toBeInTheDocument();
  });
});
