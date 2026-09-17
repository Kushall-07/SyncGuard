import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import Analyze from "./Analyze";

function renderAnalyze() {
  return render(
    <MemoryRouter>
      <Analyze />
    </MemoryRouter>,
  );
}

describe("Analyze page", () => {
  it("defaults to the audio-visual tab with the run button disabled until a file is chosen", () => {
    renderAnalyze();
    expect(screen.getByRole("button", { name: /run av analysis/i })).toBeDisabled();
  });

  it("switches to the audio-only tab", async () => {
    renderAnalyze();
    await userEvent.click(screen.getByRole("button", { name: "Audio Only" }));
    expect(screen.getByText(/drop audio file here/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /run audio analysis/i })).toBeDisabled();
  });

  it("shows the empty state before any analysis has run", () => {
    renderAnalyze();
    expect(screen.getByText(/your synchronization analysis will appear here/i)).toBeInTheDocument();
  });
});
