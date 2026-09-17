import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

// Nav calls /api/health and the Lab page calls /api/lab/samples on mount; stub fetch
// so route smoke tests don't depend on a running backend.
beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      if (url.toString().includes("/api/health")) {
        return new Response(JSON.stringify({ status: "ready", gpu_ready: false, device: "cpu" }), { status: 200 });
      }
      if (url.toString().includes("/api/lab/samples")) {
        return new Response(JSON.stringify({ samples: [], available_shifts: [] }), { status: 200 });
      }
      return new Response("{}", { status: 200 });
    }),
  );
});

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

describe("routing", () => {
  it("renders the home page hero", async () => {
    renderAt("/");
    expect(await screen.findByRole("heading", { level: 1 })).toHaveTextContent(/what you hear/i);
  });

  it("renders the analyze page", async () => {
    renderAt("/analyze");
    expect(await screen.findByRole("heading", { name: /upload and analyze/i })).toBeInTheDocument();
  });

  it("renders the technology page", async () => {
    renderAt("/technology");
    expect(await screen.findByRole("heading", { name: /multimodal approach to media verification/i })).toBeInTheDocument();
  });

  it("renders the research page with the audio metrics table", async () => {
    renderAt("/research");
    expect(await screen.findByRole("heading", { name: /research & evaluation/i })).toBeInTheDocument();
    expect(await screen.findByText(/CNN \+ Transformer ensemble/i)).toBeInTheDocument();
  });

  it("renders the about page", async () => {
    renderAt("/about");
    expect(await screen.findByRole("heading", { name: /building trust in digital media/i })).toBeInTheDocument();
  });

  it("renders results as an empty state when no analysis state is present", async () => {
    renderAt("/results");
    expect(await screen.findByRole("heading", { name: /no report loaded/i })).toBeInTheDocument();
  });

  it("renders the synchronization lab with its scientific disclaimer", async () => {
    renderAt("/lab");
    expect(await screen.findByRole("heading", { name: /synchronization lab/i })).toBeInTheDocument();
    expect(await screen.findByText(/not be interpreted as real-world deepfake detection accuracy/i)).toBeInTheDocument();
  });

  it("renders the reproducibility page", async () => {
    renderAt("/reproducibility");
    expect(await screen.findByRole("heading", { name: /system & reproducibility/i })).toBeInTheDocument();
  });
});
