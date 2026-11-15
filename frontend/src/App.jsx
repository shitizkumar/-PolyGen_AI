import { useEffect, useState } from "react";
import ModelViewer from "./components/ModelViewer";
import "./App.css";

function App() {
  const [backendStatus, setBackendStatus] = useState("Checking...");

  useEffect(() => {
    let active = true;

    async function checkBackend() {
      try {
        const response = await fetch("/api/health");

        if (!response.ok) {
          throw new Error("Backend unavailable");
        }

        const data = await response.json();

        if (active) {
          setBackendStatus(
            data.status === "ok" ? "Online" : "Offline"
          );
        }
      } catch {
        if (active) {
          setBackendStatus("Offline");
        }
      }
    }

    checkBackend();

    const interval = setInterval(checkBackend, 10000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <main className="dashboard">
      <header>
        <h1>PolyGen AI</h1>
        <p>Generative Text-to-3D Asset Pipeline</p>

        <div className="backend-status">
          <span
            className={
              backendStatus === "Online"
                ? "status-dot online"
                : "status-dot offline"
            }
          />
          Backend: {backendStatus}
        </div>
      </header>

      <section className="workspace">
        <div className="prompt-panel">
          <h2>Text-to-3D Generator</h2>
          <p>Describe the 3D asset you want to create.</p>

          <textarea
            placeholder="A medieval sword with a glowing blue gemstone..."
            rows={5}
          />

          <button disabled>
            Generate 3D Asset (Coming Soon)
          </button>
        </div>

        <div className="viewer-panel">
          <h2>3D Preview</h2>

          <div className="viewer">
            <ModelViewer />
          </div>

          <p>Drag to rotate | Scroll to zoom</p>
        </div>
      </section>
    </main>
  );
}

export default App;
