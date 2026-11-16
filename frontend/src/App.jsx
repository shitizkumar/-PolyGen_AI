import { useEffect, useState } from "react";
import ModelViewer from "./components/ModelViewer";
import "./App.css";

const MAX_OBJ_SIZE = 10 * 1024 * 1024;

function App() {
  const [backendStatus, setBackendStatus] = useState("Checking...");
  const [generationStatus, setGenerationStatus] = useState({
    available: false,
    message: "Checking generation availability...",
  });
  const [prompt, setPrompt] = useState("");
  const [objSource, setObjSource] = useState(null);
  const [fileName, setFileName] = useState("");
  const [fileError, setFileError] = useState("");

  useEffect(() => {
    let active = true;

    async function checkServices() {
      try {
        const healthResponse = await fetch("/api/health");

        if (!healthResponse.ok) {
          throw new Error("Backend unavailable");
        }

        const health = await healthResponse.json();

        if (health.status !== "ok") {
          throw new Error("Backend unhealthy");
        }

        if (active) {
          setBackendStatus("Online");
        }

        const statusResponse = await fetch("/api/generation/status");

        if (!statusResponse.ok) {
          throw new Error("Generation status unavailable");
        }

        const generation = await statusResponse.json();

        if (active) {
          setGenerationStatus({
            available: generation.available === true,
            message: generation.message,
          });
        }
      } catch {
        if (active) {
          setBackendStatus("Offline");
          setGenerationStatus({
            available: false,
            message: "Generation service is unavailable.",
          });
        }
      }
    }

    checkServices();
    const interval = setInterval(checkServices, 10000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  async function handleObjUpload(event) {
    const file = event.target.files?.[0];

    if (!file) return;

    setFileError("");

    if (!file.name.toLowerCase().endsWith(".obj")) {
      setFileError("Please select an OBJ file.");
      return;
    }

    if (file.size > MAX_OBJ_SIZE) {
      setFileError("OBJ files must be smaller than 10 MB.");
      return;
    }

    try {
      const content = await file.text();

      if (!content.trim()) {
        throw new Error("The selected OBJ file is empty.");
      }

      setObjSource(content);
      setFileName(file.name);
    } catch (error) {
      setFileError(error.message);
    }
  }

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
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="A medieval sword with a glowing blue gemstone..."
            rows={5}
            maxLength={500}
          />

          <button
            type="button"
            disabled
            title="AI generation is not configured yet"
          >
            Generate 3D Asset
          </button>

          <p className="generation-message">
            {generationStatus.message}
          </p>

          <div className="upload-section">
            <h3>Preview an OBJ Model</h3>
            <p>Upload a local 3D mesh to inspect its geometry.</p>

            <input
              type="file"
              accept=".obj"
              onChange={handleObjUpload}
            />

            {fileName && <p>Loaded: {fileName}</p>}
            {fileError && <p role="alert">{fileError}</p>}

            <button
              type="button"
              className="secondary-button"
              onClick={() => {
                setObjSource(null);
                setFileName("");
                setFileError("");
              }}
            >
              Show Sample Model
            </button>
          </div>
        </div>

        <div className="viewer-panel">
          <h2>3D Preview</h2>

          <div className="viewer">
            <ModelViewer objSource={objSource} />
          </div>

          <p>Left drag: Rotate | Right drag: Pan | Scroll: Zoom</p>
        </div>
      </section>
    </main>
  );
}

export default App;