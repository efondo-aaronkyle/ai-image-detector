import { ChangeEvent, DragEvent, useState } from "react";
import "./App.css";

type PredictionResponse = {
  filename: string;
  prediction: "REAL" | "FAKE";
  confidence: number;
  fake_probability: number;
  real_probability: number;
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectFile = (selectedFile: File | null) => {
    if (!selectedFile) {
      return;
    }

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    selectFile(event.target.files?.[0] ?? null);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files?.[0] ?? null);
  };

  const handlePredict = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Prediction request failed.");
      }

      const data: PredictionResponse = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the prediction API."
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <main className="app">
      <div className="container">
        <header className="header">
          <div className="badge">CNN IMAGE CLASSIFIER</div>

          <h1>AI Image Detector</h1>

          <p className="subtitle">
            Analyze an image and estimate whether it is real or
            AI-generated using a trained convolutional neural network.
          </p>
        </header>

        <section className="detector-card">
          <div
            className={`upload-area ${dragging ? "dragging" : ""} ${
              preview ? "has-preview" : ""
            }`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            {preview ? (
              <img
                className="preview"
                src={preview}
                alt="Selected image preview"
              />
            ) : (
              <>
                <div className="upload-icon">↑</div>

                <h2>Drop your image here</h2>

                <p>or choose an image from your computer</p>
              </>
            )}

            <label className="file-button">
              {preview ? "Choose another image" : "Choose image"}
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
              />
            </label>

            {file && <p className="filename">{file.name}</p>}
          </div>

          {error && <div className="error">{error}</div>}

          <div className="actions">
            <button
              className="analyze-button"
              onClick={handlePredict}
              disabled={!file || loading}
            >
              {loading ? "Analyzing image..." : "Analyze Image"}
            </button>

            {file && (
              <button className="reset-button" onClick={reset}>
                Reset
              </button>
            )}
          </div>

          {result && (
            <section
              className={`result-card ${
                result.prediction === "FAKE" ? "fake" : "real"
              }`}
            >
              <div className="result-header">
                <div>
                  <span className="result-label">Prediction</span>
                  <h2>{result.prediction}</h2>
                </div>

                <div className="confidence">
                  <span>Confidence</span>
                  <strong>{(result.confidence * 100).toFixed(2)}%</strong>
                </div>
              </div>

              <div className="probability">
                <div className="probability-row">
                  <span>REAL</span>
                  <strong>
                    {(result.real_probability * 100).toFixed(2)}%
                  </strong>
                </div>

                <div className="bar">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${result.real_probability * 100}%`,
                    }}
                  />
                </div>
              </div>

              <div className="probability">
                <div className="probability-row">
                  <span>AI-GENERATED</span>
                  <strong>
                    {(result.fake_probability * 100).toFixed(2)}%
                  </strong>
                </div>

                <div className="bar">
                  <div
                    className="bar-fill"
                    style={{
                      width: `${result.fake_probability * 100}%`,
                    }}
                  />
                </div>
              </div>
            </section>
          )}
        </section>

        <footer>
          <p>
            Powered by an Improved CNN trained on the CIFAKE dataset.
          </p>
        </footer>
      </div>
    </main>
  );
}

export default App;