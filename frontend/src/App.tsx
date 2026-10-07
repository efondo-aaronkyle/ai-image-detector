import { useState } from "react";

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);
    setConfidence(null);

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

      const data = await response.json();

      setPrediction(data.prediction);
      setConfidence(data.confidence);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "An unexpected error occurred."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main>
      <h1>AI Image Detector</h1>

      <p>
        Upload an image to determine whether it is REAL or AI-GENERATED.
      </p>

      <input
        type="file"
        accept="image/*"
        onChange={(event) => {
          setFile(event.target.files?.[0] ?? null);
          setPrediction(null);
          setConfidence(null);
          setError(null);
        }}
      />

      {file && <p>Selected: {file.name}</p>}

      <button onClick={handlePredict} disabled={!file || loading}>
        {loading ? "Analyzing..." : "Analyze Image"}
      </button>

      {error && <p>{error}</p>}

      {prediction && (
        <section>
          <h2>Prediction: {prediction}</h2>
          <p>
            Confidence: {(confidence! * 100).toFixed(2)}%
          </p>
        </section>
      )}
    </main>
  );
}

export default App;