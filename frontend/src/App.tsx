import { useEffect, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";

const API_URL = import.meta.env.VITE_API_URL;

type PredictionResponse = {
  filename: string;
  prediction: "REAL" | "FAKE";
  confidence: number;
  fake_probability: number;
  real_probability: number;
};

type ModelInfo = {
  model: string;
  dataset: string;
  input_size: string;
  classes: string[];
  test_accuracy: number;
  test_f1: number;
  test_roc_auc: number;
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const response = await fetch(`${API_URL}/api/model-info`);

        if (!response.ok) {
          throw new Error("Failed to load model information.");
        }

        const data: ModelInfo = await response.json();
        setModelInfo(data);
      } catch {
        setModelInfo(null);
      }
    };

    fetchModelInfo();
  }, []);

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
      const response = await fetch(`${API_URL}/api/predict`, {
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
    <main className="min-h-screen bg-[#0b1020] px-[14px] py-10 text-slate-50 sm:px-5 sm:py-16">
      <div className="mx-auto w-full max-w-[860px]">
        {/* Header */}
        <header className="mb-9 text-center">
          <div className="inline-block rounded-full border border-slate-700 bg-slate-800/70 px-3 py-1.5 text-xs font-bold tracking-[0.12em] text-blue-300">
            CNN IMAGE CLASSIFIER
          </div>

          <h1 className="my-[18px] text-4xl font-bold leading-none tracking-[-0.04em] sm:text-5xl md:text-6xl">
            AI Image Detector
          </h1>

          <p className="mx-auto max-w-[650px] text-[17px] leading-[1.7] text-slate-400">
            Analyze an image and estimate whether it is real or AI-generated
            using a trained convolutional neural network.
          </p>
        </header>

        {/* Detector Card */}
        <section className="rounded-3xl border border-slate-800 bg-slate-900/95 p-3.5 shadow-[0_25px_70px_rgba(0,0,0,0.35)] sm:p-6">
          {/* Upload Area */}
          <div
            className={`flex min-h-[320px] flex-col items-center justify-center rounded-[18px] border-2 border-dashed p-5 text-center transition duration-200 sm:min-h-[360px] sm:p-8 ${
              dragging
                ? "border-blue-400 bg-blue-500/10"
                : "border-slate-700 hover:border-blue-400 hover:bg-blue-500/5"
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
                className="mb-5 max-h-[300px] max-w-full rounded-[14px] object-contain shadow-[0_15px_35px_rgba(0,0,0,0.35)]"
                src={preview}
                alt="Selected image preview"
              />
            ) : (
              <>
                <div className="mb-5 grid h-16 w-16 place-items-center rounded-[18px] bg-blue-950 text-[32px] font-bold text-blue-400">
                  ↑
                </div>

                <h2 className="mb-2 text-2xl font-semibold">
                  Drop your image here
                </h2>

                <p className="mb-5 text-slate-500">
                  or choose an image from your computer
                </p>
              </>
            )}

            <label className="inline-flex cursor-pointer items-center justify-center rounded-[10px] bg-blue-600 px-[18px] py-[11px] font-bold text-white transition hover:bg-blue-700">
              {preview ? "Choose another image" : "Choose image"}

              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>

            {file && (
              <p className="mt-3 max-w-full overflow-hidden text-ellipsis whitespace-nowrap text-slate-400">
                {file.name}
              </p>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="mt-4 rounded-[10px] border border-red-900 bg-red-950/20 px-3.5 py-3 text-red-300">
              {error}
            </div>
          )}

          {/* Actions */}
          <div className="mt-5 flex flex-col gap-3 sm:flex-row">
            <button
              className="flex-1 cursor-pointer rounded-[10px] border-0 bg-blue-600 px-[18px] py-[13px] font-bold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
              onClick={handlePredict}
              disabled={!file || loading}
            >
              {loading ? "Analyzing image..." : "Analyze Image"}
            </button>

            {file && (
              <button
                className="cursor-pointer rounded-[10px] border-0 bg-slate-800 px-[18px] py-[13px] font-bold text-slate-300 transition hover:bg-slate-700"
                onClick={reset}
              >
                Reset
              </button>
            )}
          </div>

          {/* Result */}
          {result && (
            <section
              className={`mt-6 rounded-[18px] border bg-gray-900 p-6 ${
                result.prediction === "FAKE"
                  ? "border-red-800"
                  : "border-green-800"
              }`}
            >
              <div className="mb-[26px] flex flex-col items-start justify-between gap-5 sm:flex-row">
                <div>
                  <span className="text-[13px] uppercase tracking-[0.08em] text-slate-500">
                    Prediction
                  </span>

                  <h2
                    className={`mt-1.5 text-4xl font-bold ${
                      result.prediction === "FAKE"
                        ? "text-red-400"
                        : "text-green-400"
                    }`}
                  >
                    {result.prediction}
                  </h2>
                </div>

                <div className="text-left sm:text-right">
                  <span className="text-[13px] uppercase tracking-[0.08em] text-slate-500">
                    Confidence
                  </span>

                  <strong className="mt-1 block text-2xl">
                    {(result.confidence * 100).toFixed(2)}%
                  </strong>
                </div>
              </div>

              {/* REAL probability */}
              <div className="mt-[18px]">
                <div className="mb-2 flex justify-between text-[13px] font-bold text-slate-300">
                  <span>REAL</span>

                  <strong>
                    {(result.real_probability * 100).toFixed(2)}%
                  </strong>
                </div>

                <div className="h-[9px] overflow-hidden rounded-full bg-slate-800">
                  <div
                    className="h-full rounded-full bg-blue-500 transition-all duration-500"
                    style={{
                      width: `${result.real_probability * 100}%`,
                    }}
                  />
                </div>
              </div>

              {/* AI-generated probability */}
              <div className="mt-[18px]">
                <div className="mb-2 flex justify-between text-[13px] font-bold text-slate-300">
                  <span>AI-GENERATED</span>

                  <strong>
                    {(result.fake_probability * 100).toFixed(2)}%
                  </strong>
                </div>

                <div className="h-[9px] overflow-hidden rounded-full bg-slate-800">
                  <div
                    className="h-full rounded-full bg-blue-500 transition-all duration-500"
                    style={{
                      width: `${result.fake_probability * 100}%`,
                    }}
                  />
                </div>
              </div>
            </section>
          )}
        </section>

        {/* Model Information */}
        {modelInfo && (
          <section className="mt-5 rounded-[18px] border border-slate-800 bg-slate-900/95 p-6">
            <div className="mb-[22px] flex items-start justify-between gap-5">
              <div>
                <span className="text-xs uppercase tracking-[0.08em] text-slate-500">
                  Detection Model
                </span>

                <h2 className="mt-1.5 text-[22px] font-semibold">
                  {modelInfo.model}
                </h2>
              </div>

              <span className="rounded-full border border-green-800 bg-green-950/30 px-2.5 py-1.5 text-[11px] font-bold tracking-[0.08em] text-green-400">
                ONLINE
              </span>
            </div>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-xl border border-slate-800 bg-gray-900 p-[15px]">
                <span className="mb-1.5 block text-xs text-slate-500">
                  Dataset
                </span>

                <strong className="text-[15px] text-slate-200">
                  {modelInfo.dataset}
                </strong>
              </div>

              <div className="rounded-xl border border-slate-800 bg-gray-900 p-[15px]">
                <span className="mb-1.5 block text-xs text-slate-500">
                  Input
                </span>

                <strong className="text-[15px] text-slate-200">
                  {modelInfo.input_size}
                </strong>
              </div>

              <div className="rounded-xl border border-slate-800 bg-gray-900 p-[15px]">
                <span className="mb-1.5 block text-xs text-slate-500">
                  Test Accuracy
                </span>

                <strong className="text-[15px] text-slate-200">
                  {(modelInfo.test_accuracy * 100).toFixed(2)}%
                </strong>
              </div>

              <div className="rounded-xl border border-slate-800 bg-gray-900 p-[15px]">
                <span className="mb-1.5 block text-xs text-slate-500">
                  ROC-AUC
                </span>

                <strong className="text-[15px] text-slate-200">
                  {(modelInfo.test_roc_auc * 100).toFixed(2)}%
                </strong>
              </div>
            </div>
          </section>
        )}

        {/* Footer */}
        <footer className="mt-7 text-center text-[13px] text-slate-600">
          <p>
            Powered by an Improved CNN trained on the CIFAKE dataset.
          </p>
        </footer>
      </div>
    </main>
  );
}

export default App;