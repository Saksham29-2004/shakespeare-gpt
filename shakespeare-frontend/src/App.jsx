
import { useState } from "react";
import "./App.css";
const API_URL = import.meta.env.VITE_API_URL;

function App() {
  const [prompt, setPrompt] = useState("");
  const [generatedText, setGeneratedText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // New generation settings
  const [maxNewTokens, setMaxNewTokens] = useState(100);
  const [temperature, setTemperature] = useState(0.8);

  async function handleGenerate() {
    if (!prompt.trim()) {
      setError("Please enter a prompt.");
      return;
    }

    setLoading(true);
    setError("");
    setGeneratedText("");

    try {
      const response = await fetch(
        `${API_URL}/generate`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            prompt,
            max_new_tokens: maxNewTokens,
            temperature,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }

      const data = await response.json();
      setGeneratedText(data.generated_text);
    } catch (err) {
      setError(
        "Generation failed. Check that the backend is running and try again."
      );
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app">
      <h1>Shakespeare GPT</h1>
      <p>Generate text in the style of Shakespeare.</p>

      <textarea
        value={prompt}
        onChange={(event) => setPrompt(event.target.value)}
        placeholder="Enter a prompt, e.g. Romeo: O, wherefore art thou..."
        rows={5}
      />

      <section className="controls">
        <label htmlFor="max-tokens">
          Maximum new tokens: {maxNewTokens}
        </label>
        <input
          id="max-tokens"
          type="range"
          min="1"
          max="200"
          step="1"
          value={maxNewTokens}
          onChange={(event) =>
            setMaxNewTokens(Number(event.target.value))
          }
        />

        <label htmlFor="temperature">
          Temperature: {temperature.toFixed(1)}
        </label>
        <input
          id="temperature"
          type="range"
          min="0.1"
          max="2"
          step="0.1"
          value={temperature}
          onChange={(event) =>
            setTemperature(Number(event.target.value))
          }
        />
      </section>

      <button onClick={handleGenerate} disabled={loading}>
        {loading ? "Generating..." : "Generate"}
      </button>

      {error && <p className="error">{error}</p>}

      
{generatedText && (
  <section className="result">
    <div className="result-header">
      <h2>Generated text</h2>
      <button
        type="button"
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(generatedText);
          } catch (err) {
            console.error("Copy failed:", err);
            setError("Could not copy the text. Please copy it manually.");
          }
        }}
      >
        Copy output
      </button>
    </div>

    <pre>{generatedText}</pre>
  </section>
)}

    </main>
  );
}

export default App;
