import React, { useState } from 'react';
import { toast } from 'react-toastify';
import axios from 'axios';
import Header from './components/Header';
import TextInput from './components/TextInput';
import Results from './components/Results';
import ApproachSelector from './components/ApproachSelector';
import InfoPanel from './components/InfoPanel';

function App() {
  const [text, setText] = useState('');
  const [approach, setApproach] = useState('llm');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeText = async () => {
    if (!text.trim()) {
      toast.error('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:8000/analyze', {
        text,
        approach
      });
      setResults(response.data);
      toast.success('Analysis completed successfully');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-3xl mx-auto space-y-6">
          <ApproachSelector value={approach} onChange={setApproach} />
          <TextInput value={text} onChange={setText} />
          <button
            onClick={analyzeText}
            disabled={loading}
            className="btn btn-primary w-full"
          >
            {loading ? 'Analyzing...' : 'Analyze Text'}
          </button>
          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
              {error}
            </div>
          )}
          {results && <Results data={results} />}
        </div>
      </main>
      <InfoPanel />
    </div>
  );
}

export default App; 