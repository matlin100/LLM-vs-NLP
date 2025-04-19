import React from 'react';

function ApproachSelector({ value, onChange }) {
  const approaches = [
    { value: 'llm', label: 'LLM (OpenAI)', description: 'Advanced analysis using large language models' },
    { value: 'nlp', label: 'NLP', description: 'Basic natural language processing analysis' },
    { value: 'custom', label: 'Custom Model', description: 'Analysis using our trained model' },
  ];

  return (
    <div className="card">
      <label className="block text-sm font-medium text-gray-700 mb-2">
        Analysis Approach
      </label>
      <div className="space-y-2">
        {approaches.map((approach) => (
          <div key={approach.value} className="flex items-center">
            <input
              type="radio"
              id={approach.value}
              name="approach"
              value={approach.value}
              checked={value === approach.value}
              onChange={(e) => onChange(e.target.value)}
              className="h-4 w-4 text-primary-600 focus:ring-primary-500"
            />
            <label htmlFor={approach.value} className="ml-3">
              <span className="block text-sm font-medium text-gray-900">
                {approach.label}
              </span>
              <span className="block text-sm text-gray-500">
                {approach.description}
              </span>
            </label>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ApproachSelector; 