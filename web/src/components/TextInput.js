import React from 'react';

function TextInput({ value, onChange }) {
  return (
    <div className="card">
      <label htmlFor="text" className="block text-sm font-medium text-gray-700 mb-2">
        Enter text to analyze
      </label>
      <textarea
        id="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="input h-32"
        placeholder="Type or paste your text here..."
      />
    </div>
  );
}

export default TextInput; 