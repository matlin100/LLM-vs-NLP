import React, { useState } from 'react';

const EMOTION_CONFIG = {
  danger: { 
    bg: 'bg-red-100', 
    text: 'text-red-800',
    label: 'DANGER',
    icon: '⚠️'
  },
  emotional_distress: { 
    bg: 'bg-yellow-100', 
    text: 'text-yellow-800',
    label: 'DISTRESS',
    icon: '😔'
  },
  emotional_progress: { 
    bg: 'bg-green-100', 
    text: 'text-green-800',
    label: 'PROGRESS',
    icon: '📈'
  },
  emotionally_intense: { 
    bg: 'bg-purple-100', 
    text: 'text-purple-800',
    label: 'INTENSE',
    icon: '💫'
  }
};

const getEmotionColor = (label) => {
  return EMOTION_CONFIG[label] || { bg: 'bg-gray-100', text: 'text-gray-800' };
};

const ColorLegend = ({ showColors }) => (
  <div className={`transition-opacity duration-300 ${showColors ? 'opacity-100' : 'opacity-0'}`}>
    <div className="mt-4 p-3 bg-gray-50 rounded-lg">
      <h3 className="text-sm font-medium text-gray-700 mb-2">Color Legend:</h3>
      <div className="flex flex-wrap gap-2">
        {Object.entries(EMOTION_CONFIG).map(([key, config]) => (
          <div key={key} className="flex items-center">
            <span className={`${config.bg} ${config.text} px-2 py-1 rounded text-sm whitespace-nowrap`}>
              {config.icon} {config.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  </div>
);

const Results = ({ data }) => {
  const [showColors, setShowColors] = useState(true);

  const renderTag = (tag) => {
    const displayText = tag.text;
    const config = EMOTION_CONFIG[tag.label];
    if (!config) return displayText; // Fallback for unknown labels

    return (
      <span 
        className={`
          transition-all duration-300 ease-in-out
          ${showColors 
            ? `${config.bg} ${config.text} px-1 rounded` 
            : 'underline decoration-dotted decoration-2'
          }
        `}
        title={`${config.icon} ${config.label}`}
      >
        {displayText}
        <span className="ml-1 text-xs font-medium">({config.label})</span>
      </span>
    );
  };

  const renderText = () => {
    if (!data.text) return null;

    let result = [];
    let lastIndex = 0;

    // Sort tags by start position
    const sortedTags = [...data.tags].sort((a, b) => a.start - b.start);

    sortedTags.forEach((tag, index) => {
      // Add text before the tag
      if (tag.start > lastIndex) {
        result.push(
          <span key={`text-${index}`}>
            {data.text.slice(lastIndex, tag.start)}
          </span>
        );
      }
      // Add the tagged text
      result.push(
        <span key={`tag-${index}`} className="mx-1">
          {renderTag(tag)}
        </span>
      );
      lastIndex = tag.end;
    });

    // Add any remaining text
    if (lastIndex < data.text.length) {
      result.push(
        <span key="text-end">
          {data.text.slice(lastIndex)}
        </span>
      );
    }

    return result;
  };

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-lg font-semibold text-gray-700">Analysis Results</h2>
        <button
          onClick={() => setShowColors(!showColors)}
          className="flex items-center gap-2 text-sm font-medium text-gray-600 hover:text-gray-800 transition-colors duration-300"
        >
          <span className="transition-transform duration-300 transform hover:scale-110">
            {showColors ? '🎨' : '⚪'}
          </span>
          <span>{showColors ? 'Hide Colors' : 'Show Colors'}</span>
        </button>
      </div>
      
      <div className="prose max-w-none">
        <p className="text-gray-600 leading-relaxed">{renderText()}</p>
      </div>

      <div className="text-sm text-gray-500">
        Found {data.tags.length} emotional expressions
      </div>

      <ColorLegend showColors={showColors} />
    </div>
  );
};

export default Results; 