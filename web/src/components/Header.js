import React from 'react';

function Header() {
  return (
    <header className="bg-white shadow">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Emotion Analysis</h1>
            <p className="text-gray-600">Analyze emotions in text using AI</p>
          </div>
          <div className="flex items-center space-x-4">
            <a
              href="https://www.linkedin.com/in/yechezkel-matlin"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-600 hover:text-gray-900"
            >
              chezky linkdin
            </a>
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header; 