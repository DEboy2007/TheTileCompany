'use client';

import { useState, CSSProperties } from 'react';
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter';
import json from 'react-syntax-highlighter/dist/esm/languages/hljs/json';
import python from 'react-syntax-highlighter/dist/esm/languages/hljs/python';
import bash from 'react-syntax-highlighter/dist/esm/languages/hljs/bash';

SyntaxHighlighter.registerLanguage('json', json);
SyntaxHighlighter.registerLanguage('python', python);
SyntaxHighlighter.registerLanguage('bash', bash);

// Custom theme with cyan, pink, blue, orange, purple color scheme
const customTheme: { [key: string]: CSSProperties } = {
  hljs: {
    display: 'block',
    overflowX: 'auto',
    color: '#ffffff',
    background: '#0a0a0a',
  },
  'hljs-string': {
    color: '#ec4899', // pink-400
  },
  'hljs-number': {
    color: '#fb923c', // orange-300
  },
  'hljs-literal': {
    color: '#a78bfa', // purple-400
  },
  'hljs-attr': {
    color: '#60a5fa', // blue-300
  },
  'hljs-property': {
    color: '#60a5fa', // blue-300
  },
  'hljs-keyword': {
    color: '#ffffff', // white
  },
  'hljs-title': {
    color: '#ffffff', // white
  },
  'hljs-built_in': {
    color: '#ffffff', // white
  },
  'hljs-selector-tag': {
    color: '#ffffff', // white
  },
  'hljs-name': {
    color: '#ffffff', // white
  },
  'hljs-variable': {
    color: '#ffffff', // white
  },
  'hljs-operator': {
    color: '#ffffff', // white
  },
  'hljs-punctuation': {
    color: '#f472b6', // pink-400
  },
  'hljs-quote': {
    color: '#ec4899', // pink-300
  },
  'hljs-meta': {
    color: '#22d3ee', // cyan-300
  },
  'hljs-params': {
    color: '#22d3ee', // cyan-300 (for command flags)
  },
  'hljs-doctag': {
    color: '#60a5fa', // blue-300
  },
  'hljs-code': {
    color: '#ffffff', // white
  },
  'hljs-section': {
    color: '#ffffff', // white
  },
  'hljs-link': {
    color: '#ec4899', // pink-400 (for URLs)
  },
  'hljs-type': {
    color: '#ffffff', // white
  },
  'hljs-subst': {
    color: '#ffffff', // white
  },
  'hljs-addition': {
    color: '#ffffff', // white
  },
  'hljs-deletion': {
    color: '#ffffff', // white
  },
  'hljs-tag': {
    color: '#ffffff', // white
  },
  'hljs-regexp': {
    color: '#ec4899', // pink-400
  },
  'hljs-symbol': {
    color: '#a78bfa', // purple-400
  },
  'hljs-comment': {
    color: '#6b7280', // gray-500 (keep gray, not colored)
  },
  'hljs-class': {
    color: '#ffffff', // white
  },
  'hljs-function': {
    color: '#ffffff', // white
  },
  'hljs-literal-string': {
    color: '#ec4899', // pink-400
  },
  'hljs-bash': {
    color: '#ffffff', // white
  },
  'hljs-literal-string-literal': {
    color: '#ec4899', // pink-400
  },
  'language-bash': {
    color: '#ffffff', // white
  },
};

interface CodeBlockProps {
  code: string;
  language?: 'json' | 'python' | 'bash';
  filename?: string;
}

export default function CodeBlock({ code, language = 'json', filename = 'code' }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-[#0a0a0a] rounded-lg overflow-hidden border border-gray-700">
      {/* Mac-style header */}
      <div className="bg-[#1a1a1a] px-4 py-3 flex items-center justify-between border-b border-gray-700">
        <div className="flex items-center gap-3">
          {/* Traffic lights */}
          <div className="flex gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
          </div>
          {/* Filename */}
          <span className="text-gray-400 font-mono text-sm ml-2">{filename}</span>
        </div>
        {/* Copy button */}
        <button
          onClick={handleCopy}
          className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors text-sm font-mono"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
            />
          </svg>
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>

      {/* Code */}
      <SyntaxHighlighter
        language={language}
        style={customTheme}
        customStyle={{
          margin: 0,
          padding: '1rem',
          fontSize: '0.875rem',
          lineHeight: '1.5',
          fontFamily: 'IBM Plex Mono, monospace',
          background: '#0a0a0a',
        }}
        showLineNumbers={false}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}
