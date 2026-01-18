'use client';

import { useState } from 'react';

export default function Console() {
  const [input, setInput] = useState('');
  const [logs, setLogs] = useState<{ type: 'info' | 'error' | 'success'; message: string }[]>([
    { type: 'info', message: 'Console initialized' },
    { type: 'info', message: 'Ready for input' },
  ]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    setLogs((prev) => [
      ...prev,
      { type: 'info', message: `> ${input}` },
      { type: 'success', message: 'Command executed successfully' },
    ]);
    setInput('');
  };

  const clearLogs = () => {
    setLogs([]);
  };

  return (
    <main className="min-h-screen bg-gray-900">
      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="space-y-6">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Console</h1>
            <p className="text-gray-400">
              Interactive console for real-time analysis and command execution.
            </p>
          </div>

          {/* Console Output */}
          <div className="bg-gray-950 rounded-lg border border-gray-700 overflow-hidden flex flex-col h-96">
            <div className="flex items-center justify-between bg-gray-800 px-4 py-3 border-b border-gray-700">
              <span className="text-sm font-mono text-gray-400">Console Output</span>
              <button
                onClick={clearLogs}
                className="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded transition-colors"
              >
                Clear
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-2 font-mono text-sm">
              {logs.map((log, idx) => (
                <div
                  key={idx}
                  className={
                    log.type === 'error'
                      ? 'text-red-400'
                      : log.type === 'success'
                        ? 'text-green-400'
                        : 'text-gray-300'
                  }
                >
                  {log.message}
                </div>
              ))}
            </div>
          </div>

          {/* Command Input */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="bg-gray-950 rounded-lg border border-gray-700 p-4">
              <label className="block text-sm font-mono text-gray-400 mb-2">Input</label>
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Enter command..."
                className="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 font-mono text-sm text-white placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <button
              type="submit"
              className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
            >
              Execute
            </button>
          </form>

          <div className="bg-yellow-900 border border-yellow-700 rounded-lg p-4">
            <h3 className="font-semibold text-yellow-100 mb-2">Console Status</h3>
            <p className="text-yellow-100 text-sm">
              The console is ready for commands. This is a basic interface for testing and development.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
