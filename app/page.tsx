'use client';

import { useState } from 'react';
import DatabaseModal from '@/components/DatabaseModal';

export default function Home() {
  const [prompt, setPrompt] = useState('');
  const [mode, setMode] = useState<'dashboard' | 'interpret'>('dashboard');
  const [showDatabaseModal, setShowDatabaseModal] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setIsSubmitting(true);
    try {
      // TODO: Add API call to handle prompt submission
      console.log({
        prompt,
        mode,
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-amber-50 via-white to-orange-50 p-6 flex items-center justify-center">
      <div className="w-full max-w-2xl">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Dashboard Builder</h1>
          <p className="text-gray-600">Generate interactive dashboards with natural language</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-6">
            <label htmlFor="prompt" className="block text-sm font-medium text-gray-700 mb-3">
              Prompt
            </label>
            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the dashboard you want to create..."
              className="w-full px-4 py-3 rounded-lg border border-gray-200 bg-gray-50 text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:bg-white transition-colors resize-none"
              rows={6}
            />
          </div>

          <div className="bg-white rounded-lg shadow-sm border border-gray-100 p-6">
            <label className="block text-sm font-medium text-gray-700 mb-4">Mode</label>
            <div className="flex gap-4">
              {(['dashboard', 'interpret'] as const).map((option) => (
                <label key={option} className="flex items-center cursor-pointer">
                  <input
                    type="radio"
                    name="mode"
                    value={option}
                    checked={mode === option}
                    onChange={(e) => setMode(e.target.value as typeof mode)}
                    className="w-4 h-4 text-amber-600 bg-gray-100 border-gray-300 focus:ring-amber-500 cursor-pointer"
                  />
                  <span className="ml-3 text-gray-700 capitalize font-medium">{option}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => setShowDatabaseModal(true)}
              className="flex-1 px-4 py-3 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-900 font-medium transition-colors border border-gray-200"
            >
              Connect Database
            </button>
            <button
              type="submit"
              disabled={!prompt.trim() || isSubmitting}
              className="flex-1 px-4 py-3 rounded-lg bg-amber-600 hover:bg-amber-700 disabled:bg-gray-300 text-white font-medium transition-colors shadow-sm disabled:cursor-not-allowed"
            >
              {isSubmitting ? 'Generating...' : 'Generate'}
            </button>
          </div>
        </form>
      </div>

      <DatabaseModal open={showDatabaseModal} onOpenChange={setShowDatabaseModal} />
    </main>
  );
}
