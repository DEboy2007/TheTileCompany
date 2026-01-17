'use client';

import { useEffect, useState } from 'react';
import { createClient } from '@supabase/supabase-js';

export default function Home() {
  const [connected, setConnected] = useState<boolean | null>(null);
  const [showCredentialsForm, setShowCredentialsForm] = useState(false);
  const [supabaseUrl, setSupabaseUrl] = useState('');
  const [supabaseKey, setSupabaseKey] = useState('');
  const [savedCredentials, setSavedCredentials] = useState<{ url: string; key: string } | null>(null);

  useEffect(() => {
    // Load saved credentials from localStorage
    const saved = localStorage.getItem('supabase_credentials');
    if (saved) {
      try {
        const creds = JSON.parse(saved);
        setSavedCredentials(creds);
        setSupabaseUrl(creds.url);
        setSupabaseKey(creds.key);
      } catch (e) {
        console.error('Failed to parse saved credentials');
      }
    }
  }, []);

  useEffect(() => {
    const checkConnection = async () => {
      if (!savedCredentials) {
        setConnected(false);
        return;
      }

      try {
        const client = createClient(savedCredentials.url, savedCredentials.key);
        const { error } = await client.from('_health_check').select('*').limit(1);
        setConnected(!error || error.code !== 'PGRST301');
      } catch {
        setConnected(false);
      }
    };
    checkConnection();
  }, [savedCredentials]);

  const handleSaveCredentials = (e: React.FormEvent) => {
    e.preventDefault();

    if (!supabaseUrl || !supabaseKey) {
      alert('Please enter both URL and API key');
      return;
    }

    const credentials = { url: supabaseUrl, key: supabaseKey };
    localStorage.setItem('supabase_credentials', JSON.stringify(credentials));
    setSavedCredentials(credentials);
    setShowCredentialsForm(false);
    setConnected(null);
  };

  const handleClearCredentials = () => {
    localStorage.removeItem('supabase_credentials');
    setSavedCredentials(null);
    setSupabaseUrl('');
    setSupabaseKey('');
    setConnected(false);
  };

  return (
    <main className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="w-full max-w-2xl space-y-6">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Welcome to NexHacks Project
          </h1>
          <p className="text-lg text-gray-600">
            Next.js + Supabase + Tailwind CSS
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">Status</h2>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Next.js:</span>
              <span className="text-green-600 font-medium">✓ Ready</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Tailwind CSS:</span>
              <span className="text-green-600 font-medium">✓ Ready</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Supabase:</span>
              {connected === null ? (
                <span className="text-gray-400 font-medium">Checking...</span>
              ) : connected ? (
                <span className="text-green-600 font-medium">✓ Connected</span>
              ) : (
                <span className="text-amber-600 font-medium">⚠ Not configured</span>
              )}
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-800">Supabase Credentials</h2>
            {savedCredentials && (
              <button
                onClick={handleClearCredentials}
                className="text-sm text-red-600 hover:text-red-700 font-medium"
              >
                Clear
              </button>
            )}
          </div>

          {!showCredentialsForm && !savedCredentials && (
            <div className="text-center py-4">
              <p className="text-gray-600 mb-4">No credentials saved</p>
              <button
                onClick={() => setShowCredentialsForm(true)}
                className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 transition-colors"
              >
                Add Credentials
              </button>
            </div>
          )}

          {!showCredentialsForm && savedCredentials && (
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Supabase URL
                </label>
                <div className="text-sm text-gray-900 bg-gray-50 p-2 rounded border border-gray-200 font-mono break-all">
                  {savedCredentials.url}
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  API Key
                </label>
                <div className="text-sm text-gray-900 bg-gray-50 p-2 rounded border border-gray-200 font-mono">
                  {savedCredentials.key.substring(0, 20)}...
                </div>
              </div>
              <button
                onClick={() => setShowCredentialsForm(true)}
                className="w-full bg-gray-100 text-gray-700 px-4 py-2 rounded-md hover:bg-gray-200 transition-colors"
              >
                Update Credentials
              </button>
            </div>
          )}

          {showCredentialsForm && (
            <form onSubmit={handleSaveCredentials} className="space-y-4">
              <div>
                <label htmlFor="supabase-url" className="block text-sm font-medium text-gray-700 mb-1">
                  Supabase URL
                </label>
                <input
                  id="supabase-url"
                  type="url"
                  value={supabaseUrl}
                  onChange={(e) => setSupabaseUrl(e.target.value)}
                  placeholder="https://your-project.supabase.co"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  required
                />
              </div>
              <div>
                <label htmlFor="supabase-key" className="block text-sm font-medium text-gray-700 mb-1">
                  Supabase Anon Key
                </label>
                <input
                  id="supabase-key"
                  type="password"
                  value={supabaseKey}
                  onChange={(e) => setSupabaseKey(e.target.value)}
                  placeholder="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  required
                />
              </div>
              <div className="flex gap-2">
                <button
                  type="submit"
                  className="flex-1 bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 transition-colors"
                >
                  Save Credentials
                </button>
                <button
                  type="button"
                  onClick={() => setShowCredentialsForm(false)}
                  className="flex-1 bg-gray-100 text-gray-700 px-4 py-2 rounded-md hover:bg-gray-200 transition-colors"
                >
                  Cancel
                </button>
              </div>
            </form>
          )}
        </div>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-blue-900 mb-2">Local PostgreSQL Database</h3>
          <p className="text-sm text-blue-800 mb-2">
            Want to test locally? Run the sample PostgreSQL database:
          </p>
          <code className="block bg-blue-100 text-blue-900 px-3 py-2 rounded text-xs font-mono">
            cd database && ./start.sh
          </code>
        </div>
      </div>
    </main>
  );
}
