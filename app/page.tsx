'use client';

import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';

export default function Home() {
  const [connected, setConnected] = useState<boolean | null>(null);

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const { error } = await supabase.from('_health_check').select('*').limit(1);
        setConnected(!error || error.code !== 'PGRST301');
      } catch {
        setConnected(false);
      }
    };
    checkConnection();
  }, []);

  return (
    <main className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="text-center space-y-6">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          Welcome to NexHacks Project
        </h1>
        <p className="text-lg text-gray-600">
          Next.js + Supabase + Tailwind CSS
        </p>

        <div className="mt-8 p-6 bg-white rounded-lg shadow-md max-w-md mx-auto">
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
          {connected === false && (
            <p className="mt-4 text-sm text-gray-500">
              Add your Supabase credentials to .env.local to connect
            </p>
          )}
        </div>
      </div>
    </main>
  );
}
