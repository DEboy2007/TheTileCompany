'use client';

import { useState } from 'react';

interface DatabaseModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export default function DatabaseModal({ open, onOpenChange }: DatabaseModalProps) {
  const [dbType, setDbType] = useState<'postgres' | 'supabase'>('supabase');
  const [connectionString, setConnectionString] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!connectionString.trim()) return;

    setIsSaving(true);
    try {
      // TODO: Save connection string securely
      localStorage.setItem(
        'database_connection',
        JSON.stringify({ type: dbType, connection: connectionString })
      );
      onOpenChange(false);
      setConnectionString('');
    } finally {
      setIsSaving(false);
    }
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
      <div className="bg-white rounded-lg shadow-lg w-full max-w-md">
        <div className="border-b border-gray-100 px-6 py-4">
          <h2 className="text-lg font-bold text-gray-900">Connect Database</h2>
          <p className="text-sm text-gray-600 mt-1">Select your database type and provide connection details</p>
        </div>

        <form onSubmit={handleSave} className="p-6 space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-3">Database Type</label>
            <div className="space-y-2">
              {(['supabase', 'postgres'] as const).map((type) => (
                <label key={type} className="flex items-center cursor-pointer p-3 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors" style={{
                  borderColor: dbType === type ? '#b45309' : undefined,
                  backgroundColor: dbType === type ? '#fffbf0' : undefined,
                }}>
                  <input
                    type="radio"
                    name="dbType"
                    value={type}
                    checked={dbType === type}
                    onChange={(e) => setDbType(e.target.value as typeof dbType)}
                    className="w-4 h-4 cursor-pointer"
                  />
                  <span className="ml-3 capitalize font-medium text-gray-900">{type}</span>
                </label>
              ))}
            </div>
          </div>

          <div>
            <label htmlFor="connection-string" className="block text-sm font-medium text-gray-700 mb-2">
              {dbType === 'supabase' ? 'Supabase Connection String' : 'PostgreSQL Connection String'}
            </label>
            <textarea
              id="connection-string"
              value={connectionString}
              onChange={(e) => setConnectionString(e.target.value)}
              placeholder={
                dbType === 'supabase'
                  ? 'postgresql://user:password@host:5432/database'
                  : 'postgresql://user:password@host:5432/database'
              }
              className="w-full px-4 py-2 rounded-lg border border-gray-200 bg-gray-50 text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:bg-white transition-colors resize-none"
              rows={4}
            />
          </div>

          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => {
                onOpenChange(false);
                setConnectionString('');
              }}
              className="flex-1 px-4 py-2 rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-900 font-medium transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!connectionString.trim() || isSaving}
              className="flex-1 px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-700 disabled:bg-gray-300 text-white font-medium transition-colors disabled:cursor-not-allowed"
            >
              {isSaving ? 'Saving...' : 'Save'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
