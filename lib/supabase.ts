import { createClient, SupabaseClient } from '@supabase/supabase-js';

// Default client using environment variables
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Helper function to get Supabase client with saved credentials
export function getSupabaseClient(): SupabaseClient {
  if (typeof window === 'undefined') {
    // Server-side: use environment variables
    return supabase;
  }

  // Client-side: try to use saved credentials from localStorage
  try {
    const saved = localStorage.getItem('supabase_credentials');
    if (saved) {
      const creds = JSON.parse(saved);
      if (creds.url && creds.key) {
        return createClient(creds.url, creds.key);
      }
    }
  } catch (e) {
    console.error('Failed to load saved credentials');
  }

  // Fallback to environment variables
  return supabase;
}
