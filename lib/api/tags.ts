import { getSupabaseClient } from '@/lib/supabase';
import type { Tag } from '@/lib/types';

// Get all tags
export async function getTags() {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tags')
    .select('*')
    .order('name', { ascending: true });

  if (error) throw error;
  return data as Tag[];
}

// Get tag by ID
export async function getTagById(id: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tags')
    .select('*')
    .eq('id', id)
    .single();

  if (error) throw error;
  return data as Tag;
}

// Get tag by name
export async function getTagByName(name: string) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tags')
    .select('*')
    .eq('name', name)
    .single();

  if (error) throw error;
  return data as Tag;
}

// Create a new tag
export async function createTag(tag: Omit<Tag, 'id' | 'created_at'>) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tags')
    .insert(tag)
    .select()
    .single();

  if (error) throw error;
  return data as Tag;
}

// Update tag
export async function updateTag(id: number, updates: Partial<Omit<Tag, 'id' | 'created_at'>>) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tags')
    .update(updates)
    .eq('id', id)
    .select()
    .single();

  if (error) throw error;
  return data as Tag;
}

// Delete tag
export async function deleteTag(id: number) {
  const supabase = getSupabaseClient();
  const { error } = await supabase
    .from('tags')
    .delete()
    .eq('id', id);

  if (error) throw error;
  return true;
}

// Get tags for a project
export async function getTagsByProjectId(projectId: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('project_tags')
    .select('tag:tags(*)')
    .eq('project_id', projectId);

  if (error) throw error;
  return data.map((pt: any) => pt.tag) as Tag[];
}
