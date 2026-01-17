import { getSupabaseClient } from '@/lib/supabase';
import type { Project, ProjectWithUser, ProjectWithTags, ProjectStats } from '@/lib/types';

// Get all projects
export async function getProjects() {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('projects')
    .select('*')
    .order('created_at', { ascending: false });

  if (error) throw error;
  return data as Project[];
}

// Get projects with user details
export async function getProjectsWithUsers() {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('projects')
    .select(`
      *,
      user:users(*)
    `)
    .order('created_at', { ascending: false });

  if (error) throw error;
  return data as ProjectWithUser[];
}

// Get project by ID
export async function getProjectById(id: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('projects')
    .select('*')
    .eq('id', id)
    .single();

  if (error) throw error;
  return data as Project;
}

// Get project with tags
export async function getProjectWithTags(id: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('projects')
    .select(`
      *,
      tags:project_tags(tag:tags(*))
    `)
    .eq('id', id)
    .single();

  if (error) throw error;

  // Transform the data to flatten tags
  const project = data as any;
  const transformedProject: ProjectWithTags = {
    ...project,
    tags: project.tags?.map((pt: any) => pt.tag) || []
  };

  return transformedProject;
}

// Get projects by user ID
export async function getProjectsByUserId(userId: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('projects')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false });

  if (error) throw error;
  return data as Project[];
}

// Get projects by status
export async function getProjectsByStatus(status: string) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('projects')
    .select('*')
    .eq('status', status)
    .order('created_at', { ascending: false });

  if (error) throw error;
  return data as Project[];
}

// Create a new project
export async function createProject(project: Omit<Project, 'id' | 'created_at' | 'updated_at'>) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('projects')
    .insert(project)
    .select()
    .single();

  if (error) throw error;
  return data as Project;
}

// Update project
export async function updateProject(id: number, updates: Partial<Omit<Project, 'id' | 'created_at' | 'updated_at'>>) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('projects')
    .update(updates)
    .eq('id', id)
    .select()
    .single();

  if (error) throw error;
  return data as Project;
}

// Delete project
export async function deleteProject(id: number) {
  const supabase = getSupabaseClient();
  const { error } = await supabase
    .from('projects')
    .delete()
    .eq('id', id);

  if (error) throw error;
  return true;
}

// Get project statistics using the database function
export async function getProjectStats(projectId: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .rpc('get_project_stats', { project_id_param: projectId });

  if (error) throw error;
  return data[0] as ProjectStats;
}

// Add tag to project
export async function addTagToProject(projectId: number, tagId: number) {
  const supabase = getSupabaseClient();
  const { error } = await supabase
    .from('project_tags')
    .insert({ project_id: projectId, tag_id: tagId });

  if (error) throw error;
  return true;
}

// Remove tag from project
export async function removeTagFromProject(projectId: number, tagId: number) {
  const supabase = getSupabaseClient();
  const { error } = await supabase
    .from('project_tags')
    .delete()
    .eq('project_id', projectId)
    .eq('tag_id', tagId);

  if (error) throw error;
  return true;
}
