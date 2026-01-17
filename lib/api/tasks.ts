import { getSupabaseClient } from '@/lib/supabase';
import type { Task, TaskWithDetails } from '@/lib/types';

// Get all tasks
export async function getTasks() {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tasks')
    .select('*')
    .order('created_at', { ascending: false });

  if (error) throw error;
  return data as Task[];
}

// Get tasks with all details (project, assigned user, comments)
export async function getTasksWithDetails() {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tasks')
    .select(`
      *,
      project:projects(*),
      assigned_user:users!tasks_assigned_to_fkey(*),
      comments:comments(
        *,
        user:users(*)
      )
    `)
    .order('created_at', { ascending: false });

  if (error) throw error;
  return data as TaskWithDetails[];
}

// Get task by ID
export async function getTaskById(id: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tasks')
    .select('*')
    .eq('id', id)
    .single();

  if (error) throw error;
  return data as Task;
}

// Get task with details
export async function getTaskWithDetails(id: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tasks')
    .select(`
      *,
      project:projects(*),
      assigned_user:users!tasks_assigned_to_fkey(*),
      comments:comments(
        *,
        user:users(*)
      )
    `)
    .eq('id', id)
    .single();

  if (error) throw error;
  return data as TaskWithDetails;
}

// Get tasks by project ID
export async function getTasksByProjectId(projectId: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tasks')
    .select('*')
    .eq('project_id', projectId)
    .order('created_at', { ascending: false });

  if (error) throw error;
  return data as Task[];
}

// Get tasks assigned to a user
export async function getTasksByAssignedUser(userId: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tasks')
    .select('*')
    .eq('assigned_to', userId)
    .order('due_date', { ascending: true, nullsFirst: false });

  if (error) throw error;
  return data as Task[];
}

// Get tasks by status
export async function getTasksByStatus(status: string) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tasks')
    .select('*')
    .eq('status', status)
    .order('created_at', { ascending: false });

  if (error) throw error;
  return data as Task[];
}

// Create a new task
export async function createTask(task: Omit<Task, 'id' | 'created_at' | 'updated_at'>) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tasks')
    .insert(task)
    .select()
    .single();

  if (error) throw error;
  return data as Task;
}

// Update task
export async function updateTask(id: number, updates: Partial<Omit<Task, 'id' | 'created_at' | 'updated_at'>>) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tasks')
    .update(updates)
    .eq('id', id)
    .select()
    .single();

  if (error) throw error;
  return data as Task;
}

// Mark task as completed
export async function completeTask(id: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tasks')
    .update({
      status: 'completed',
      completed_at: new Date().toISOString()
    })
    .eq('id', id)
    .select()
    .single();

  if (error) throw error;
  return data as Task;
}

// Delete task
export async function deleteTask(id: number) {
  const supabase = getSupabaseClient();
  const { error } = await supabase
    .from('tasks')
    .delete()
    .eq('id', id);

  if (error) throw error;
  return true;
}

// Assign task to user
export async function assignTask(taskId: number, userId: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tasks')
    .update({ assigned_to: userId })
    .eq('id', taskId)
    .select()
    .single();

  if (error) throw error;
  return data as Task;
}

// Unassign task
export async function unassignTask(taskId: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('tasks')
    .update({ assigned_to: null })
    .eq('id', taskId)
    .select()
    .single();

  if (error) throw error;
  return data as Task;
}
