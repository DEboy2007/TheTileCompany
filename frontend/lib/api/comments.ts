import { getSupabaseClient } from '@/lib/supabase';
import type { Comment, CommentWithUser } from '@/lib/types';

// Get all comments for a task
export async function getCommentsByTaskId(taskId: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('comments')
    .select('*')
    .eq('task_id', taskId)
    .order('created_at', { ascending: true });

  if (error) throw error;
  return data as Comment[];
}

// Get comments with user details
export async function getCommentsWithUserByTaskId(taskId: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('comments')
    .select(`
      *,
      user:users(*)
    `)
    .eq('task_id', taskId)
    .order('created_at', { ascending: true });

  if (error) throw error;
  return data as CommentWithUser[];
}

// Get comment by ID
export async function getCommentById(id: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('comments')
    .select('*')
    .eq('id', id)
    .single();

  if (error) throw error;
  return data as Comment;
}

// Create a new comment
export async function createComment(comment: Omit<Comment, 'id' | 'created_at' | 'updated_at'>) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('comments')
    .insert(comment)
    .select()
    .single();

  if (error) throw error;
  return data as Comment;
}

// Update comment
export async function updateComment(id: number, content: string) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('comments')
    .update({ content })
    .eq('id', id)
    .select()
    .single();

  if (error) throw error;
  return data as Comment;
}

// Delete comment
export async function deleteComment(id: number) {
  const supabase = getSupabaseClient();
  const { error } = await supabase
    .from('comments')
    .delete()
    .eq('id', id);

  if (error) throw error;
  return true;
}

// Get comments by user
export async function getCommentsByUserId(userId: number) {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from('comments')
    .select('*')
    .eq('user_id', userId)
    .order('created_at', { ascending: false });

  if (error) throw error;
  return data as Comment[];
}
