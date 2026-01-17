// Database type definitions

export interface User {
  id: number;
  username: string;
  email: string;
  full_name: string | null;
  avatar_url: string | null;
  bio: string | null;
  created_at: string;
  updated_at: string;
}

export interface Project {
  id: number;
  user_id: number;
  title: string;
  description: string | null;
  status: 'active' | 'planning' | 'completed' | 'archived';
  priority: 'low' | 'medium' | 'high';
  start_date: string | null;
  end_date: string | null;
  created_at: string;
  updated_at: string;
}

export interface Task {
  id: number;
  project_id: number;
  title: string;
  description: string | null;
  status: 'pending' | 'in_progress' | 'completed' | 'cancelled';
  assigned_to: number | null;
  due_date: string | null;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface Comment {
  id: number;
  task_id: number;
  user_id: number;
  content: string;
  created_at: string;
  updated_at: string;
}

export interface Tag {
  id: number;
  name: string;
  color: string;
  created_at: string;
}

export interface ProjectTag {
  project_id: number;
  tag_id: number;
}

export interface ProjectStats {
  total_tasks: number;
  completed_tasks: number;
  pending_tasks: number;
  in_progress_tasks: number;
  completion_percentage: number;
}

// Extended types with relations
export interface ProjectWithUser extends Project {
  user?: User;
}

export interface TaskWithDetails extends Task {
  project?: Project;
  assigned_user?: User;
  comments?: CommentWithUser[];
}

export interface CommentWithUser extends Comment {
  user?: User;
}

export interface ProjectWithTags extends Project {
  tags?: Tag[];
}
