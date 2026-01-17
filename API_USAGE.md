# API Usage Guide

This document provides examples for using the Supabase backend API functions and Next.js API routes.

## Table of Contents
- [Database Schema](#database-schema)
- [Using API Functions (Client-Side)](#using-api-functions-client-side)
- [Using API Routes (HTTP)](#using-api-routes-http)
- [Examples](#examples)

## Database Schema

The database includes the following tables:

- **users** - User profiles
- **projects** - Project management
- **tasks** - Individual tasks within projects
- **comments** - Comments on tasks
- **tags** - Project tags
- **project_tags** - Many-to-many relationship between projects and tags

## Using API Functions (Client-Side)

Import the API functions from `@/lib/api`:

```typescript
import {
  getProjects,
  createProject,
  updateProject,
  getTasks,
  createTask,
  completeTask
} from '@/lib/api';
```

### Projects

```typescript
// Get all projects
const projects = await getProjects();

// Get projects with user details
const projectsWithUsers = await getProjectsWithUsers();

// Get project by ID
const project = await getProjectById(1);

// Get project with tags
const projectWithTags = await getProjectWithTags(1);

// Get project statistics
const stats = await getProjectStats(1);

// Create a new project
const newProject = await createProject({
  user_id: 1,
  title: 'My New Project',
  description: 'Project description',
  status: 'active',
  priority: 'high',
  start_date: '2026-01-01',
  end_date: '2026-12-31'
});

// Update project
const updated = await updateProject(1, {
  status: 'completed',
  priority: 'low'
});

// Delete project
await deleteProject(1);

// Add tag to project
await addTagToProject(1, 2);

// Remove tag from project
await removeTagFromProject(1, 2);
```

### Tasks

```typescript
// Get all tasks
const tasks = await getTasks();

// Get tasks with full details (project, user, comments)
const tasksWithDetails = await getTasksWithDetails();

// Get task by ID
const task = await getTaskById(1);

// Get task with details
const taskWithDetails = await getTaskWithDetails(1);

// Get tasks by project
const projectTasks = await getTasksByProjectId(1);

// Get tasks by assigned user
const userTasks = await getTasksByAssignedUser(1);

// Get tasks by status
const pendingTasks = await getTasksByStatus('pending');

// Create a new task
const newTask = await createTask({
  project_id: 1,
  title: 'Implement feature X',
  description: 'Detailed description',
  status: 'pending',
  assigned_to: 2,
  due_date: '2026-02-15T00:00:00Z'
});

// Update task
const updatedTask = await updateTask(1, {
  status: 'in_progress'
});

// Complete task
const completedTask = await completeTask(1);

// Assign task to user
await assignTask(1, 2);

// Unassign task
await unassignTask(1);

// Delete task
await deleteTask(1);
```

### Comments

```typescript
// Get comments for a task
const comments = await getCommentsByTaskId(1);

// Get comments with user details
const commentsWithUsers = await getCommentsWithUserByTaskId(1);

// Create a comment
const newComment = await createComment({
  task_id: 1,
  user_id: 1,
  content: 'This is my comment'
});

// Update comment
const updatedComment = await updateComment(1, 'Updated content');

// Delete comment
await deleteComment(1);
```

### Users

```typescript
// Get all users
const users = await getUsers();

// Get user by ID
const user = await getUserById(1);

// Get user by email
const user = await getUserByEmail('john@example.com');

// Create user
const newUser = await createUser({
  username: 'newuser',
  email: 'newuser@example.com',
  full_name: 'New User',
  bio: 'My bio'
});

// Update user
const updated = await updateUser(1, {
  bio: 'Updated bio'
});

// Delete user
await deleteUser(1);
```

### Tags

```typescript
// Get all tags
const tags = await getTags();

// Get tag by ID
const tag = await getTagById(1);

// Get tags for a project
const projectTags = await getTagsByProjectId(1);

// Create tag
const newTag = await createTag({
  name: 'NewTag',
  color: '#FF5733'
});

// Update tag
const updated = await updateTag(1, {
  color: '#00FF00'
});

// Delete tag
await deleteTag(1);
```

## Using API Routes (HTTP)

### Projects

```bash
# Get all projects
GET /api/projects

# Get projects with user details
GET /api/projects?withUsers=true

# Get project by ID
GET /api/projects/1

# Get project with tags
GET /api/projects/1?withTags=true

# Get project with stats
GET /api/projects/1?withStats=true

# Create project
POST /api/projects
{
  "user_id": 1,
  "title": "New Project",
  "description": "Description",
  "status": "active",
  "priority": "high"
}

# Update project
PATCH /api/projects/1
{
  "status": "completed"
}

# Delete project
DELETE /api/projects/1
```

### Tasks

```bash
# Get all tasks
GET /api/tasks

# Get tasks with details
GET /api/tasks?withDetails=true

# Get tasks by status
GET /api/tasks?status=pending

# Get tasks by project
GET /api/tasks?projectId=1

# Get task by ID
GET /api/tasks/1

# Get task with details
GET /api/tasks/1?withDetails=true

# Create task
POST /api/tasks
{
  "project_id": 1,
  "title": "New Task",
  "description": "Description",
  "status": "pending",
  "assigned_to": 2
}

# Update task
PATCH /api/tasks/1
{
  "status": "in_progress"
}

# Delete task
DELETE /api/tasks/1
```

## Examples

### React Component Example

```typescript
'use client';

import { useEffect, useState } from 'react';
import { getProjects, getProjectStats } from '@/lib/api';
import type { Project, ProjectStats } from '@/lib/types';

export default function ProjectList() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadProjects() {
      try {
        const data = await getProjects();
        setProjects(data);
      } catch (error) {
        console.error('Failed to load projects:', error);
      } finally {
        setLoading(false);
      }
    }
    loadProjects();
  }, []);

  if (loading) return <div>Loading...</div>;

  return (
    <div>
      <h1>Projects</h1>
      {projects.map(project => (
        <div key={project.id}>
          <h2>{project.title}</h2>
          <p>{project.description}</p>
          <span>Status: {project.status}</span>
        </div>
      ))}
    </div>
  );
}
```

### Fetching from API Routes

```typescript
// Using fetch with API routes
async function fetchProjects() {
  const response = await fetch('/api/projects?withUsers=true');
  const result = await response.json();

  if (result.success) {
    return result.data;
  } else {
    throw new Error(result.error);
  }
}

// Create a project via API route
async function createProjectViaAPI(projectData: any) {
  const response = await fetch('/api/projects', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(projectData)
  });

  const result = await response.json();
  return result;
}
```

### Server Component Example

```typescript
import { getProjectsWithUsers } from '@/lib/api';

export default async function ProjectsPage() {
  const projects = await getProjectsWithUsers();

  return (
    <div>
      <h1>Projects</h1>
      {projects.map(project => (
        <div key={project.id}>
          <h2>{project.title}</h2>
          <p>Owner: {project.user?.full_name}</p>
        </div>
      ))}
    </div>
  );
}
```

## Error Handling

Always wrap API calls in try-catch blocks:

```typescript
try {
  const project = await getProjectById(1);
  console.log(project);
} catch (error) {
  console.error('Error fetching project:', error);
  // Handle error appropriately
}
```

## TypeScript Types

All types are exported from `@/lib/types`:

```typescript
import type {
  User,
  Project,
  Task,
  Comment,
  Tag,
  ProjectWithUser,
  TaskWithDetails,
  ProjectStats
} from '@/lib/types';
```
