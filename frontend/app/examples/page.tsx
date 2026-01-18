'use client';

import { useEffect, useState } from 'react';
import { getProjects, getTasks, getUsers, getProjectStats } from '@/lib/api';
import type { Project, Task, User, ProjectStats } from '@/lib/types';

export default function ExamplesPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [selectedProject, setSelectedProject] = useState<number | null>(null);
  const [projectStats, setProjectStats] = useState<ProjectStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [projectsData, tasksData, usersData] = await Promise.all([
          getProjects(),
          getTasks(),
          getUsers()
        ]);

        setProjects(projectsData);
        setTasks(tasksData);
        setUsers(usersData);
      } catch (err: any) {
        setError(err.message || 'Failed to load data');
        console.error('Error loading data:', err);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  useEffect(() => {
    async function loadStats() {
      if (selectedProject) {
        try {
          const stats = await getProjectStats(selectedProject);
          setProjectStats(stats);
        } catch (err) {
          console.error('Error loading stats:', err);
        }
      } else {
        setProjectStats(null);
      }
    }

    loadStats();
  }, [selectedProject]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-md">
          <h2 className="text-red-800 font-semibold mb-2">Error Loading Data</h2>
          <p className="text-red-600">{error}</p>
          <p className="text-sm text-red-500 mt-2">
            Make sure your Supabase credentials are configured and the database is running.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">API Examples</h1>
          <p className="text-gray-600">
            This page demonstrates using the Supabase API functions. Check the code in{' '}
            <code className="bg-gray-200 px-2 py-1 rounded text-sm">app/examples/page.tsx</code>
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Users Card */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Users</h2>
            <p className="text-3xl font-bold text-indigo-600 mb-2">{users.length}</p>
            <div className="space-y-2">
              {users.slice(0, 3).map(user => (
                <div key={user.id} className="text-sm text-gray-600">
                  {user.full_name || user.username}
                </div>
              ))}
            </div>
          </div>

          {/* Projects Card */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Projects</h2>
            <p className="text-3xl font-bold text-green-600 mb-2">{projects.length}</p>
            <div className="space-y-1">
              {['active', 'planning', 'completed'].map(status => {
                const count = projects.filter(p => p.status === status).length;
                return count > 0 ? (
                  <div key={status} className="text-sm text-gray-600">
                    {status}: {count}
                  </div>
                ) : null;
              })}
            </div>
          </div>

          {/* Tasks Card */}
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">Tasks</h2>
            <p className="text-3xl font-bold text-blue-600 mb-2">{tasks.length}</p>
            <div className="space-y-1">
              {['pending', 'in_progress', 'completed'].map(status => {
                const count = tasks.filter(t => t.status === status).length;
                return count > 0 ? (
                  <div key={status} className="text-sm text-gray-600">
                    {status.replace('_', ' ')}: {count}
                  </div>
                ) : null;
              })}
            </div>
          </div>
        </div>

        {/* Projects List */}
        <div className="bg-white rounded-lg shadow overflow-hidden mb-8">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-800">Projects</h2>
          </div>
          <div className="divide-y divide-gray-200">
            {projects.map(project => {
              const owner = users.find(u => u.id === project.user_id);
              const projectTasks = tasks.filter(t => t.project_id === project.id);
              const isSelected = selectedProject === project.id;

              return (
                <div key={project.id} className="p-6 hover:bg-gray-50 transition-colors">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="text-lg font-semibold text-gray-900">{project.title}</h3>
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          project.status === 'active' ? 'bg-green-100 text-green-700' :
                          project.status === 'completed' ? 'bg-blue-100 text-blue-700' :
                          'bg-gray-100 text-gray-700'
                        }`}>
                          {project.status}
                        </span>
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          project.priority === 'high' ? 'bg-red-100 text-red-700' :
                          project.priority === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                          'bg-gray-100 text-gray-700'
                        }`}>
                          {project.priority}
                        </span>
                      </div>
                      <p className="text-gray-600 mb-2">{project.description}</p>
                      <p className="text-sm text-gray-500">
                        Owner: {owner?.full_name || owner?.username || 'Unknown'} • {projectTasks.length} tasks
                      </p>
                    </div>
                    <button
                      onClick={() => setSelectedProject(isSelected ? null : project.id)}
                      className="ml-4 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors text-sm"
                    >
                      {isSelected ? 'Hide Stats' : 'View Stats'}
                    </button>
                  </div>

                  {isSelected && projectStats && (
                    <div className="mt-4 p-4 bg-indigo-50 rounded-lg">
                      <h4 className="font-semibold text-indigo-900 mb-3">Project Statistics</h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div>
                          <p className="text-sm text-indigo-600">Total Tasks</p>
                          <p className="text-2xl font-bold text-indigo-900">{projectStats.total_tasks}</p>
                        </div>
                        <div>
                          <p className="text-sm text-green-600">Completed</p>
                          <p className="text-2xl font-bold text-green-900">{projectStats.completed_tasks}</p>
                        </div>
                        <div>
                          <p className="text-sm text-yellow-600">In Progress</p>
                          <p className="text-2xl font-bold text-yellow-900">{projectStats.in_progress_tasks}</p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-600">Completion</p>
                          <p className="text-2xl font-bold text-gray-900">{projectStats.completion_percentage}%</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Recent Tasks */}
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-800">Recent Tasks</h2>
          </div>
          <div className="divide-y divide-gray-200">
            {tasks.slice(0, 10).map(task => {
              const project = projects.find(p => p.id === task.project_id);
              const assignedUser = task.assigned_to ? users.find(u => u.id === task.assigned_to) : null;

              return (
                <div key={task.id} className="p-4 hover:bg-gray-50 transition-colors">
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="font-medium text-gray-900">{task.title}</h3>
                      <p className="text-sm text-gray-500 mt-1">
                        {project?.title}
                        {assignedUser && ` • Assigned to ${assignedUser.full_name || assignedUser.username}`}
                      </p>
                    </div>
                    <span className={`px-2 py-1 text-xs rounded-full whitespace-nowrap ${
                      task.status === 'completed' ? 'bg-green-100 text-green-700' :
                      task.status === 'in_progress' ? 'bg-blue-100 text-blue-700' :
                      'bg-gray-100 text-gray-700'
                    }`}>
                      {task.status.replace('_', ' ')}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="mt-8 p-6 bg-blue-50 border border-blue-200 rounded-lg">
          <h3 className="font-semibold text-blue-900 mb-2">💡 API Usage</h3>
          <p className="text-blue-800 text-sm mb-2">
            This page uses the following API functions:
          </p>
          <ul className="text-sm text-blue-700 space-y-1 list-disc list-inside">
            <li><code className="bg-blue-100 px-1 rounded">getProjects()</code> - Fetch all projects</li>
            <li><code className="bg-blue-100 px-1 rounded">getTasks()</code> - Fetch all tasks</li>
            <li><code className="bg-blue-100 px-1 rounded">getUsers()</code> - Fetch all users</li>
            <li><code className="bg-blue-100 px-1 rounded">getProjectStats(id)</code> - Get project statistics</li>
          </ul>
          <p className="text-sm text-blue-600 mt-3">
            See <code className="bg-blue-100 px-1 rounded">API_USAGE.md</code> for complete documentation.
          </p>
        </div>
      </div>
    </div>
  );
}
