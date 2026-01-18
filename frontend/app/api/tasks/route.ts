import { NextRequest, NextResponse } from 'next/server';
import { getTasks, getTasksWithDetails, createTask } from '@/lib/api';

// GET /api/tasks - Get all tasks
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const withDetails = searchParams.get('withDetails') === 'true';
    const status = searchParams.get('status');
    const projectId = searchParams.get('projectId');

    let tasks;

    if (withDetails) {
      tasks = await getTasksWithDetails();
    } else {
      tasks = await getTasks();
    }

    // Filter by status if provided
    if (status) {
      tasks = tasks.filter((task: any) => task.status === status);
    }

    // Filter by project if provided
    if (projectId) {
      tasks = tasks.filter((task: any) => task.project_id === parseInt(projectId));
    }

    return NextResponse.json({
      success: true,
      data: tasks
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to fetch tasks'
      },
      { status: 500 }
    );
  }
}

// POST /api/tasks - Create a new task
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const task = await createTask(body);

    return NextResponse.json({
      success: true,
      data: task
    }, { status: 201 });
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to create task'
      },
      { status: 500 }
    );
  }
}
