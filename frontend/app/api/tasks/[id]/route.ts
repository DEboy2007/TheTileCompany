import { NextRequest, NextResponse } from 'next/server';
import { getTaskById, getTaskWithDetails, updateTask, deleteTask, completeTask } from '@/lib/api';

// GET /api/tasks/:id - Get task by ID
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id: idStr } = await params;
    const id = parseInt(idStr);
    const searchParams = request.nextUrl.searchParams;
    const withDetails = searchParams.get('withDetails') === 'true';

    const task = withDetails ? await getTaskWithDetails(id) : await getTaskById(id);

    return NextResponse.json({
      success: true,
      data: task
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to fetch task'
      },
      { status: 500 }
    );
  }
}

// PATCH /api/tasks/:id - Update task
export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id: idStr } = await params;
    const id = parseInt(idStr);
    const body = await request.json();

    const task = await updateTask(id, body);

    return NextResponse.json({
      success: true,
      data: task
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to update task'
      },
      { status: 500 }
    );
  }
}

// DELETE /api/tasks/:id - Delete task
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id: idStr } = await params;
    const id = parseInt(idStr);

    await deleteTask(id);

    return NextResponse.json({
      success: true,
      message: 'Task deleted successfully'
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to delete task'
      },
      { status: 500 }
    );
  }
}
