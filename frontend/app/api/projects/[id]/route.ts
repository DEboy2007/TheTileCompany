import { NextRequest, NextResponse } from 'next/server';
import { getProjectById, getProjectWithTags, updateProject, deleteProject, getProjectStats } from '@/lib/api';

// GET /api/projects/:id - Get project by ID
export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const id = parseInt(params.id);
    const searchParams = request.nextUrl.searchParams;
    const withTags = searchParams.get('withTags') === 'true';
    const withStats = searchParams.get('withStats') === 'true';

    const project = withTags ? await getProjectWithTags(id) : await getProjectById(id);

    if (withStats) {
      const stats = await getProjectStats(id);
      return NextResponse.json({
        success: true,
        data: { ...project, stats }
      });
    }

    return NextResponse.json({
      success: true,
      data: project
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to fetch project'
      },
      { status: 500 }
    );
  }
}

// PATCH /api/projects/:id - Update project
export async function PATCH(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const id = parseInt(params.id);
    const body = await request.json();

    const project = await updateProject(id, body);

    return NextResponse.json({
      success: true,
      data: project
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to update project'
      },
      { status: 500 }
    );
  }
}

// DELETE /api/projects/:id - Delete project
export async function DELETE(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const id = parseInt(params.id);

    await deleteProject(id);

    return NextResponse.json({
      success: true,
      message: 'Project deleted successfully'
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to delete project'
      },
      { status: 500 }
    );
  }
}
