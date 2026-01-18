import { NextRequest, NextResponse } from 'next/server';
import { getProjects, getProjectsWithUsers, createProject } from '@/lib/api';

// GET /api/projects - Get all projects
export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const withUsers = searchParams.get('withUsers') === 'true';

    const projects = withUsers ? await getProjectsWithUsers() : await getProjects();

    return NextResponse.json({
      success: true,
      data: projects
    });
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to fetch projects'
      },
      { status: 500 }
    );
  }
}

// POST /api/projects - Create a new project
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    const project = await createProject(body);

    return NextResponse.json({
      success: true,
      data: project
    }, { status: 201 });
  } catch (error: any) {
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to create project'
      },
      { status: 500 }
    );
  }
}
