import { NextRequest, NextResponse } from 'next/server';
import { writeFile, unlink } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';

export async function POST(request: NextRequest) {
  let tempFilePath: string | null = null;

  try {
    const body = await request.json();
    const { image, reduction, threshold } = body;

    if (!image) {
      return NextResponse.json(
        { status: 1, message: 'No image provided' },
        { status: 400 }
      );
    }

    // Extract base64 data (remove data URL prefix if present)
    const base64Data = image.includes('base64,')
      ? image.split('base64,')[1]
      : image;

    // Create a temporary file with a short name to avoid path length issues
    const tempFileName = `img_${Date.now()}_${Math.random().toString(36).slice(2, 9)}.png`;
    tempFilePath = join(tmpdir(), tempFileName);

    // Convert base64 to buffer and write to temp file asynchronously
    const buffer = Buffer.from(base64Data, 'base64');
    await writeFile(tempFilePath, buffer);

    // Forward the request to the backend with the temp file path
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5001';

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout

    try {
      const response = await fetch(`${backendUrl}/compress`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: tempFilePath,
          reduction: reduction ?? 0.3,
          threshold: threshold ?? 0.3,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      const data = await response.json();

      if (!response.ok) {
        return NextResponse.json(data, { status: response.status });
      }

      return NextResponse.json(data);
    } catch (fetchError) {
      clearTimeout(timeoutId);

      if (fetchError instanceof Error && fetchError.name === 'AbortError') {
        return NextResponse.json(
          {
            status: 1,
            message: 'Request timeout - compression took too long'
          },
          { status: 504 }
        );
      }
      throw fetchError;
    }
  } catch (error) {
    console.error('Error proxying compress request:', error);
    return NextResponse.json(
      {
        status: 1,
        message: error instanceof Error ? error.message : 'Failed to compress image'
      },
      { status: 500 }
    );
  } finally {
    // Clean up the temporary file asynchronously
    if (tempFilePath) {
      try {
        await unlink(tempFilePath);
      } catch (cleanupError) {
        console.error('Error cleaning up temp file:', cleanupError);
      }
    }
  }
}
