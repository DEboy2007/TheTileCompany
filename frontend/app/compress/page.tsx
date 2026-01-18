'use client';

import { useState, useRef } from 'react';
import ImageUploadBox, { ImageUploadBoxHandle } from '@/components/ImageUploadBox';

/**
 * Response structure from the backend compression API
 * Matches the Flask API endpoint: POST /compress
 */
interface CompressionResult {
  status: number; // 0 = success, 1 = error
  reduction_pct: number; // Actual percentage of pixels saved
  gray_overlay_base64: string; // Base64 encoded gray overlay image (attention visualization)
  compressed_image_base64: string; // Base64 encoded compressed image
  stats: {
    original_size: [number, number]; // [width, height] of original image
    compressed_size: [number, number]; // [width, height] of compressed image
    original_pixels: number; // Total pixels in original
    compressed_pixels: number; // Total pixels in compressed
    pixels_saved: number; // Difference in pixel count
  };
}

export default function CompressPage() {
  const uploadBoxRef = useRef<ImageUploadBoxHandle>(null);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  // Backend defaults: reduction=0.3 (30% pixel reduction), threshold=0.3 (attention threshold)
  const [reduction, setReduction] = useState<number>(0.3);
  const [threshold, setThreshold] = useState<number>(0.3);
  const [isCompressing, setIsCompressing] = useState(false);
  const [result, setResult] = useState<CompressionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = (base64Image: string) => {
    setUploadedImage(base64Image);
    setResult(null);
    setError(null);
  };

  const handleCompress = async () => {
    if (!uploadedImage) {
      setError('Please upload an image first');
      return;
    }

    // Validate parameters match backend expectations
    if (reduction < 0 || reduction > 1) {
      setError('Reduction must be between 0 and 1');
      return;
    }

    if (threshold < 0 || threshold > 1) {
      setError('Threshold must be between 0 and 1');
      return;
    }

    setIsCompressing(true);
    setError(null);
    setResult(null);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000); // 2 minute timeout

    try {
      // Extract base64 data without the data URL prefix
      const base64Data = uploadedImage.includes('base64,')
        ? uploadedImage.split('base64,')[1]
        : uploadedImage;

      // Send request to Next.js API route (proxies to Flask backend at /compress)
      // Backend expects: { image: string, reduction?: number, threshold?: number }
      const response = await fetch('/api/compress', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: base64Data,
          reduction, // Target pixel reduction (0-1)
          threshold, // Attention threshold for gray overlay (0-1)
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        let errorMessage = `Server error (${response.status})`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.message || errorMessage;
        } catch {
          const errorText = await response.text();
          errorMessage = errorText || errorMessage;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();

      // Backend returns status: 0 for success, 1 for error
      if (data.status === 0) {
        // Validate response structure matches backend API
        if (!data.compressed_image_base64 || !data.stats) {
          throw new Error('Invalid response structure from server');
        }
        setResult(data);
      } else {
        setError(data.message || 'Compression failed');
      }
    } catch (err) {
      clearTimeout(timeoutId);

      if (err instanceof Error) {
        if (err.name === 'AbortError') {
          setError('Request timeout - compression took too long. Try a smaller image or lower reduction value.');
        } else {
          setError(`Error: ${err.message}`);
        }
      } else {
        setError('An unknown error occurred');
      }
    } finally {
      setIsCompressing(false);
    }
  };

  const handleDownload = () => {
    if (!result) return;

    const link = document.createElement('a');
    link.href = `data:image/png;base64,${result.compressed_image_base64}`;
    link.download = `compressed_${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <main className="bg-[#FAF9F5] min-h-screen">
      <section className="px-6 py-20">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-12 text-center">
            <h1 className="text-5xl text-[var(--color-dark)] mb-4 font-serif">
              Image Compression
            </h1>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Upload an image and compress it using attention-guided seam carving
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Left Column - Upload & Controls */}
            <div className="space-y-6">
              {/* Upload Box */}
              <div className="bg-white p-6 border border-gray-200 rounded-lg">
                <h2 className="text-xl font-bold text-[var(--color-dark)] mb-4">
                  Upload Image
                </h2>
                <ImageUploadBox
                  ref={uploadBoxRef}
                  onImageUpload={handleImageUpload}
                />
              </div>

              {/* Compression Controls */}
              {uploadedImage && (
                <div className="bg-white p-6 border border-gray-200 rounded-lg">
                  <h2 className="text-xl font-bold text-[var(--color-dark)] mb-4">
                    Compression Settings
                  </h2>

                  <div className="space-y-6">
                    {/* Reduction Slider */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Compression Reduction: {reduction.toFixed(2)}
                      </label>
                      <input
                        type="range"
                        min="0.1"
                        max="0.5"
                        step="0.05"
                        value={reduction}
                        onChange={(e) => setReduction(parseFloat(e.target.value))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-black"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Target pixel reduction ratio
                      </p>
                    </div>

                    {/* Threshold Slider */}
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Attention Threshold: {threshold.toFixed(2)}
                      </label>
                      <input
                        type="range"
                        min="0.1"
                        max="0.9"
                        step="0.1"
                        value={threshold}
                        onChange={(e) => setThreshold(parseFloat(e.target.value))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-black"
                      />
                      <p className="text-xs text-gray-500 mt-1">
                        Threshold for gray overlay visualization
                      </p>
                    </div>

                    {/* Compress Button */}
                    <button
                      onClick={handleCompress}
                      disabled={isCompressing}
                      className="w-full inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all cursor-pointer disabled:pointer-events-none disabled:opacity-50 outline-none bg-black text-[#FAF9F5] hover:bg-black/90 h-10 px-8 py-2 font-mono"
                    >
                      {isCompressing ? (
                        <>
                          <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          Compressing...
                        </>
                      ) : (
                        'Compress Image'
                      )}
                    </button>
                  </div>
                </div>
              )}

              {/* Error Display */}
              {error && (
                <div className="bg-red-50 border border-red-200 p-4 rounded-lg">
                  <p className="text-red-800 text-sm">{error}</p>
                </div>
              )}
            </div>

            {/* Right Column - Results */}
            <div className="space-y-6">
              {result && (
                <>
                  {/* Statistics */}
                  <div className="bg-white p-6 border border-gray-200 rounded-lg">
                    <h2 className="text-xl font-bold text-[var(--color-dark)] mb-4">
                      Compression Results
                    </h2>
                    <div className="space-y-3">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Reduction:</span>
                        <span className="font-mono font-bold text-[var(--color-dark)]">
                          {result.reduction_pct.toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Original Size:</span>
                        <span className="font-mono">
                          {result.stats.original_size[0]} x {result.stats.original_size[1]}
                        </span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Compressed Size:</span>
                        <span className="font-mono">
                          {result.stats.compressed_size[0]} x {result.stats.compressed_size[1]}
                        </span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">Pixels Saved:</span>
                        <span className="font-mono">
                          {result.stats.pixels_saved.toLocaleString()}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Compressed Image */}
                  <div className="bg-white p-6 border border-gray-200 rounded-lg">
                    <div className="flex justify-between items-center mb-4">
                      <h2 className="text-xl font-bold text-[var(--color-dark)]">
                        Compressed Image
                      </h2>
                      <button
                        onClick={handleDownload}
                        className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-black rounded-md hover:bg-black/90 transition-colors"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                          <polyline points="7 10 12 15 17 10"></polyline>
                          <line x1="12" y1="15" x2="12" y2="3"></line>
                        </svg>
                        Download
                      </button>
                    </div>
                    <div className="border border-gray-200 rounded-lg overflow-hidden">
                      <img
                        src={`data:image/png;base64,${result.compressed_image_base64}`}
                        alt="Compressed"
                        className="w-full h-auto"
                      />
                    </div>
                  </div>

                </>
              )}

              {/* Placeholder when no results */}
              {!result && uploadedImage && (
                <div className="bg-white p-12 border border-gray-200 rounded-lg text-center">
                  <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="mx-auto mb-4 text-gray-400">
                    <rect width="18" height="18" x="3" y="3" rx="2" ry="2"></rect>
                    <circle cx="9" cy="9" r="2"></circle>
                    <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"></path>
                  </svg>
                  <p className="text-gray-500">
                    Adjust settings and click &quot;Compress Image&quot; to see results
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
