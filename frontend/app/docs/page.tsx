'use client';

import { useState, useEffect } from 'react';
import CodeBlock from '@/components/CodeBlock';
import Breadcrumb from '@/components/Breadcrumb';

export default function Docs() {
  const [activeSection, setActiveSection] = useState('api-reference');

  // Detect which section is currently in view
  useEffect(() => {
    const handleScroll = () => {
      const sections = document.querySelectorAll('section[id]');
      const scrollPosition = window.scrollY + 150; // Offset for header
      const windowHeight = window.innerHeight;
      const documentHeight = document.documentElement.scrollHeight;

      // Check if we're near the bottom of the page
      if (window.scrollY + windowHeight >= documentHeight - 100) {
        // Highlight the last section when near bottom
        const lastSection = sections[sections.length - 1] as HTMLElement;
        if (lastSection) {
          setActiveSection(lastSection.id);
          return;
        }
      }

      // Normal scroll detection
      for (let i = sections.length - 1; i >= 0; i--) {
        const section = sections[i] as HTMLElement;
        if (section.offsetTop <= scrollPosition) {
          setActiveSection(section.id);
          break;
        }
      }
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Check initial position

    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const sections = [
    {
      title: 'API Reference',
      id: 'api-reference',
      content: (
        <div className="space-y-6">
          <div>
            <h2 className="text-2xl font-mono font-semibold text-[var(--color-dark)] mb-4">Endpoints</h2>
            <div className="space-y-3">
              <div className="flex items-center gap-3 bg-white border border-gray-200 rounded-lg px-4 py-3 overflow-x-auto">
                <span className="text-xs font-mono font-semibold bg-blue-600 text-white px-2 py-1 rounded whitespace-nowrap">GET</span>
                <code className="text-sm font-mono text-[var(--color-dark)]">http://localhost:5000/health</code>
              </div>
              <div className="flex items-center gap-3 bg-white border border-gray-200 rounded-lg px-4 py-3 overflow-x-auto">
                <span className="text-xs font-mono font-semibold bg-green-700 text-white px-2 py-1 rounded whitespace-nowrap">POST</span>
                <code className="text-sm font-mono text-[var(--color-dark)]">http://localhost:5000/compress</code>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-mono font-semibold text-[var(--color-dark)] mb-4">Health Check</h3>
            <p className="text-gray-600 mb-4">
              Check if the compression API is healthy and running.
            </p>
            <div className="space-y-3 mb-4">
              <div>
                <p className="text-gray-600 text-sm font-medium mb-2">Response:</p>
                <CodeBlock
                  code={`{
  "status": "healthy",
  "service": "tile-api",
  "version": "1.0.0"
}`}
                  language="json"
                  filename="health.json"
                />
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-mono font-semibold text-[var(--color-dark)] mb-4">Compression Request</h3>
            <p className="text-gray-600 mb-4">
              Compress an image using DINOv2 attention-guided seam carving. Images are automatically resized to 518×518 pixels for processing.
            </p>
            <div className="space-y-3 mb-4">
              <div>
                <p className="text-gray-600 text-sm font-medium mb-2">Request:</p>
                <CodeBlock
                  code={`{
  "image": "https://example.com/image.jpg",
  "reduction": 0.3,
  "threshold": 0.3
}`}
                  language="json"
                  filename="request.json"
                />
              </div>
              <div>
                <p className="text-gray-600 text-sm font-medium mb-2">Success Response (status: 0):</p>
                <CodeBlock
                  code={`{
  "status": 0,
  "reduction_pct": 30.5,
  "gray_overlay_base64": "iVBORw0KGgoAAAANS...",
  "compressed_image_base64": "iVBORw0KGgoAAAANS...",
  "stats": {
    "original_size": [640, 480],
    "compressed_size": [445, 333],
    "original_pixels": 307200,
    "compressed_pixels": 148185,
    "pixels_saved": 159015
  }
}`}
                  language="json"
                  filename="response.json"
                />
              </div>
              <div>
                <p className="text-gray-600 text-sm font-medium mb-2">Error Response (status: 1):</p>
                <CodeBlock
                  code={`{
  "status": 1,
  "message": "Error description"
}`}
                  language="json"
                  filename="error.json"
                />
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-mono font-semibold text-[var(--color-dark)] mb-4">Request Parameters</h3>
            <div className="space-y-3">
              <div className="border-l-4 border-[var(--color-dark)] pl-4 bg-white py-2">
                <h4 className="font-mono font-medium text-[var(--color-dark)]">image (string, required)</h4>
                <p className="text-gray-600 text-sm">URL or local file path to the image to compress.</p>
              </div>
              <div className="border-l-4 border-[var(--color-dark)] pl-4 bg-white py-2">
                <h4 className="font-mono font-medium text-[var(--color-dark)]">reduction (float, optional)</h4>
                <p className="text-gray-600 text-sm">Target reduction factor (0.0-1.0). Default: 0.3. Higher values = more aggressive compression.</p>
              </div>
              <div className="border-l-4 border-[var(--color-dark)] pl-4 bg-white py-2">
                <h4 className="font-mono font-medium text-[var(--color-dark)]">threshold (float, optional)</h4>
                <p className="text-gray-600 text-sm">Attention threshold (0.0-1.0). Default: 0.3. Controls which pixels are considered important.</p>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-mono font-semibold text-[var(--color-dark)] mb-4">Response Fields</h3>
            <div className="space-y-3">
              <div className="border-l-4 border-[var(--color-dark)] pl-4 bg-white py-2">
                <h4 className="font-mono font-medium text-[var(--color-dark)]">status (integer)</h4>
                <p className="text-gray-600 text-sm">0 for success, 1 for error</p>
              </div>
              <div className="border-l-4 border-[var(--color-dark)] pl-4 bg-white py-2">
                <h4 className="font-mono font-medium text-[var(--color-dark)]">gray_overlay_base64 (string)</h4>
                <p className="text-gray-600 text-sm">Base64 encoded attention visualization overlay (shows low-attention areas)</p>
              </div>
              <div className="border-l-4 border-[var(--color-dark)] pl-4 bg-white py-2">
                <h4 className="font-mono font-medium text-[var(--color-dark)]">compressed_image_base64 (string)</h4>
                <p className="text-gray-600 text-sm">Base64 encoded compressed image in PNG format</p>
              </div>
              <div className="border-l-4 border-[var(--color-dark)] pl-4 bg-white py-2">
                <h4 className="font-mono font-medium text-[var(--color-dark)]">stats (object)</h4>
                <p className="text-gray-600 text-sm">Compression statistics including original/compressed sizes, pixel counts, and pixels saved</p>
              </div>
            </div>
          </div>
        </div>
      )
    },
    {
      title: 'Python SDK',
      id: 'python-sdk',
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-mono font-semibold text-[var(--color-dark)] mb-4">Installation</h3>
            <p className="text-gray-600 mb-4">Install the Tile Python SDK via pip:</p>
            <CodeBlock
              code={`pip install tile-sdk`}
              language="bash"
              filename="terminal"
            />
          </div>

          <div>
            <h3 className="text-lg font-mono font-semibold text-[var(--color-dark)] mb-4">Quick Start</h3>
            <p className="text-gray-600 mb-4">Get up and running in minutes:</p>
            <CodeBlock
              code={`from python_sdk import TileClient
import base64

# Initialize client (default: http://localhost:5000)
client = TileClient()

try:
    # Compress an image
    result = client.compress_image(
        image_path="https://example.com/image.jpg",
        reduction=0.3,
        threshold=0.3
    )

    # Decode base64 responses
    compressed_data = base64.b64decode(result['compressed_image_base64'])
    overlay_data = base64.b64decode(result['gray_overlay_base64'])

    # Save compressed image
    with open('compressed.png', 'wb') as f:
        f.write(compressed_data)

    # Print statistics
    stats = result['stats']
    print(f"Original: {stats['original_size']}")
    print(f"Compressed: {stats['compressed_size']}")
    print(f"Pixels saved: {stats['pixels_saved']}")

finally:
    client.close()`}
              language="python"
              filename="quickstart.py"
            />
          </div>

          <div>
            <h3 className="text-lg font-mono font-semibold text-[var(--color-dark)] mb-4">Configuration</h3>
            <div className="space-y-3">
              <div className="border border-gray-200 rounded-lg p-4 bg-white">
                <h4 className="font-mono font-medium text-[var(--color-dark)] mb-2">Base URL</h4>
                <p className="text-gray-600 text-sm">Default: http://localhost:5000. Set custom server with TileClient(base_url="...")</p>
              </div>
              <div className="border border-gray-200 rounded-lg p-4 bg-white">
                <h4 className="font-mono font-medium text-[var(--color-dark)] mb-2">Timeout</h4>
                <p className="text-gray-600 text-sm">Default: 30 seconds. Configure with TileClient(timeout=60)</p>
              </div>
              <div className="border border-gray-200 rounded-lg p-4 bg-white">
                <h4 className="font-mono font-medium text-[var(--color-dark)] mb-2">Reduction Factor (0 - 1.0)</h4>
                <p className="text-gray-600 text-sm">Controls compression aggressiveness. Default: 0.3. Higher values = more aggressive seam removal.</p>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-mono font-semibold text-[var(--color-dark)] mb-4">Error Handling</h3>
            <CodeBlock
              code={`from tile_client import TileClient, TileClientError, TileAPIError

client = TileClient()

try:
    result = client.compress_image("image.jpg")
except TileAPIError as e:
    print(f"API Error: {e}")
except TileConnectionError as e:
    print(f"Connection Error: {e}")
except TileClientError as e:
    print(f"Client Error: {e}")`}
              language="python"
              filename="error_handling.py"
            />
          </div>
        </div>
      )
    },
    {
      title: 'Resources',
      id: 'resources',
      content: (
        <div className="space-y-4">
          <div className="border border-gray-200 rounded-lg p-6 bg-white hover:border-gray-300 transition-colors">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-lg font-mono font-semibold text-[var(--color-dark)] mb-2">GitHub Repository</h3>
                <p className="text-gray-600 text-sm mb-4">
                  Explore the source code, contribute, or report issues on GitHub. Find examples, integrations, and community contributions.
                </p>
              </div>
              <svg className="w-6 h-6 text-gray-400 flex-shrink-0 mt-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </div>
            <a
              href="https://github.com/nexhacks/tile-company"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center text-[var(--color-dark)] hover:text-gray-600 transition-colors"
            >
              View on GitHub →
            </a>
          </div>

          <div className="border border-gray-200 rounded-lg p-6 bg-white hover:border-gray-300 transition-colors">
            <h3 className="text-lg font-mono font-semibold text-[var(--color-dark)] mb-2">Research Paper</h3>
            <p className="text-gray-600 text-sm mb-4">
              Learn the science behind our compression algorithm. Read about attention map-based pruning, token reduction techniques, and performance benchmarks.
            </p>
            <a
              href="#"
              className="inline-flex items-center text-[var(--color-dark)] hover:text-gray-600 transition-colors"
            >
              Read the Paper →
            </a>
          </div>

          
        </div>
      )
    }
  ];

  return (
    <main className="min-h-screen bg-[#FAF9F5] font-mono">
      <div className="max-w-7xl mx-auto px-6 py-12">
        {/* Header with breadcrumb and button */}
        <div className="mb-8 flex items-center justify-between">
          <Breadcrumb items={[{ label: 'Home', href: '/' }, { label: 'Docs' }]} />
          <a
            href="#"
            className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all cursor-pointer px-4 py-2 bg-black text-[#FAF9F5] hover:bg-black/90"
          >
            Get started
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="w-4 h-4" aria-hidden="true">
              <path d="M5 12h14"></path>
              <path d="m12 5 7 7-7 7"></path>
            </svg>
          </a>
        </div>

        <div className="grid grid-cols-4 gap-8">
          {/* Sidebar */}
          <aside className="col-span-1">
            <nav className="sticky top-20 space-y-6">
              <div>
                <h3 className="text-xs font-semibold text-[var(--color-dark)] mb-3 uppercase tracking-wider">Documentation</h3>
                <div className="space-y-2">
                  {sections.map((section) => (
                    <a
                      key={section.id}
                      href={`#${section.id}`}
                      className={`flex items-center gap-2 text-sm transition-colors py-2 pl-3 -ml-3 border-l-2 ${
                        activeSection === section.id
                          ? 'text-[var(--color-dark)] border-[var(--color-dark)] font-medium'
                          : 'text-gray-600 hover:text-[var(--color-dark)] border-transparent hover:border-gray-300'
                      }`}
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      {section.title}
                    </a>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-xs font-semibold text-[var(--color-dark)] mb-3 uppercase tracking-wider">Resources</h3>
                <div className="space-y-2">
                  <a
                    href="https://github.com/nexhacks/tile-company"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-2 text-sm text-gray-600 hover:text-[var(--color-dark)] transition-colors py-2 pl-3 -ml-3"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                    Python SDK (GitHub)
                  </a>
                </div>
              </div>
            </nav>
          </aside>

          {/* Main Content */}
          <div className="col-span-3 space-y-12">
            {/* Top breadcrumb navigation */}
            <div className="flex items-center gap-2 text-sm text-gray-600 pb-4 border-b border-gray-200">
              <span>Documentation</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
              <span className="text-[var(--color-dark)]">{sections.find(s => s.id === activeSection)?.title}</span>
            </div>

            {sections.map((section) => (
              <section key={section.id} id={section.id} className="scroll-mt-20">
                <h2 className="text-3xl font-mono font-bold text-[var(--color-dark)] mb-4">{section.title}</h2>
                <p className="text-gray-600 mb-6">
                  {section.id === 'api-reference' && 'Compress text for optimized LLM inference. Reduce token usage, lower costs, and speed up your AI applications.'}
                  {section.id === 'python-sdk' && 'Install and use the TheTileCompany Python SDK to compress images programmatically.'}
                  {section.id === 'resources' && 'Find helpful resources, documentation, and community support.'}
                </p>
                <div className="bg-white rounded-lg border border-gray-200 p-6">
                  {section.content}
                </div>
              </section>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}
