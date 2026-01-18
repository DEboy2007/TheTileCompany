'use client';

import { useState, useEffect } from 'react';

export default function Docs() {
  const [activeSection, setActiveSection] = useState('getting-started');

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
      title: 'Getting Started', 
      id: 'getting-started',
      content: (
        <div className="space-y-4">
          <p className="text-gray-600 mb-4">
            Welcome to TheTileCompany! Our AI-powered compression platform optimizes images for better performance and cost efficiency.
          </p>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">What is TheTileCompany?</h3>
          <ul className="list-disc list-inside space-y-2 text-gray-600">
            <li>Reduce image tokens by up to 95% using attention maps</li>
            <li>Speed up AI inference and reduce API costs</li>
            <li>Works with any vision model without retraining</li>
            <li>Intelligent pixel pruning based on semantic importance</li>
          </ul>
        </div>
      )
    },
    { 
      title: 'Installation', 
      id: 'installation',
      content: (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Quick Setup</h3>
          <div className="bg-gray-900 rounded-lg p-4 text-white text-sm">
            <div className="mb-2"># Install dependencies</div>
            <div className="mb-2">npm install</div>
            <div className="mb-2"># Start development server</div>
            <div>npm run dev</div>
          </div>
        </div>
      )
    },
    { 
      title: 'Configuration', 
      id: 'configuration',
      content: (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Image Compression Settings</h3>
          <div className="space-y-3">
            <div className="border-l-4 border-blue-500 pl-4">
              <h4 className="font-medium text-gray-900">Reduction Factor (0 - 0.5)</h4>
              <p className="text-gray-600 text-sm">Controls how aggressive the pruning is. Higher values = more compression.</p>
            </div>
          </div>
        </div>
      )
    },
    { 
      title: 'API Reference', 
      id: 'api-reference',
      content: (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Image Compression Endpoint</h3>
          <div className="bg-gray-900 rounded-lg p-4 text-white text-sm">
            <div className="text-green-400 mb-2">POST /api/compress-image</div>
            <div className="text-gray-300">{"{"}</div>
            <div className="ml-4 text-blue-400">"image"<span className="text-white">: </span><span className="text-yellow-400">"path/to/image.jpg"</span>,</div>
            <div className="ml-4 text-blue-400">"threshold"<span className="text-white">: </span><span className="text-orange-400">0.3</span></div>
            <div className="text-gray-300">{"}"}</div>
          </div>
        </div>
      )
    },
    { 
      title: 'Examples', 
      id: 'examples',
      content: (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Basic Image Compression</h3>
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="bg-gray-900 rounded p-3 text-white text-sm mb-3">
              <div className="text-gray-400"># Python SDK Example</div>
              <div className="text-purple-400">from</div> <div className="text-white inline">tile_client</div> <div className="text-purple-400 inline">import</div> <div className="text-white inline">TileClient</div>
              <div className="mt-2">client = <div className="text-blue-400 inline">TileClient</div>()</div>
              <div>result = client.<div className="text-green-400 inline">compress_image</div>(</div>
              <div className="ml-4 text-yellow-400">"image.jpg"</div><div className="text-white inline">,</div>
              <div className="ml-4 text-yellow-400">"compressed.jpg"</div><div className="text-white inline">,</div>
              <div className="ml-4">threshold=<div className="text-orange-400 inline">0.3</div></div>
              <div>)</div>
            </div>
            <p className="text-gray-600 text-sm">This will compress your image by removing pixels with low attention scores.</p>
          </div>
        </div>
      )
    }
  ];

  return (
    <main className="min-h-screen bg-white">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-4 gap-8">
          {/* Sidebar */}
          <aside className="col-span-1">
            <nav className="sticky top-20 space-y-2">
              <h3 className="text-sm font-semibold text-gray-900 mb-4">Documentation</h3>
              {sections.map((section) => (
                <a
                  key={section.id}
                  href={`#${section.id}`}
                  className={`block text-sm transition-colors py-2 border-l-2 pl-3 -ml-3 ${
                    activeSection === section.id
                      ? 'text-blue-600 border-blue-600 bg-blue-50 font-medium'
                      : 'text-gray-600 hover:text-gray-900 border-transparent hover:border-gray-300'
                  }`}
                >
                  {section.title}
                </a>
              ))}
            </nav>
          </aside>

          {/* Main Content */}
          <div className="col-span-3 space-y-12">
            <div>
              <h1 className="text-4xl font-bold text-gray-900 mb-2">Documentation</h1>
              <p className="text-gray-600">
                Compress images through attention maps for optimized LLM inference. Create optimized images, reduce token usage, lower costs, and speed up your AI applications.
              </p>
            </div>

            {sections.map((section) => (
              <section key={section.id} id={section.id} className="scroll-mt-20">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">{section.title}</h2>
                <div className="bg-gray-50 rounded-lg border border-gray-200 p-6">
                  {section.content}
                </div>
              </section>
            ))}

            <section className="bg-blue-50 border border-blue-200 rounded-lg p-6">
              <h3 className="font-semibold text-blue-900 mb-2">Help & Support</h3>
              <p className="text-blue-800 text-sm">
                For additional help, visit our GitHub repository or contact the development team.
              </p>
            </section>
          </div>
        </div>
      </div>
    </main>
  );
}
