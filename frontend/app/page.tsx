'use client';

import { useRef } from 'react';
import ImageUploadBox, { ImageUploadBoxHandle } from '@/components/ImageUploadBox';

export default function Home() {
  const uploadBoxRef = useRef<ImageUploadBoxHandle>(null);
  return (
    <main className="bg-[#FAF9F5]">
      {/* Hero Section */}
      <section className="min-h-screen flex items-center justify-center px-6 py-20 -mt-16 pt-16">
        <div className="max-w-6xl mx-auto w-full grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
          <div className="space-y-8">
            <div className="space-y-6">
              <h1 className="text-5xl text-[var(--color-dark)] mb-6 mt-8 first:mt-0 font-serif">
                Supercharge LLM performance by removing redundant pixels
              </h1>
              <p className="text-lg text-gray-600 leading-relaxed max-w-2xl">
                Reduce image tokens by up to 95% while preserving semantic information. Our intelligent pixel pruning technology identifies and removes irrelevant pixels, cutting inference costs and accelerating LLM performance.
              </p>
            </div>

            {/* CTA Button */}
            <div>
              <button onClick={() => uploadBoxRef.current?.triggerUpload()} data-slot="button" className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all cursor-pointer disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive bg-black text-[#FAF9F5] hover:bg-black/90 h-9 has-[>svg]:px-3 px-8 py-2 font-mono">
                Upload an image to get started
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-arrow-right ml-1 w-5 h-5" aria-hidden="true">
                  <path d="M5 12h14"></path>
                  <path d="m12 5 7 7-7 7"></path>
                </svg>
              </button>
            </div>
          </div>

          {/* Upload Box */}
          <ImageUploadBox ref={uploadBoxRef} />
        </div>
      </section>

      {/* Features Section */}
      <section className="bg-white py-20 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="mb-16 space-y-4">
            <p className="section-label">How It Works</p>
            <h2 className="text-3xl md:text-4xl font-bold text-[var(--color-dark)]">
              Intelligent pixel pruning for modern AI
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {[
              {
                title: 'Smart Detection',
                description: 'AI analyzes each image to identify which pixels contain critical semantic information and which are redundant.',
              },
              {
                title: 'Adaptive Compression',
                description: 'Dynamically adjust compression levels based on image content and your specific use case.',
              },
              {
                title: 'Zero Quality Loss',
                description: 'Preserve all visually relevant details while aggressively removing noise and background.',
              },
              {
                title: 'Simple Integration',
                description: 'Drop-in API that works with any vision model. No retraining or fine-tuning required.',
              }
            ].map((feature, idx) => (
              <div key={idx} className="bg-white p-8 border border-gray-200">
                <h3 className="text-lg font-bold text-[var(--color-dark)] mb-3">
                  {feature.title}
                </h3>
                <p className="text-gray-600 leading-relaxed">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-[#FAF9F5] py-20 px-6">
        <div className="max-w-3xl mx-auto text-center space-y-8">
          <h2 className="text-3xl md:text-4xl font-bold text-[var(--color-dark)]">
            Ready to optimize your image tokens?
          </h2>
          <p className="text-lg text-gray-600">
            Join teams worldwide reducing inference costs and improving LLM performance.
          </p>
          <button className="btn-primary inline-block">
            Get Started Free
          </button>
        </div>
      </section>

      {/* Navigation Links */}
      <section className="bg-white border-t border-gray-200 py-12 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { href: '/benchmark', label: 'Benchmark', desc: 'Performance comparison' },
              { href: '/docs', label: 'Documentation', desc: 'Technical details' },
              { href: '/console', label: 'Console', desc: 'Try it now' }
            ].map((link) => (
              <a key={link.href} href={link.href} className="group">
                <h3 className="font-bold text-[var(--color-dark)] mb-1 group-hover:text-gray-600 transition-colors">
                  {link.label}
                </h3>
                <p className="text-sm text-gray-600">{link.desc}</p>
              </a>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
