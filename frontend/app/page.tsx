'use client';

import { useRef, useState } from 'react';
import ImageUploadBox, { ImageUploadBoxHandle } from '@/components/ImageUploadBox';
import { ScrollLockStepper, StepData } from '@/components/ScrollLockStepper';

export default function Home() {
  const uploadBoxRef = useRef<ImageUploadBoxHandle>(null);
  const [image, setImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);

  const pipelineSteps: StepData[] = [
    {
      id: 'hero',
      label: 'Upload',
      title: 'Supercharge LLM performance by removing redundant pixels',
      content: (
        <p>
          Our intelligent pixel pruning technology identifies and removes irrelevant pixels, cutting inference costs and accelerating LLM performance.
        </p>
      ),
      customRender: (
        <div className="max-w-6xl mx-auto w-full grid grid-cols-1 md:grid-cols-2 gap-12 items-center px-6">
          <div className="space-y-8">
            {!processedImage ? (
              <div className="space-y-6">
                <h1 className="text-5xl text-[var(--color-dark)] mb-6 mt-8 first:mt-0 font-serif leading-tight">
                  Supercharge LLM performance by removing redundant pixels
                </h1>
                <p className="text-lg text-gray-600 leading-relaxed max-w-2xl">
                  Reduce image tokens by up to 95% while preserving semantic information. Our intelligent pixel pruning technology identifies and removes irrelevant pixels, cutting inference costs and accelerating LLM performance.
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                <h2 className="text-3xl text-[var(--color-dark)] font-serif">Results</h2>
                <div className="grid grid-cols-2 gap-8">
                  <div>
                    <p className="text-sm text-gray-600 mb-3 font-medium">Original</p>
                    <img src={image || ''} alt="Original" className="w-full rounded-lg border border-gray-300" />
                  </div>
                  <div>
                    <p className="text-sm text-gray-600 mb-3 font-medium">Optimized</p>
                    <img src={processedImage || ''} alt="Processed" className="w-full rounded-lg border border-gray-300" />
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 pt-4">
                  <div className="bg-white p-4 rounded-lg border border-gray-200">
                    <p className="text-2xl font-bold text-[var(--color-dark)]">95%</p>
                    <p className="text-sm text-gray-600">Token reduction</p>
                  </div>
                  <div className="bg-white p-4 rounded-lg border border-gray-200">
                    <p className="text-2xl font-bold text-[var(--color-dark)]">0%</p>
                    <p className="text-sm text-gray-600">Quality loss</p>
                  </div>
                  <div className="bg-white p-4 rounded-lg border border-gray-200">
                    <p className="text-2xl font-bold text-[var(--color-dark)]">10x</p>
                    <p className="text-sm text-gray-600">Faster tokens</p>
                  </div>
                </div>
              </div>
            )}

            {!processedImage && (
              <div>
                <button
                  onClick={() => image ? handleSubmit() : uploadBoxRef.current?.triggerUpload()}
                  data-slot="button"
                  className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all cursor-pointer disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 shrink-0 [&_svg]:shrink-0 outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive bg-black text-[#FAF9F5] hover:bg-black/90 h-9 has-[>svg]:px-3 px-8 py-2 font-mono"
                >
                  {image ? 'Submit' : 'Upload an image to get started'}
                  {!image && (
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-arrow-right ml-1 w-5 h-5" aria-hidden="true">
                      <path d="M5 12h14"></path>
                      <path d="m12 5 7 7-7 7"></path>
                    </svg>
                  )}
                </button>
              </div>
            )}
          </div>

          <ImageUploadBox
            ref={uploadBoxRef}
            image={processedImage || image}
            onImageChange={setImage}
          />
        </div>
      ),
    },
    {
      id: 'preprocessing',
      label: 'Step 1',
      title: 'Preprocessing & Attention Mapping',
      content: (
        <>
          <p>
            We standardize your image to 518×518 pixels, then use{' '}
            <span className="font-bold text-gray-800">DINOv2</span>, a Vision
            Transformer, to analyze visual importance. The model scans every
            pixel and assigns an <span className="font-bold text-gray-800">attention score</span>
            : brighter areas = critical for understanding, darker = irrelevant and redundant.
          </p>
          <p>
            This <span className="font-bold text-gray-800">attention map</span> becomes the foundation for the next step,
            providing the numerical basis to determine exactly which pixels contribute minimally to visual understanding.
          </p>
        </>
      ),
      image: '/attention_output.png',
      imageAlt: 'Attention map visualization: Bright = important pixels, Dark = irrelevant pixels',
    },
    {
      id: 'pruning',
      label: 'Step 2',
      title: 'Building the Pruning Mask',
      content: (
        <>
          <p>
            Using the attention scores from the previous step, we compute an <span className="font-bold text-gray-800">importance threshold</span> that separates critical pixels from redundant ones. Any pixel scoring below this threshold is marked for removal.
          </p>
          <p>
            We then construct a <span className="font-bold text-gray-800">pruning mask</span>—a precise binary map identifying every pixel marked for removal: background details, gradual transitions, and fine textures in non-critical regions. This mask ensures surgical precision, marking only pixels that genuinely don't contribute to LLM understanding before the final execution.
          </p>
        </>
      ),
      image: '@Image_Compression_Testing/Graphs/pruning_comparison.png',
      imageAlt: 'Pruning comparison visualization showing masked pixels for removal',
    },
    {
      id: 'carving',
      label: 'Step 3',
      title: 'Executing Content-Aware Removal',
      content: (
        <>
          <p>
            With the pruning mask in hand, we execute the final step: <span className="font-bold text-gray-800">content-aware pixel removal</span>. Using the mask as our guide, intelligent algorithms eliminate every marked pixel while maintaining structural coherence and avoiding artifacts at boundaries.
          </p>
          <p>
            The result: a dramatically optimized image—often <span className="font-bold text-gray-800">95% smaller in token count</span>—with complete semantic integrity preserved. Every remaining pixel has been validated as essential by the attention model, delivering your LLM a perfectly distilled visual representation, free of noise.
          </p>
        </>
      ),
      image: '@Image_Compression_Testing/Graphs/seam_carving_comparison.png',
      imageAlt: 'Seam carving comparison showing content-aware pixel removal results',
    },
  ];

  const handleSubmit = async () => {
    if (!image) return;

    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      setProcessedImage(image);
    } catch (error) {
      console.error('Error submitting image:', error);
    }
  };

  return (
    <main className="bg-[#FAF9F5]">
      {/* Pipeline Steps - Scroll Locked (includes hero as first step) */}
      <ScrollLockStepper steps={pipelineSteps} uploadBoxRef={uploadBoxRef} />

      {/* Navigation Links */}
      <section className="min-h-screen bg-white border-t border-gray-200 px-6 flex items-center justify-center">
        <div className="max-w-4xl mx-auto w-full">
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
