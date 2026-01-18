'use client';

import { useState } from 'react';

export default function Home() {
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);

  return (
    <main className="min-h-screen bg-[var(--color-cream)]">
      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* Gradient Orb Background */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-20 right-20 w-96 h-96 bg-[var(--color-warm-orange)] rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-pulse"></div>
          <div className="absolute -bottom-8 left-10 w-72 h-72 bg-[var(--color-accent-yellow)] rounded-full mix-blend-multiply filter blur-3xl opacity-15 animate-pulse" style={{ animationDelay: '2s' }}></div>
        </div>

        <div className="relative z-10 max-w-5xl mx-auto px-6 text-center space-y-8">
          <div className="space-y-6">
            <p className="section-label">Image Intelligence</p>
            <h1 className="serif-display text-[var(--color-dark)]">
              Every pixel <span className="text-[var(--color-warm-orange)]">that matters</span>
            </h1>
            <p className="text-lg md:text-xl text-[var(--color-text)] max-w-2xl mx-auto leading-relaxed">
              Reduce image tokens by up to 95% while preserving the visual information that matters. Our AI intelligently identifies and removes irrelevant pixels, cutting costs and speeding up inference.
            </p>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-8">
            <button className="btn-primary">
              Start Optimizing
            </button>
            <button className="btn-outline">
              Learn More
            </button>
          </div>

          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-16">
            <div className="bg-white/50 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
              <p className="text-3xl md:text-4xl font-bold text-[var(--color-warm-orange)] mb-2">95%</p>
              <p className="text-[var(--color-text)] font-medium">Token Reduction</p>
            </div>
            <div className="bg-white/50 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
              <p className="text-3xl md:text-4xl font-bold text-[var(--color-warm-orange)] mb-2">10x</p>
              <p className="text-[var(--color-text)] font-medium">Faster Processing</p>
            </div>
            <div className="bg-white/50 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
              <p className="text-3xl md:text-4xl font-bold text-[var(--color-warm-orange)] mb-2">0%</p>
              <p className="text-[var(--color-text)] font-medium">Quality Loss</p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-6 bg-white">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16 space-y-4">
            <p className="section-label">Capabilities</p>
            <h2 className="text-4xl md:text-5xl font-bold text-[var(--color-dark)]" style={{ fontFamily: 'var(--font-serif)' }}>
              Intelligent pixel pruning
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {[
              {
                id: 'smart-detection',
                title: 'Smart Pixel Detection',
                description: 'AI identifies which pixels contain semantic information and which are noise or background.',
                icon: '🎯'
              },
              {
                id: 'adaptive-compression',
                title: 'Adaptive Compression',
                description: 'Dynamically adjust compression levels based on image content and use case.',
                icon: '⚡'
              },
              {
                id: 'lossless-quality',
                title: 'Lossless Optimization',
                description: 'Preserve critical visual details while aggressively removing redundant information.',
                icon: '✓'
              },
              {
                id: 'api-integration',
                title: 'Easy Integration',
                description: 'Drop-in API that works with any vision model. No retraining required.',
                icon: '🔌'
              }
            ].map((feature) => (
              <div
                key={feature.id}
                onMouseEnter={() => setHoveredCard(feature.id)}
                onMouseLeave={() => setHoveredCard(null)}
                className="relative p-8 rounded-2xl border-2 border-[var(--color-cream)] bg-gradient-to-br from-white to-[var(--color-cream)] hover:border-[var(--color-warm-orange)] transition-all duration-300 cursor-pointer group"
              >
                {/* Accent corner */}
                {hoveredCard === feature.id && (
                  <div className="absolute top-0 right-0 w-20 h-20 bg-[var(--color-warm-orange)] rounded-bl-3xl opacity-10"></div>
                )}

                <div className="text-4xl mb-4">{feature.icon}</div>
                <h3 className="text-xl font-bold text-[var(--color-dark)] mb-3" style={{ fontFamily: 'var(--font-serif)' }}>
                  {feature.title}
                </h3>
                <p className="text-[var(--color-text)] leading-relaxed">
                  {feature.description}
                </p>

                {hoveredCard === feature.id && (
                  <div className="absolute bottom-4 right-4 text-[var(--color-warm-orange)] font-bold text-lg">
                    →
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto text-center space-y-8 bg-gradient-to-br from-[var(--color-warm-orange)]/10 via-[var(--color-accent-yellow)]/5 to-transparent rounded-3xl p-12">
          <h2 className="text-4xl md:text-5xl font-bold text-[var(--color-dark)]" style={{ fontFamily: 'var(--font-serif)' }}>
            Ready to optimize your image tokens?
          </h2>
          <p className="text-lg text-[var(--color-text)] max-w-2xl mx-auto">
            Join teams reducing inference costs and improving performance with intelligent pixel pruning.
          </p>
          <button className="btn-primary inline-block">
            Get Started Free
          </button>
        </div>
      </section>

      {/* Navigation Links */}
      <section className="py-12 px-6 border-t border-gray-200">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { href: '/benchmark', label: 'Benchmark', desc: 'Compare performance' },
              { href: '/docs', label: 'Documentation', desc: 'Learn the details' },
              { href: '/console', label: 'Console', desc: 'Try it now' }
            ].map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="group p-6 rounded-xl hover:bg-[var(--color-warm-orange)]/10 transition-colors"
              >
                <h3 className="font-bold text-[var(--color-dark)] mb-2 group-hover:text-[var(--color-warm-orange)] transition-colors" style={{ fontFamily: 'var(--font-serif)' }}>
                  {link.label}
                </h3>
                <p className="text-sm text-[var(--color-text)]">{link.desc}</p>
              </a>
            ))}
          </div>
        </div>
      </section>
    </main>
  );
}
