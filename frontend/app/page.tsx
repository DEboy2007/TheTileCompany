export default function Home() {
  return (
    <main className="min-h-screen bg-[#FAF9F5]">
      {/* Hero Section */}
      <section className="min-h-[80vh] flex items-center justify-center px-6 py-20">
        <div className="max-w-4xl mx-auto space-y-8">
          <div className="space-y-6">
            <h1 className="serif-display text-[var(--color-dark)]">
              Supercharge LLM performance by removing redundant pixels
            </h1>
            <p className="text-lg text-gray-600 leading-relaxed max-w-2xl">
              Reduce image tokens by up to 95% while preserving semantic information. Our intelligent pixel pruning technology identifies and removes irrelevant pixels, cutting inference costs and accelerating LLM performance.
            </p>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 pt-4">
            <button className="btn-primary">
              Start Free
            </button>
            <button className="btn-outline">
              Learn More
            </button>
          </div>

          {/* Key Stats */}
          <div className="grid grid-cols-3 gap-8 pt-12">
            <div>
              <p className="text-3xl font-bold text-[var(--color-dark)] mb-1">95%</p>
              <p className="text-sm text-gray-600">Token reduction</p>
            </div>
            <div>
              <p className="text-3xl font-bold text-[var(--color-dark)] mb-1">10x</p>
              <p className="text-sm text-gray-600">Faster inference</p>
            </div>
            <div>
              <p className="text-3xl font-bold text-[var(--color-dark)] mb-1">0%</p>
              <p className="text-sm text-gray-600">Quality loss</p>
            </div>
          </div>
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
