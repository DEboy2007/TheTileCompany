export default function Docs() {
  const sections = [
    { title: 'Getting Started', id: 'getting-started' },
    { title: 'Installation', id: 'installation' },
    { title: 'Configuration', id: 'configuration' },
    { title: 'API Reference', id: 'api-reference' },
    { title: 'Examples', id: 'examples' },
    { title: 'Troubleshooting', id: 'troubleshooting' },
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
                  className="block text-sm text-gray-600 hover:text-gray-900 transition-colors py-2"
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
                Compress images through attention maps for optimized LLM inference. Reduce token usage, lower costs, and speed up your AI applications.
              </p>
            </div>

            {sections.map((section) => (
              <section key={section.id} id={section.id} className="scroll-mt-20">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">{section.title}</h2>
                <div className="bg-gray-50 rounded-lg border border-gray-200 p-6">
                  <p className="text-gray-600">
                    Documentation for {section.title.toLowerCase()} is being prepared.
                  </p>
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
