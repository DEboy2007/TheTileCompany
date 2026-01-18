export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50">
      <div className="max-w-7xl mx-auto px-6 py-20">
        <div className="space-y-6">
          <h1 className="text-5xl font-bold text-gray-900">Welcome to NexHacks</h1>
          <p className="text-xl text-gray-600 max-w-2xl">
            An AI-powered analytics platform for benchmarking, documentation, and real-time analysis.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-16">
          <div className="bg-white rounded-lg border border-gray-200 p-6 hover:shadow-lg transition-shadow">
            <h2 className="text-lg font-bold text-gray-900 mb-2">Benchmark</h2>
            <p className="text-gray-600 text-sm mb-4">
              Compare performance metrics and analyze system behavior.
            </p>
            <a href="/benchmark" className="text-blue-600 hover:text-blue-700 font-medium text-sm">
              Explore →
            </a>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-6 hover:shadow-lg transition-shadow">
            <h2 className="text-lg font-bold text-gray-900 mb-2">Docs</h2>
            <p className="text-gray-600 text-sm mb-4">
              Read comprehensive documentation and guides.
            </p>
            <a href="/docs" className="text-blue-600 hover:text-blue-700 font-medium text-sm">
              Learn →
            </a>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-6 hover:shadow-lg transition-shadow">
            <h2 className="text-lg font-bold text-gray-900 mb-2">Console</h2>
            <p className="text-gray-600 text-sm mb-4">
              Access the interactive console for real-time analysis.
            </p>
            <a href="/console" className="text-blue-600 hover:text-blue-700 font-medium text-sm">
              Open →
            </a>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-6 hover:shadow-lg transition-shadow">
            <h2 className="text-lg font-bold text-gray-900 mb-2">Home</h2>
            <p className="text-gray-600 text-sm mb-4">
              Start your journey with NexHacks today.
            </p>
            <span className="text-gray-400 text-sm">You are here</span>
          </div>
        </div>
      </div>
    </main>
  );
}
