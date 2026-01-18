export default function Benchmark() {
  return (
    <main className="min-h-screen bg-white">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="space-y-8">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 mb-2">Benchmark</h1>
            <p className="text-gray-600">
              Compare performance metrics and analyze system behavior across different configurations.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-gray-50 rounded-lg border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Performance Metrics</h2>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-600">Average Response Time</span>
                  <span className="font-mono text-gray-900">--</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Throughput</span>
                  <span className="font-mono text-gray-900">--</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Error Rate</span>
                  <span className="font-mono text-gray-900">--</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Memory Usage</span>
                  <span className="font-mono text-gray-900">--</span>
                </div>
              </div>
            </div>

            <div className="bg-gray-50 rounded-lg border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Comparison Results</h2>
              <p className="text-gray-600 text-sm">
                Run benchmarks to compare different configurations and see detailed performance analysis.
              </p>
              <button className="mt-6 w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors">
                Run Benchmark
              </button>
            </div>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
            <h3 className="font-semibold text-blue-900 mb-2">Coming Soon</h3>
            <p className="text-blue-800 text-sm">
              Benchmark functionality is being prepared. Check back soon for performance analysis tools.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}
