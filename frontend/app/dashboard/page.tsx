import { Suspense } from 'react';
import { DashboardContent } from '@/components/DashboardContent';

export const dynamic = 'force-dynamic';

export default function DashboardPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-gradient-to-br from-amber-50 via-white to-orange-50 flex items-center justify-center">
          <div className="text-gray-600">Loading dashboard...</div>
        </div>
      }
    >
      <DashboardContent />
    </Suspense>
  );
}
