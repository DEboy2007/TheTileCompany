'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navigation() {
  const pathname = usePathname();

  const links = [
    { href: '/', label: 'Home' },
    { href: '/benchmark', label: 'Benchmark' },
    { href: '/docs', label: 'Docs' },
    { href: '/console', label: 'Console' },
  ];

  return (
    <nav className="sticky top-0 z-50 bg-[var(--color-cream)]/95 backdrop-blur-md border-b border-gray-200/50">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <Link href="/" className="text-2xl font-bold text-[var(--color-dark)]" style={{ fontFamily: 'var(--font-serif)' }}>
            Pixel
          </Link>
          <div className="flex gap-8">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`font-medium transition-colors relative ${
                  pathname === link.href
                    ? 'text-[var(--color-warm-orange)]'
                    : 'text-[var(--color-text)] hover:text-[var(--color-warm-orange)]'
                }`}
              >
                {link.label}
                {pathname === link.href && (
                  <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-[var(--color-warm-orange)] rounded-full"></span>
                )}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
