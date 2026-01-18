'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState, useEffect } from 'react';

export default function Navigation() {
  const pathname = usePathname();
  const [isVisible, setIsVisible] = useState(true);
  const [lastScrollY, setLastScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const currentScrollY = window.scrollY;

      if (currentScrollY > lastScrollY && currentScrollY > 100) {
        // Scrolling down
        setIsVisible(false);
      } else {
        // Scrolling up
        setIsVisible(true);
      }

      setLastScrollY(currentScrollY);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [lastScrollY]);

  const links = [
    { href: '/', label: 'Home' },
    { href: '/benchmark', label: 'Benchmark' },
    { href: '/docs', label: 'Docs' },
    { href: '/console', label: 'Console' },
  ];

  return (
    <nav
      className="fixed top-0 left-0 right-0 z-50 bg-[#FAF9F5] border-b border-gray-200 transition-transform duration-300"
      style={{
        transform: isVisible ? 'translateY(0)' : 'translateY(-100%)',
        fontFamily: 'var(--font-mono)',
      }}
    >
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <Link
            href="/"
            className="text-sm font-semibold tracking-wide text-[var(--color-dark)]"
            style={{ fontFamily: 'var(--font-mono)' }}
          >
            THE TILE COMPANY
          </Link>
          <div className="flex gap-8">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={`text-sm font-medium transition-colors relative ${
                  pathname === link.href
                    ? 'text-[var(--color-dark)]'
                    : 'text-gray-600 hover:text-[var(--color-dark)]'
                }`}
                style={{ fontFamily: 'var(--font-mono)' }}
              >
                {link.label}
                {pathname === link.href && (
                  <span className="absolute bottom-0 left-0 right-0 h-px bg-[var(--color-dark)]"></span>
                )}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
}
