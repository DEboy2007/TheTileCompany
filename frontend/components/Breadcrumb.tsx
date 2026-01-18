'use client';

interface BreadcrumbProps {
  items: Array<{
    label: string;
    href?: string;
  }>;
}

export default function Breadcrumb({ items }: BreadcrumbProps) {
  return (
    <nav className="flex items-center space-x-2 text-sm">
      {items.map((item, index) => (
        <div key={index} className="flex items-center space-x-2">
          {index > 0 && (
            <span className="text-gray-600">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </span>
          )}
          {item.href ? (
            <a
              href={item.href}
              className="text-gray-600 hover:text-[var(--color-dark)] transition-colors"
            >
              {item.label}
            </a>
          ) : (
            <span className="text-[var(--color-dark)]">{item.label}</span>
          )}
        </div>
      ))}
    </nav>
  );
}
