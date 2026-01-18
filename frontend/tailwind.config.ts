import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        cream: 'var(--color-cream)',
        'warm-orange': 'var(--color-warm-orange)',
        'deep-orange': 'var(--color-deep-orange)',
        'accent-yellow': 'var(--color-accent-yellow)',
        dark: 'var(--color-dark)',
      },
      fontFamily: {
        serif: 'var(--font-serif)',
        body: 'var(--font-body)',
        mono: 'var(--font-mono)',
      },
      animation: {
        pulse: 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        fadeInUp: {
          '0%': {
            opacity: '0',
            transform: 'translateY(10px)',
          },
          '100%': {
            opacity: '1',
            transform: 'translateY(0)',
          },
        },
      },
    },
  },
  plugins: [],
};

export default config;
