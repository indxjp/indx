import type { Config } from 'tailwindcss';

const config: Config = {
  // Theme is user-controlled: the store toggles a `dark` class on <html>, so dark
  // variants must follow that class rather than the OS `prefers-color-scheme`.
  darkMode: 'class',
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './lib/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // ink/panel/surface are CSS variables (defined in globals.css) so a single `.dark`
        // block re-themes every consumer; accent stays a literal (it reads well on both themes).
        ink: 'var(--color-ink)',
        panel: 'var(--color-panel)',
        surface: 'var(--color-surface)',
        accent: '#4f46e5',
        'accent-soft': '#eef2ff',
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
};

export default config;
