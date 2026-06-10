import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'INDX',
  description: 'Turn a pile of files into an organized, AI-ready knowledge base you can explore and ask.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
