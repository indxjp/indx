import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'indx · web tester',
  description: 'Configure, build, inspect, and query an INDX knowledge space.',
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
