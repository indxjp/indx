/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static HTML/CSS/JS export — no Node at runtime. FastAPI serves out/ directly.
  output: 'export',
  images: { unoptimized: true },
  trailingSlash: true,
  // Dev-only proxy so `npm run dev` (Next on :3000) can reach the FastAPI backend
  // on :8000. In production the SPA is served same-origin by FastAPI, so this must
  // be a no-op (Next disallows rewrites() with output:'export' at build time anyway).
  async rewrites() {
    if (process.env.NODE_ENV === 'production') {
      return [];
    }
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/api/:path*',
      },
    ];
  },
};

export default nextConfig;
