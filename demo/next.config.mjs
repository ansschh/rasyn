/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  // Environment variable for API base URL (used by lib/api.ts)
  env: {
    NEXT_PUBLIC_API_BASE: process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000',
  },
};

export default nextConfig;
