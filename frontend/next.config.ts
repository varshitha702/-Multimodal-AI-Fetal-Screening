import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The model page became the evidence locker; keep old links (README, slides) working.
  async redirects() {
    return [{ source: "/model", destination: "/evidence", permanent: false }];
  },
};

export default nextConfig;
