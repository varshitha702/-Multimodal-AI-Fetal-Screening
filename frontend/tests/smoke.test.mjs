// Frontend smoke test: starts the production build (`next start`) and checks the pages render.
// Run after `npm run build`:  npm run test:smoke
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { after, before, test } from "node:test";

const PORT = 3123;
const BASE = `http://127.0.0.1:${PORT}`;
let server;

async function waitForServer(timeoutMs = 60000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(BASE);
      if (res.ok) return;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error("next start did not come up in time");
}

before(async () => {
  server = spawn(process.execPath, ["node_modules/next/dist/bin/next", "start", "-p", String(PORT)], {
    stdio: "ignore",
  });
  await waitForServer();
});

after(() => server?.kill());

test("S1 landing page renders the pitch, the hero scan and the disclaimer", async () => {
  const html = await (await fetch(BASE)).text();
  assert.match(html, /Measure only/);
  assert.match(html, /not a medical device/);
  assert.match(html, /Open the workstation/);
  assert.match(html, /What we do not claim/);
});

test("S2 workspace pages render", async () => {
  const pages = {
    "/scan": /Drop an ultrasound image here/,
    "/ctg": /Cardiotocography classification/,
    "/report": /One summary, every reason shown/,
    "/evidence": /Every claim, with its numbers/,
  };
  for (const [path, re] of Object.entries(pages)) {
    const res = await fetch(`${BASE}${path}`);
    assert.equal(res.status, 200, path);
    assert.match(await res.text(), re, path);
  }
  const scan = await (await fetch(`${BASE}/scan`)).text();
  assert.match(scan, /Pixel spacing \(mm \/ pixel\)/);
});

test("S3 old /model link redirects to /evidence; unknown route returns 404", async () => {
  const res = await fetch(`${BASE}/model`, { redirect: "manual" });
  assert.ok([307, 308].includes(res.status), String(res.status));
  assert.match(res.headers.get("location") ?? "", /\/evidence$/);
  assert.equal((await fetch(`${BASE}/does-not-exist`)).status, 404);
});
