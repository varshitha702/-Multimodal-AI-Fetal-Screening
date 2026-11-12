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

test("S1 home page renders the analyser and the disclaimer", async () => {
  const html = await (await fetch(BASE)).text();
  assert.match(html, /Fetal head circumference/);
  assert.match(html, /not a medical device/);
  assert.match(html, /Drop an ultrasound image here/);
  assert.match(html, /Pixel spacing \(mm \/ pixel\)/);
});

test("S2 model page renders", async () => {
  const res = await fetch(`${BASE}/model`);
  assert.equal(res.status, 200);
  assert.match(await res.text(), /held-out test split/);
});

test("S3 unknown route returns 404", async () => {
  const res = await fetch(`${BASE}/does-not-exist`);
  assert.equal(res.status, 404);
});
