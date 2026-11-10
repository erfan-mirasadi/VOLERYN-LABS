// Serves the static export for headless checks: node scripts/serve-out.mjs [port], after `npx next build`.
import { readFile, stat } from 'node:fs/promises';
import { createServer } from 'node:http';
import { extname, join } from 'node:path';

const root = new URL('../out', import.meta.url).pathname;
const port = Number(process.argv[2] ?? 3451);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.png': 'image/png', '.woff2': 'font/woff2' };

createServer(async (request, response) => {
  const path = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
  for (const candidate of [path, `${path}.html`, join(path, 'index.html')]) {
    const file = join(root, candidate);
    if (!(await stat(file).catch(() => null))?.isFile()) continue;
    response.writeHead(200, { 'content-type': types[extname(file)] ?? 'application/octet-stream' });
    response.end(await readFile(file));
    return;
  }
  response.writeHead(404).end('not found');
}).listen(port, '127.0.0.1', () => console.log(`serving out/ on http://127.0.0.1:${port}`));
