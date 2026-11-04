// pypi-2fa.mjs -- after the registration: verify the email (VERIFY_URL), enrol TOTP, save recovery codes, make an API token.
// Runs in a fresh context with the session of /out/pypi-state.json. Secrets go only to /out/pypi-secrets.env (0600), never to
// stdout. Env: PYPI_USER PYPI_PASS VERIFY_URL [STEP=all|verify|totp|token]
import { chromium } from 'playwright';
import { readFileSync, appendFileSync, chmodSync, writeFileSync, existsSync } from 'fs';
import crypto from 'crypto';
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const SEC = '/out/pypi-secrets.env';
const save = (k, v) => { appendFileSync(SEC, `${k}=${v}\n`); chmodSync(SEC, 0o600); };
const mask = s => s.replace(/pypi-[A-Za-z0-9_-]{20,}/g, 'pypi-‹…›').replace(/\b[A-Z2-7]{16,}\b/g, '‹key›').replace(/\b[0-9a-f]{16}\b/g, '‹code›');
function totp(b32, t = Date.now()) {
  const A = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567';
  let bits = '';
  for (const c of b32.replace(/[\s=]/g, '').toUpperCase()) bits += A.indexOf(c).toString(2).padStart(5, '0');
  const key = Buffer.from(bits.match(/.{8}/g).map(b => parseInt(b, 2)));
  const ctr = Buffer.alloc(8); ctr.writeBigUInt64BE(BigInt(Math.floor(t / 30000)));
  const h = crypto.createHmac('sha1', key).update(ctr).digest();
  const o = h[h.length - 1] & 15;
  return String((h.readUInt32BE(o) & 0x7fffffff) % 1000000).padStart(6, '0');
}
async function confirmPassword(p) {   // PyPI asks for the password again before sensitive pages
  if (await p.locator('input[name="password"]').count()) {
    await p.fill('input[name="password"]', process.env.PYPI_PASS);
    await Promise.all([p.waitForNavigation({ timeout: 60000 }).catch(() => null), p.locator('form:has(input[name="password"]) [type=submit]').first().click()]);
    log('password confirmed ->', p.url());
  }
}
const text = async p => (await p.locator('main').innerText().catch(() => '')).replace(/\s+/g, ' ');
const b = await chromium.connectOverCDP(process.env.CDP || 'http://127.0.0.1:9222');
const ctx = await b.newContext({ storageState: '/out/pypi-state.json', viewport: { width: 900, height: 1000 }, locale: 'en-US' });
const p = await ctx.newPage();
const step = process.env.STEP || 'all';
try {
  if (step === 'all' || step === 'verify') {
    await p.goto(process.env.VERIFY_URL, { waitUntil: 'load' }); await p.waitForTimeout(3000);
    await confirmPassword(p);
    log('verify ->', p.url(), mask((await text(p)).slice(0, 300)));
  }
  if (step === 'all' || step === 'totp') {
    await p.goto('https://pypi.org/manage/account/totp-provision', { waitUntil: 'load' }); await p.waitForTimeout(3000);
    await confirmPassword(p);
    const html = await p.content();
    const m = html.match(/secret=([A-Z2-7]{16,})/) || (await text(p)).match(/\b([A-Z2-7]{4}(?: ?[A-Z2-7]{4}){3,})\b/);
    if (!m) { log('TOTP page without a key:', mask((await text(p)).slice(0, 600))); throw 1; }
    const key = m[1].replace(/\s/g, '');
    save('PYPI_TOTP_SECRET', key);
    // a code from the first half of its 30-s window, so it is still valid when the form arrives
    while ((Date.now() / 1000) % 30 > 20) await p.waitForTimeout(1000);
    await p.fill('input[name="totp_value"]', totp(key));
    await Promise.all([p.waitForNavigation({ timeout: 60000 }).catch(() => null), p.locator('form:has(input[name="totp_value"]) [type=submit]').first().click()]);
    log('totp ->', p.url(), mask((await text(p)).slice(0, 300)));
    await p.goto('https://pypi.org/manage/account/recovery-codes/generate', { waitUntil: 'load' }).catch(() => null);
    await p.waitForTimeout(2000); await confirmPassword(p);
    const codes = (await text(p)).match(/\b[0-9a-f]{16}\b/g) || [];
    if (codes.length) save('PYPI_RECOVERY', codes.join(','));
    log('recovery codes:', codes.length, '->', p.url());
  }
  if (step === 'all' || step === 'token') {
    await p.goto('https://pypi.org/manage/account/token/', { waitUntil: 'load' }); await p.waitForTimeout(2000);
    await confirmPassword(p);
    await p.fill('input[name="description"]', 'ozaki-blackwell upload (04-11-2026)');
    const sel = p.locator('select[name="token_scope"]');
    if (await sel.count()) await sel.selectOption({ index: 1 }).catch(() => null);
    await Promise.all([p.waitForNavigation({ timeout: 60000 }).catch(() => null), p.locator('form:has(input[name="description"]) [type=submit]').first().click()]);
    const tok = (await p.content()).match(/pypi-[A-Za-z0-9_-]{50,}/);
    if (tok) { save('PYPI_TOKEN', tok[0]); log('token saved'); } else log('no token on', p.url(), mask((await text(p)).slice(0, 500)));
  }
  writeFileSync('/out/pypi-state.json', JSON.stringify(await ctx.storageState())); chmodSync('/out/pypi-state.json', 0o600);
} catch (e) { if (e !== 1) log('PERJASHTIM', mask(String(e)).slice(0, 300)); process.exitCode = 2; }
finally { await ctx.close().catch(() => {}); await b.close().catch(() => {}); }
