// pypi-reg.mjs -- PyPI registration in a fresh context of the shared Chrome (CDP 9222, display :99 behind noVNC).
// Fills every field, leaves the hCaptcha to a human over noVNC, submits once when its token appears, prints the outcome.
// Env: PYPI_USER PYPI_EMAIL PYPI_PASS EMRI [PRIT_MIN=60]. Session cookies -> /out/pypi-state.json (for the 2FA step).
import { chromium } from 'playwright';
import { writeFileSync, chmodSync } from 'fs';
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);
const b = await chromium.connectOverCDP(process.env.CDP || 'http://127.0.0.1:9222');
const ctx = await b.newContext({ viewport: { width: 900, height: 1000 }, locale: 'en-US' });
const p = await ctx.newPage();
try {
  const u = process.env.PYPI_USER;
  const r = await p.goto(`https://pypi.org/user/${u}/`, { waitUntil: 'load' });
  await p.waitForTimeout(4000); // the Fastly client challenge reloads the page once
  const title = await p.title();
  log('user page', u, r?.status(), JSON.stringify(title));
  if (!/not found|404/i.test(title)) { log('EMRI I ZENE ose i papercaktuar -> ndal'); process.exitCode = 4; throw 0; }
  await p.goto('https://pypi.org/account/register/', { waitUntil: 'load' });
  await p.waitForSelector('input[name="username"]', { timeout: 60000 });
  await p.fill('input[name="full_name"]', process.env.EMRI);
  await p.fill('input[name="username"]', u);
  await p.fill('input[name="email"]', process.env.PYPI_EMAIL);
  await p.fill('input[name="new_password"]', process.env.PYPI_PASS);
  await p.fill('input[name="password_confirm"]', process.env.PYPI_PASS);
  await p.check('input[name="acceptable_use"]');
  await p.bringToFront();
  await p.locator('.h-captcha').scrollIntoViewIfNeeded();
  log('GATI: forma e mbushur, pritet CAPTCHA');
  const end = Date.now() + Number(process.env.PRIT_MIN || 60) * 60000;
  let tok = '';
  while (Date.now() < end && !tok) {
    tok = await p.evaluate(() => document.querySelector('[name="h-captcha-response"]')?.value || '').catch(() => '');
    if (!tok) await p.waitForTimeout(2000);
  }
  if (!tok) { log('SKADOI: CAPTCHA s\'u zgjidh'); process.exitCode = 3; throw 0; }
  log('CAPTCHA U ZGJIDH -> dergohet');
  const form = p.locator('form:has(input[name="username"])');
  await Promise.all([p.waitForNavigation({ timeout: 60000 }).catch(() => null), form.locator('[type=submit]').first().click()]);
  await p.waitForTimeout(3000);
  log('URL', p.url());
  const errs = (await p.locator('.form-error, .form-errors li, .callout-block--danger').allInnerTexts()).map(s => s.trim()).filter(Boolean);
  log('GABIME', JSON.stringify(errs));
  log('FAQJA', (await p.locator('main').innerText().catch(() => '')).replace(/\s+/g, ' ').slice(0, 400));
  const st = await ctx.storageState();
  writeFileSync('/out/pypi-state.json', JSON.stringify(st)); chmodSync('/out/pypi-state.json', 0o600);
  log(errs.length ? 'PERFUNDIMI: GABIM' : 'PERFUNDIMI: OK');
} catch (e) { if (e !== 0) { log('PERJASHTIM', String(e).slice(0, 300)); process.exitCode = 2; } }
finally { await ctx.close().catch(() => {}); await b.close().catch(() => {}); }
