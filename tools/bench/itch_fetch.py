#!/usr/bin/env python3
"""Download a free (no-payment) upload from an itch.io page, stdlib only.

    itch_fetch.py https://quaternius.itch.io/universal-animation-library "[Standard]" out.zip [sha256]

Follows the page's own "Download" flow (csrf token -> download page ->
file URL). Used for the CC0 Quaternius Universal Animation Library.
"""
import hashlib, http.cookiejar, json, re, sys, urllib.parse, urllib.request

page, want, out = sys.argv[1:4]
sha = sys.argv[4] if len(sys.argv) > 4 else None
cj = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
op.addheaders = [("User-Agent", "Mozilla/5.0 forge-bench")]
get = lambda u, d=None: op.open(u, urllib.parse.urlencode(d).encode() if d else None, timeout=60).read()
html = get(page).decode()
tok = re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)
dl = json.loads(get(page + "/download_url", {"csrf_token": tok}))["url"]
html = get(dl).decode()
tok = re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)
ups = re.findall(r'data-upload_id="(\d+)".*?class="name"[^>]*title="([^"]*)"', html, re.S) or \
      [(u, "") for u in re.findall(r'data-upload_id="(\d+)"', html)]
uid = next((u for u, n in ups if want in n), ups[0][0])
r = json.loads(get(f"{page}/file/{uid}?source=game_download", {"csrf_token": tok}))
if "url" not in r:
    sys.exit(f"itch said: {r}")
url = r["url"]
data = op.open(url, timeout=600).read()
h = hashlib.sha256(data).hexdigest()
if sha and h != sha:
    sys.exit(f"sha256 mismatch: {h}")
open(out, "wb").write(data)
print(out, len(data), h)
