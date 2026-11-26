"""Live India listings from public Greenhouse and Lever company job boards."""
import html, re, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
import httpx

BOARDS = [
    ('greenhouse','emergentlabsinc','Emergent Labs'), ('greenhouse','hackerrank','HackerRank'),
    ('greenhouse','enterpret','Enterpret'), ('greenhouse','nirmata','Nirmata'),
    ('greenhouse','gleanwork','Glean'), ('greenhouse','thena','Thena'),
    ('greenhouse','crashplan','CrashPlan'), ('lever','hevodata','Hevo Data'),
    ('lever','acceldata','Acceldata'), ('lever','saviynt','Saviynt'),
    ('lever','zimperium','Zimperium'), ('lever','smarsh','Smarsh'),
    ('lever','jumpcloud','JumpCloud'),
]
INDIA = re.compile(r'\b(india|bengaluru|bangalore|hyderabad|pune|gurugram|gurgaon|noida|mumbai|chennai|delhi)\b', re.I)
TTL = 3600
_cache = {'time': 0.0, 'jobs': [], 'status': {}}
_lock = Lock()

def clean(text):
    text = re.sub(r'<[^>]+>', ' ', html.unescape(text or ''))
    return ' '.join(html.unescape(text).split())[:3000]

def parse(kind, token, company, data):
    jobs = []
    for item in (data['jobs'] if kind == 'greenhouse' else data):
        if kind == 'greenhouse':
            title, location = item.get('title'), (item.get('location') or {}).get('name') or ''
            url, body = item.get('absolute_url'), item.get('content')
            date = str(item.get('updated_at') or '')[:10]
        else:
            cats = item.get('categories') or {}
            title = item.get('text')
            location = ', '.join(cats.get('allLocations') or [cats.get('location') or ''])
            url, body = item.get('applyUrl') or item.get('hostedUrl'), item.get('descriptionPlain')
            made = item.get('createdAt')
            date = datetime.fromtimestamp(made/1000, timezone.utc).date().isoformat() if made else ''
        if title and url and url.startswith('https://') and INDIA.search(location):
            jobs.append({'id': f'{token}-{item.get("id")}', 'title': title, 'company': company,
                         'location': location, 'skills': '', 'description': clean(body),
                         'apply_url': url, 'date': date})
    return jobs

def fetch(board):
    kind, token, company = board
    url = (f'https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true' if kind == 'greenhouse'
           else f'https://api.lever.co/v0/postings/{token}?mode=json')
    try:
        r = httpx.get(url, timeout=15)
        r.raise_for_status()
        return company, parse(kind, token, company, r.json()), None
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as e:
        return company, [], type(e).__name__

def load():
    """Return (jobs, per-company status), refreshed at most once an hour."""
    with _lock:
        if time.time() - _cache['time'] > TTL:
            with ThreadPoolExecutor(8) as pool:
                results = list(pool.map(fetch, BOARDS))
            jobs = [j for _, found, _ in results for j in found]
            _cache['status'] = {c: (error or len(found)) for c, found, error in results}
            if jobs:
                _cache['jobs'] = jobs
            # retry sooner when nothing came back
            _cache['time'] = time.time() if jobs else time.time() - TTL + 60
        return _cache['jobs'], _cache['status']
