"""JobMatch AI: local search, persistent profiles and optional Groq tool calling."""
import json, os, sqlite3
from pathlib import Path
from typing import Literal
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB = os.getenv('JOBMATCH_DB', 'jobmatch.db')
JOBS = json.loads(Path(__file__).with_name('jobs.json').read_text())
app = FastAPI(title='JobMatch AI')

@app.get('/', include_in_schema=False)
def home():
    return FileResponse(Path(__file__).with_name('index.html'))

def connect():
    db = sqlite3.connect(DB)
    db.execute('CREATE TABLE IF NOT EXISTS profiles (user_id TEXT PRIMARY KEY, preferences TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, user_id TEXT, question TEXT, answer TEXT)')
    return db

class Search(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    location: str = ''
    limit: int = Field(default=3, ge=1, le=10)
class Profile(BaseModel):
    skills: str = Field(default='', max_length=1000)
    location: str = Field(default='', max_length=100)
class Chat(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=1000)

@app.put('/profiles/{user_id}')
def save_profile(user_id: str, profile: Profile):
    with connect() as db:
        db.execute('INSERT OR REPLACE INTO profiles VALUES (?,?)',(user_id,profile.model_dump_json()))
    return profile

def get_profile(user_id):
    with connect() as db:
        row = db.execute('SELECT preferences FROM profiles WHERE user_id=?',(user_id,)).fetchone()
    return Profile.model_validate_json(row[0]) if row else Profile()

def normalize(text):
    # Small explicit synonym map; offline mode is lexical, not a trained semantic encoder.
    for a,b in [('generative ai','llm'),('artificial intelligence','ai'),('back end','backend'),('postgresql','sql'),('machine learning','ml')]:
        text=text.lower().replace(a,b)
    return text.lower()

@app.post('/search')
def search(req: Search):
    jobs=[j for j in JOBS if not req.location or req.location.casefold() in j['location'].casefold()]
    if not jobs: return []
    docs=[normalize(j['title']+' '+j['skills']+' '+j['description']) for j in jobs]
    query=normalize(req.query)
    if os.getenv('EMBEDDING_MODE')=='semantic':
        from sentence_transformers import SentenceTransformer
        model=SentenceTransformer('all-MiniLM-L6-v2')
        vectors=model.encode(docs+[query],normalize_embeddings=True)
        scores=cosine_similarity(vectors[-1:],vectors[:-1])[0]
    else:
        vectors=TfidfVectorizer(ngram_range=(1,2)).fit_transform(docs+[query])
        scores=cosine_similarity(vectors[-1:],vectors[:-1])[0]
    return [dict(jobs[i],score=round(float(scores[i]),4)) for i in sorted(range(len(jobs)),key=lambda i:-scores[i])[:req.limit] if scores[i]>0]

TOOLS=[{'type':'function','function':{'name':'search_jobs','description':'Retrieve and rank matching sample jobs.','parameters':{'type':'object','properties':{'query':{'type':'string'},'location':{'type':'string'},'limit':{'type':'integer'}},'required':['query'],'additionalProperties':False}}}]

def llm(messages, tools=None):
    payload={'model':os.getenv('GROQ_MODEL','openai/gpt-oss-120b'),'messages':messages,'temperature':0}
    if tools: payload.update(tools=tools,tool_choice='auto')
    r=httpx.post('https://api.groq.com/openai/v1/chat/completions',headers={'Authorization':'Bearer '+os.environ['GROQ_API_KEY']},json=payload,timeout=25)
    r.raise_for_status()
    return r.json()['choices'][0]['message']

@app.post('/chat')
def chat(req: Chat):
    profile=get_profile(req.user_id)
    with connect() as db:
        past=db.execute('SELECT question,answer FROM history WHERE user_id=? ORDER BY id DESC LIMIT 3',(req.user_id,)).fetchall()
    matches=[]; mode='offline'; answer=''
    if os.getenv('GROQ_API_KEY'):
        messages=[{'role':'system','content':'You are a job matching agent. Use search_jobs before making recommendations. Only mention jobs returned by the tool. Sample listings are not live. User profile: '+profile.model_dump_json()}]
        for q,a in reversed(past): messages.extend([{'role':'user','content':q},{'role':'assistant','content':a}])
        messages.append({'role':'user','content':req.question})
        try:
            response=llm(messages,TOOLS)
            calls=response.get('tool_calls') or []
            if calls:
                messages.append({'role':'assistant','content':response.get('content'),'tool_calls':calls})
                for call in calls:
                    if call['function']['name']!='search_jobs': raise ValueError('Unknown tool')
                    args=Search.model_validate_json(call['function']['arguments'])
                    args.query+=' '+profile.skills; args.location=profile.location or args.location
                    matches=search(args)
                    messages.append({'role':'tool','tool_call_id':call['id'],'content':json.dumps(matches)})
                answer=llm(messages).get('content') or ''
                mode='llm_tool_call'
        except (httpx.HTTPError,ValueError,KeyError,IndexError,TypeError): pass
    if not answer:
        matches=search(Search(query=(req.question+' '+profile.skills)[:1000],location=profile.location))
        answer='Sample matches: '+', '.join(j['title'] for j in matches) if matches else 'No matching sample jobs.'
        mode='offline'
    with connect() as db: db.execute('INSERT INTO history(user_id,question,answer) VALUES (?,?,?)',(req.user_id,req.question,answer))
    return {'answer':answer,'jobs':matches,'mode':mode}
