import app
from fastapi.testclient import TestClient
import json
import live
import pytest

@pytest.fixture(autouse=True)
def sample_data(monkeypatch):
    monkeypatch.setenv('JOBS_SOURCE','sample')


def test_retrieval():
    assert app.search(app.Search(query='Python FastAPI RAG',location='Hyderabad'))[0]['id']=='j1'
    assert app.search(app.Search(query='React CSS'))[0]['id']=='j2'
    assert app.search(app.Search(query='Arduino sensors'))[0]['id']=='j5'
    assert app.search(app.Search(query='Python',location='Nowhere'))==[]

def test_memory_and_api(tmp_path,monkeypatch):
    monkeypatch.setattr(app,'DB',str(tmp_path/'test.db'))
    monkeypatch.delenv('GROQ_API_KEY',raising=False)
    c=TestClient(app.app)
    assert c.put('/profiles/u1',json={'skills':'Python FastAPI RAG','location':'Hyderabad'}).status_code==200
    r=c.post('/chat',json={'user_id':'u1','question':'Find an internship'}).json()
    assert r['jobs'][0]['id']=='j1' and r['mode']=='offline'
    with app.connect() as db: assert db.execute('SELECT COUNT(*) FROM history').fetchone()[0]==1

def test_tool_call_contract(tmp_path,monkeypatch):
    monkeypatch.setattr(app,'DB',str(tmp_path/'test.db'))
    monkeypatch.setenv('GROQ_API_KEY','mock-key')
    def fake(messages,tools=None):
        if tools: return {'content':None,'tool_calls':[{'id':'call1','type':'function','function':{'name':'search_jobs','arguments':json.dumps({'query':'Python RAG','location':'Hyderabad'})}}]}
        assert json.loads(messages[-1]['content'])[0]['id']=='j1'
        return {'content':'GenAI Backend Intern matches your skills.'}
    monkeypatch.setattr(app,'llm',fake)
    c=TestClient(app.app)
    r=c.post('/chat',json={'user_id':'u1','question':'Find RAG jobs'}).json()
    assert r['mode']=='llm_tool_call' and r['jobs'][0]['id']=='j1'

def test_home_page():
    r=TestClient(app.app).get('/')
    assert r.status_code==200 and 'JobMatch AI' in r.text

def test_live_parsing_and_search(monkeypatch):
    gh={'jobs':[{'id':1,'title':'Software Engineer Intern','location':{'name':'Bangalore'},'absolute_url':'https://example.com/1','content':'&lt;p&gt;Python APIs&lt;/p&gt;','updated_at':'2026-11-01T00:00:00Z'},
                {'id':2,'title':'Engineer','location':{'name':'London'},'absolute_url':'https://example.com/2'}]}
    lv=[{'id':'a','text':'Senior Python Engineer','categories':{'location':'Pune, India'},'applyUrl':'https://example.com/3','descriptionPlain':'Python internal tools'}]
    jobs=live.parse('greenhouse','x','X',gh)+live.parse('lever','y','Y',lv)
    assert [j['company'] for j in jobs]==['X','Y'] and jobs[0]['description']=='Python APIs'
    monkeypatch.delenv('JOBS_SOURCE')
    monkeypatch.setattr(live,'load',lambda:(jobs,{}))
    found=app.search(app.Search(query='Python internship',location='Bengaluru'))
    assert [j['company'] for j in found]==['X']
