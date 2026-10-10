const { test, expect } = require('@playwright/test');

const metrics = { perplexity: 80, burstiness: .7, entropy: 4, ai_phrase_rate: 0, avg_sentence_length: 12, structural_regularity: .4 };
const scores = { human:.6, ai_written:.3, ai_paraphrased:.1, observed_signals:[], truncated:false, analyzed_tokens:720, total_tokens:720, window_count:3, complete:true, warnings:[] };
const rewrite = {
  original_text:'It is worth noting that we utilize this.', humanized_text:'We use this.',
  status:'accepted_heuristic', warnings:[], similarity_threshold:.85, style:'natural', strength:'light',
  original_metrics:{...metrics, perplexity:100}, final_metrics:metrics,
  post_heuristic_metrics:{...metrics, perplexity:999},
  heuristic_similarity:.93, llm_similarity:null, chunks_processed:1, complete:true,
};

async function fakeApi(page, { ready=true, pending=false, deferred=false, failure=false }={}) {
  const bodies = [];
  const cancellations = [];
  const jobs = new Map();
  await page.route('http://localhost:8000/**', async (route) => {
    const request = route.request(), url = new URL(request.url());
    if (request.method()==='OPTIONS') return route.fulfill({status:204, headers:{'Access-Control-Allow-Origin':'*','Access-Control-Allow-Headers':'Content-Type','Access-Control-Allow-Methods':'GET, POST, OPTIONS'}});
    let result, status=200;
    if (url.pathname==='/ready') {
      result = {status:ready?'ready':'not_ready',classifier:{ready}};
      status = ready?200:503;
    } else if (url.pathname==='/jobs' && request.method()==='POST') {
      const body = request.postDataJSON(); bodies.push(body);
      const id = String(bodies.length);
      const job = {id,operation:body.operation,state:'running',stage:'rewriting',completed_units:1,total_units:3,result:null,error:null};
      jobs.set(id, job); result={...job}; status=202;
      if (deferred) await new Promise(resolve=>setTimeout(resolve,400));
    } else if (url.pathname.endsWith('/cancel')) {
      const id = url.pathname.split('/')[2]; cancellations.push(id);
      const job = jobs.get(id); job.state='cancelled'; job.stage='cancelled'; result=job;
    } else {
      const id = url.pathname.split('/')[2], job=jobs.get(id);
      if (!pending && job.state!=='cancelled') {
        if (failure) { job.state='failed'; job.stage='failed'; job.error={code:'model_unavailable',detail:'Required local model is unavailable'}; }
        else { job.state='completed'; job.stage='completed';
        job.result=job.operation==='classify'?scores:job.operation==='humanize'?rewrite:{
          operation:'compare',original_scores:scores,revised_scores:{human:.7,ai_written:.2,ai_paraphrased:.1},
          delta_percentage_points:{human:10,ai_written:-10,ai_paraphrased:0},
          original_analysis:{analyzed_tokens:720},revised_analysis:{analyzed_tokens:700},rewrite,
        }; }
      }
      result=job;
    }
    await route.fulfill({status,json:result,headers:{'Access-Control-Allow-Origin':'*'}});
  });
  return {bodies,cancellations};
}

test('rewrite remains available without classifier and uses accepted metrics', async ({page}) => {
  await fakeApi(page,{ready:false});
  await page.goto('/');
  await page.getByLabel('Input text').fill(rewrite.original_text);
  await expect(page.getByRole('button',{name:'Analyze text'})).toBeDisabled();
  await page.getByRole('button',{name:'Rewrite text'}).click();
  await expect(page.getByText('We use this.',{exact:true})).toBeVisible();
  await expect(page.locator('.metric-card').first().locator('.value')).toHaveText('80.00');
  const download = page.waitForEvent('download');
  await page.getByRole('button',{name:'Download .txt'}).click();
  expect((await download).suggestedFilename()).toBe('textorigin-humanized.txt');
});

test('classify covers whole document and compare displays signed changes', async ({page}) => {
  await fakeApi(page);
  await page.goto('/');
  await page.getByLabel('Input text').fill(rewrite.original_text);
  await page.getByRole('button',{name:'Analyze text'}).click();
  await expect(page.getByText('720 content tokens analyzed across 3 windows.')).toBeVisible();
  await page.getByRole('button',{name:'Compare scores'}).click();
  await expect(page.getByRole('heading',{name:'Compare model scores'})).toBeVisible();
  await expect(page.locator('tbody tr').first()).toContainText('10.0');
});

test('progress and cancel reach the server', async ({page}) => {
  const api=await fakeApi(page,{pending:true});
  await page.goto('/');
  await page.getByLabel('Input text').fill('Text to rewrite.');
  await page.getByRole('button',{name:'Rewrite text'}).click();
  await expect(page.getByRole('progressbar')).toHaveAttribute('value','1');
  await page.getByRole('button',{name:'Cancel operation'}).click();
  await expect(page.getByText('cancelled',{exact:true})).toBeVisible();
  expect(api.cancellations).toEqual(['1']);
  await expect(page.locator('.diff-view')).toHaveCount(0);
});

test('edited input cancels delayed submissions and cannot show stale results', async ({page}) => {
  const api=await fakeApi(page,{deferred:true});
  await page.goto('/');
  await page.getByLabel('Input text').fill(rewrite.original_text);
  await page.getByRole('button',{name:'Rewrite text'}).click();
  await expect.poll(()=>api.bodies.length).toBe(1);
  await page.getByLabel('Input text').fill('Different input.');
  await expect.poll(()=>api.cancellations.length).toBe(1);
  await expect(page.locator('.diff-view')).toHaveCount(0);
});

test('BYOK clears from UI and is absent from classify and persistent storage', async ({page}) => {
  const api=await fakeApi(page);
  await page.goto('/');
  await page.getByLabel('Input text').fill(rewrite.original_text);
  await page.getByLabel('Add minor DeepSeek polish').check();
  await page.getByLabel('DeepSeek API key',{exact:false}).fill('test-secret');
  await page.getByRole('button',{name:'Analyze text'}).click();
  await expect(page.getByText('720 content tokens analyzed across 3 windows.')).toBeVisible();
  expect(api.bodies[0].deepseek_api_key).toBeUndefined();
  await page.getByRole('button',{name:'Rewrite text'}).click();
  await expect(page.getByText('We use this.',{exact:true})).toBeVisible();
  expect(api.bodies[1].deepseek_api_key).toBe('test-secret');
  await page.getByRole('button',{name:'Clear key',exact:true}).click();
  await expect(page.getByLabel('DeepSeek API key',{exact:false})).toHaveValue('');
  expect(await page.evaluate(()=>JSON.stringify({...localStorage})+JSON.stringify({...sessionStorage}))).not.toContain('test-secret');
});


test('failed operation reports error and allows retry without stale output', async ({page}) => {
  await fakeApi(page,{failure:true});
  await page.goto('/');
  await page.getByLabel('Input text').fill('Input.');
  await page.getByRole('button',{name:'Rewrite text'}).click();
  await expect(page.getByRole('alert').filter({hasText:'Required local model is unavailable'})).toBeVisible();
  await expect(page.getByRole('button',{name:'Rewrite text'})).toBeEnabled();
  await expect(page.locator('.diff-view')).toHaveCount(0);
});


test('cancel before the submission returns still cancels the server job', async ({page}) => {
  const api=await fakeApi(page,{pending:true,deferred:true});
  await page.goto('/');
  await page.getByLabel('Input text').fill('Input.');
  await page.getByRole('button',{name:'Rewrite text'}).click();
  await page.getByRole('button',{name:'Cancel operation'}).click();
  await expect.poll(()=>api.cancellations.length).toBe(1);
  await expect(page.getByText('cancelled',{exact:true})).toBeVisible();
});
