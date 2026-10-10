import json
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.errors import JobCapacityError, ModelUnavailableError
from src.inference.classifier import ClassifierEngine, token_windows
from src.inference.similarity import SimilarityEngine, balanced_windows
from src.services.comparison import ComparisonService
from src.services.jobs import JobManager
from src.services.rewrite import protected_spans_preserved, rewrite_document
from tests.test_api import METRICS, make_client
from tests.test_humanizer import make_service


def wait_terminal(manager, key):
    deadline = monotonic() + 3
    while monotonic() < deadline:
        state = manager.get(key)
        if state["state"] in {"completed", "cancelled", "failed"}:
            return state
        sleep(.005)
    raise AssertionError("Job never terminated")


def test_multiline_fences_and_indentation_remain_exact():
    fence = chr(96)*3
    code = fence + 'python\r\n  utilize = "get"\r\n# delve into\r\n' + fence
    text = '  - It is worth noting that we utilize this.\r\n' + code + '\r\n\r\n  Keep this.'
    result = rewrite_document(text)
    assert code in result
    assert result.startswith('  - We use this.')
    assert '\r\n\r\n  Keep this.' in result
    assert protected_spans_preserved(text, result)


def test_user_placeholder_characters_cannot_be_replaced():
    placeholder = chr(0xe000) + '0' + chr(0xe001)
    text = placeholder + ' "exact utilize". We utilize it.'
    result = rewrite_document(text)
    assert placeholder in result
    assert '"exact utilize"' in result


def test_all_window_tokens_are_covered_once():
    ids = list(range(19))
    windows = list(token_windows(ids, 6))
    assert [item for window in windows for item in window] == ids
    assert list(map(len, windows)) == [6, 6, 6, 1]
    for count in (1, 2, 5):
        assert [i for window in balanced_windows(ids, count) for i in window] == ids


def test_classifier_aggregation_includes_final_window(tmp_path):
    import torch
    class Tokenizer:
        def encode(self, text, **kwargs): return list(range(9))
        def num_special_tokens_to_add(self, **kwargs): return 2
        def build_inputs_with_special_tokens(self, ids): return [99] + ids + [99]
        def pad(self, encoded, **kwargs):
            ids = [item["input_ids"] for item in encoded]
            size = max(map(len, ids))
            class Batch(dict):
                def to(self, _device): return self
            return Batch(input_ids=torch.tensor([row + [0] * (size-len(row)) for row in ids]))
    class Model:
        def parameters(self): return iter([torch.tensor(0.)])
        def __call__(self, input_ids):
            return SimpleNamespace(logits=torch.tensor([[3., 0., 0.] if row[1] < 8 else [0., 3., 0.] for row in input_ids]))
    engine = ClassifierEngine(tmp_path, default_max_length=6)
    engine._model, engine._tokenizer = Model(), Tokenizer()
    result = engine.predict('real text')
    early, tail = torch.softmax(torch.tensor([3., 0., 0.]),0), torch.softmax(torch.tensor([0., 3., 0.]),0)
    assert result['scores']['human'] == pytest.approx((early[0]*8+tail[0]).item()/9)
    assert sum(result['scores'].values()) == pytest.approx(1)
    assert result['analyzed_tokens'] == result['total_tokens'] == 9
    assert result['window_count'] == 3 and not result['truncated']


def test_similarity_changed_tail_cannot_hide_behind_prefix():
    import torch
    class Tokenizer:
        cls_token_id = 101
        sep_token_id = 102
        def encode(self, text, **kwargs): return [1]*8 + ([1]*4 if text=='original' else [2]*4)
        def num_special_tokens_to_add(self, **kwargs): return 2
        def build_inputs_with_special_tokens(self, ids): return ids
        def pad(self, items, **kwargs):
            class Batch(dict):
                def to(self, _device): return self
            return Batch(input_ids=torch.tensor([item['input_ids'] for item in items]))
    class Model:
        max_seq_length = 6
        device = 'cpu'
        tokenizer = Tokenizer()
        def __call__(self, features):
            embeddings = [[1.,0.] if row[1]==1 else [0.,1.] for row in features['input_ids']]
            return {'sentence_embedding': torch.tensor(embeddings)}
    engine = SimilarityEngine(); engine._model = Model()
    assert engine.compare('original', 'changed') == pytest.approx(0.)


def test_final_metrics_describe_fallback_not_rejected_candidate():
    service, _ = make_service([.2])
    original = 'It is worth noting that we utilize this.'
    with patch('src.services.humanization.extract_features', side_effect=lambda text, **kwargs: {'length': len(text)}):
        result = service.humanize(original, use_llm=False)
    assert result['final_metrics'] == {'length': len(original)}
    assert result['post_heuristic_metrics'] != result['final_metrics']


def test_oversized_protected_chunk_is_never_sent_to_provider():
    service, provider = make_service([])
    original = chr(96)*3 + '\n' + 'utilize\n'*700 + chr(96)*3
    with patch('src.services.humanization.extract_features', return_value=METRICS):
        result = service.humanize(original)
    assert result['humanized_text'] == original
    assert any('oversized' in item for item in result['warnings'])
    provider.polish.assert_not_called()


def test_shared_compare_uses_accepted_output_and_percentage_points():
    classifier = MagicMock()
    classifier.classify.side_effect = [
        {'scores': {'human': .2, 'ai_written': .6, 'ai_paraphrased': .2}, 'model_version': 'same'},
        {'scores': {'human': .4, 'ai_written': .5, 'ai_paraphrased': .1}, 'model_version': 'same'},
    ]
    humanizer = MagicMock(); humanizer.humanize.return_value = {'humanized_text': 'Accepted.'}
    result = ComparisonService(classifier, humanizer).compare('Original.')
    assert classifier.classify.call_args_list[1].args[0] == 'Accepted.'
    assert result['delta_percentage_points']['human'] == pytest.approx(20)


def test_jobs_queue_cancel_race_and_secret_cleanup():
    entered, release = Event(), Event()
    def handler(operation, payload, context):
        entered.set(); context.progress('work', 1, 3); release.wait(2)
        return {'value': 'late result'}
    manager = JobManager(handler, max_queued=1)
    try:
        first = manager.submit('humanize', SimpleNamespace(text='one', key='secret'))
        assert entered.wait(1)
        assert manager.get(first['id'])['completed_units'] == 1
        second = manager.submit('humanize', SimpleNamespace(text='two'))
        with pytest.raises(JobCapacityError): manager.submit('humanize', SimpleNamespace(text='three'))
        assert manager.cancel(second['id'])['state'] == 'cancelled'
        assert manager.cancel(second['id'])['state'] == 'cancelled'
        manager.cancel(first['id']); release.set()
        state = wait_terminal(manager, first['id'])
        assert state['state'] == 'cancelled' and state['result'] is None
        assert manager._jobs[first['id']].payload is None
        assert 'secret' not in json.dumps(state)
    finally: release.set(); manager.close()


def test_jobs_failure_redaction_expiry_and_memory_admission():
    def handler(*args): raise ValueError('secret submitted-text')
    manager = JobManager(handler, max_payload_bytes=8192, ttl_seconds=1)
    try:
        with pytest.raises(JobCapacityError): manager.submit('classify', SimpleNamespace(text='x'*5000))
        job = manager.submit('classify', SimpleNamespace(text='tiny'))
        state = wait_terminal(manager, job['id'])
        assert state['state'] == 'failed' and 'secret' not in json.dumps(state)
        manager._jobs[job['id']].finished_at = monotonic()-2
        with pytest.raises(KeyError): manager.get(job['id'])
    finally: manager.close()


def test_readiness_rejects_absent_and_malformed_artifacts(tmp_path):
    engine = ClassifierEngine(tmp_path)
    assert not engine.readiness()['ready']
    for name in ('config.json','tokenizer_config.json','tokenizer.json','model.safetensors'):
        (tmp_path/name).write_text('invalid')
    assert not engine.readiness()['ready']
    with pytest.raises(ModelUnavailableError): engine.predict('Text.')


def test_job_api_and_sync_results_are_equivalent():
    with make_client() as client:
        expected = client.post('/classify', json={'text': 'Valid.', 'include_signals': False}).json()
        created = client.post('/jobs', json={'operation':'classify','text':'Valid.'})
        assert created.status_code == 202
        key = created.json()['id']
        deadline = monotonic()+2
        while monotonic()<deadline:
            status = client.get('/jobs/'+key).json()
            if status['state'] in {'completed','failed'}: break
            sleep(.005)
        assert status['state']=='completed'
        assert status['result']==expected
        assert client.post('/jobs/'+key+'/cancel').json()['state']=='completed'
        assert client.get('/jobs/missing').status_code==404


def test_cli_noninteractive_missing_key_and_keyboard_interrupt(monkeypatch, capsys):
    from src.cli import main
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.setattr('sys.stdin.isatty',lambda:False)
    assert main(['rewrite','--text','Text.','--provider','deepseek'])==2
    monkeypatch.setattr('src.cli.rewrite_document',MagicMock(side_effect=KeyboardInterrupt))
    assert main(['rewrite','--text','Text.'])==130
    assert 'Cancelled' in capsys.readouterr().err


def test_cancellation_releases_capacity_for_next_operation():
    from src.errors import OperationCancelled
    from src.services.capacity import InferenceCapacity
    from src.services.execution import ExecutionContext
    capacity = InferenceCapacity(1, .01)
    context = ExecutionContext(); context.cancelled.set()
    with pytest.raises(OperationCancelled):
        with capacity.claim(context):
            raise AssertionError("cancelled operation entered capacity")
    with capacity.claim():
        pass


def test_provider_retries_do_not_change_a_supplied_empty_key():
    from src.integrations.deepseek import DeepSeekClient
    from src.errors import ExternalServiceError
    provider = DeepSeekClient("https://api.deepseek.com", "deepseek-chat", 1, api_key="owner")
    try:
        with pytest.raises(ExternalServiceError):
            provider.polish("Text.", style="natural", strength="light", api_key="")
    finally:
        provider.close()


def test_structure_guard_rejects_provider_merging_paragraphs():
    service, _ = make_service([.9, .99], polished="First paragraph. Second paragraph.")
    text = "First paragraph.\n\nSecond paragraph."
    with patch('src.services.humanization.extract_features', return_value=METRICS):
        result = service.humanize(text)
    assert result['humanized_text']==text
    assert 'chunk_1_polisher_unavailable' in result['warnings']


def test_split_rejects_conflicting_lineage_and_cross_family_duplicates():
    from tests.test_split import make_dataset
    from src.data.split import grouped_stratified_split
    frame = make_dataset()
    frame.loc[1, 'source_id']=frame.loc[0, 'source_id']
    with pytest.raises(ValueError, match='identities'):
        grouped_stratified_split(frame)
    frame = make_dataset()
    frame.loc[1, 'text']=frame.loc[0, 'text']
    with pytest.raises(ValueError, match='duplicate'):
        grouped_stratified_split(frame)


def test_buffer_sampling_is_bounded_and_reproducible(monkeypatch):
    from src.data.loader import load_raw_data
    class Stream:
        def shuffle(self, **kwargs):
            assert kwargs=={'seed':42,'buffer_size':10_000}
            return [{'text':f'{kind} {i}','source':kind} for i in range(5) for kind in ('human','ai')]
    monkeypatch.setattr('src.data.loader.load_dataset',lambda *args,**kwargs:Stream())
    first=load_raw_data(sample_size=3,streaming=True)
    second=load_raw_data(sample_size=3,streaming=True)
    assert first.equals(second)
    assert first['source'].value_counts().to_dict()=={'human':3,'ai_written':3}
    assert first['source_id'].is_unique


def test_cli_masked_key_uses_shared_service_without_echo(monkeypatch, capsys):
    from src.cli import _deepseek_rewrite
    humanizer=MagicMock()
    humanizer.humanize.return_value={'humanized_text':'Accepted.'}
    services=SimpleNamespace(humanization=humanizer,close=MagicMock())
    monkeypatch.delenv('DEEPSEEK_API_KEY',raising=False)
    monkeypatch.setattr('sys.stdin.isatty',lambda:True)
    monkeypatch.setattr('getpass.getpass',lambda prompt:'test-secret')
    monkeypatch.setattr('src.bootstrap.build_services',lambda settings:services)
    assert _deepseek_rewrite('Input.','natural','light')['humanized_text']=='Accepted.'
    assert humanizer.humanize.call_args.kwargs['api_key']=='test-secret'
    assert 'test-secret' not in capsys.readouterr().out
    services.close.assert_called_once()


def test_compare_http_contract_and_job_parity():
    with make_client() as client:
        sync = client.post('/compare',json={'text':'Original.','use_llm':False})
        assert sync.status_code==200
        assert sync.json()['delta_percentage_points']['human']==0
        created=client.post('/jobs',json={'operation':'compare','text':'Original.'}).json()
        deadline=monotonic()+2
        while monotonic()<deadline:
            status=client.get('/jobs/'+created['id']).json()
            if status['state'] in {'completed','failed'}:break
            sleep(.005)
        assert status['state']=='completed'
        assert status['result']==sync.json()


def test_periodic_checkpoint_preserves_epoch_controls_and_saves_mid_epoch():
    from src.model.classifier import PeriodicCheckpoint
    callback=PeriodicCheckpoint(250)
    control=SimpleNamespace(should_save=False,should_evaluate=True)
    callback.on_step_end(None,SimpleNamespace(global_step=249),control)
    assert not control.should_save and control.should_evaluate
    callback.on_step_end(None,SimpleNamespace(global_step=250),control)
    assert control.should_save and control.should_evaluate


def test_cancellation_stops_long_feature_work_between_windows(monkeypatch):
    import torch
    from src.errors import OperationCancelled
    from src.features.extractor import compute_perplexity
    from src.services.execution import ExecutionContext
    context=ExecutionContext()
    calls=[]
    class Model:
        def parameters(self): return iter([torch.tensor(0.)])
        def __call__(self, input_ids, labels):
            calls.append(input_ids.shape[1])
            context.cancelled.set()
            return SimpleNamespace(loss=torch.tensor(1.))
    tokenizer=SimpleNamespace(encode=lambda *args,**kwargs:list(range(20)))
    monkeypatch.setattr('src.features.extractor.get_perplexity_model',lambda:(Model(),tokenizer))
    with pytest.raises(OperationCancelled):
        compute_perplexity('Long text.',max_length=5,context=context)
    assert calls==[5]


def test_single_quotes_and_indented_code_preserve_protected_words():
    text="We utilize it. 'utilize get' remains quoted.\n    get = 'utilize'\n\tutilize(get)"
    result=rewrite_document(text,style='formal')
    assert "'utilize get'" in result
    assert "    get = 'utilize'" in result
    assert '\tutilize(get)' in result
    assert protected_spans_preserved(text,result)
    assert rewrite_document('We get through a lot of water.',style='formal')=='We get through a lot of water.'
