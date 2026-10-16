"""Verify saved experimental evidence without generating actions or new layouts."""
import collections
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
MODEL='Qwen/Qwen3-4B-Instruct-2507-FP8'


def main():
    expected={x['case_id'] for x in json.loads((HERE/'development_manifest.json').read_text())}
    groups=collections.defaultdict(set);calibration_groups=collections.defaultdict(set)
    anchors={};calls=tokens=episodes=setup=calibration=0
    stop_reasons=collections.Counter()
    paths=list((HERE/'runs').glob('*/*/*/result.json'))+list((HERE/'e3/runs').glob('*/*/*/result.json'))
    paths+=list((HERE/'protocol/calibration/runs').glob('*/*/result.json'))
    calibration_expected={x['case_id'].rsplit(':',1)[0]+':12'
                          for x in json.loads((HERE/'development_manifest.json').read_text()) if x['environment']=='babaisai'}
    for path in sorted(paths):
        r=json.loads(path.read_text());folder=path.parent
        assert r['seed'] in (11,12),str(path)
        assert r['status'] in ('setup','complete'),str(path)
        is_calibration=r.get('stage')=='variance_calibration'
        assert r['case_id'] in (calibration_expected if is_calibration else expected),str(path)
        key=(str(path.parents[3].relative_to(HERE)),path.parents[2].name,r['variant'])
        if is_calibration:
            calibration+=1
            assert r['case_id'] not in calibration_groups[r['variant']]
            calibration_groups[r['variant']].add(r['case_id'])
        elif r['status']=='complete':
            assert r['case_id'] not in groups[key],('duplicate',key,r['case_id'])
            groups[key].add(r['case_id']);episodes+=1
        else:setup+=1
        initial=json.loads((folder/'initial.json').read_text())
        assert initial['fingerprint']==r['initial_fingerprint']
        assert hashlib.sha256(json.dumps(dict(text=initial['text'],mission=initial['mission']),sort_keys=True).encode()).hexdigest()==r['initial_fingerprint']
        if r['case_id'] in anchors:assert anchors[r['case_id']]==r['initial_fingerprint'],('layout mismatch',path)
        anchors[r['case_id']]=r['initial_fingerprint']
        requests={};responses={}
        for line in (folder/'calls.jsonl').open():
            c=json.loads(line)
            assert c['event'] in ('request','response'),('error in completed case',path)
            if c['event']=='request':
                assert c['model']==MODEL
                assert c['call_id'] not in requests
                assert c['max_tokens'] in (96,128,512)
                requests[c['call_id']]=c
            else:
                assert c['call_id'] not in responses
                responses[c['call_id']]=c
                stop_reasons[c['stop_reason']]+=1
        assert set(requests)==set(responses)
        assert len(requests)==r['calls']
        assert sum(c['input_tokens'] for c in responses.values())==r['input_tokens']
        assert sum(c['output_tokens'] for c in responses.values())==r['output_tokens']
        steps=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
        assert len(steps)==r['steps']
        assert sum(s['input_tokens'] for s in steps)==r['input_tokens']
        assert sum(s['output_tokens'] for s in steps)==r['output_tokens']
        assert sum(s['invalid'] for s in steps)==r['invalid_commands']
        assert sum(s['text_unchanged'] for s in steps)==r['unchanged_text']
        assert sum(s['repeated_action'] for s in steps)==r['repeated_actions']
        if r.get('stage')=='E3':
            actor=json.loads((HERE/'FROZEN.json').read_text())['selected_actor']
            reference=HERE/'runs/screen_1'/actor/folder.name/'calls.jsonl'
            first=json.loads(next(reference.open()))
            assert requests[0]['messages']==first['messages'],('frozen actor initial prompt changed',path)
            assert requests[0]['max_tokens']==first['max_tokens']
        if r.get('stage')=='E3' and r['variant'].endswith('_reflection'):
            assert 'Within-episode corrective lessons' not in json.dumps(requests[0]['messages'])
            file=folder/'reflections.jsonl'
            refs=[json.loads(s) for s in file.read_text().splitlines()] if file.exists() else []
            assert len(refs)==r['reflections']<=2
            assert all(len(x['memory'])<=2 and all(len(lesson)<=400 for lesson in x['memory']) for x in refs)
        calls+=r['calls'];tokens+=r['output_tokens']
    for key,cases in groups.items():assert cases==expected,('incomplete full panel',key,len(cases))
    for variant,cases in calibration_groups.items():assert cases==calibration_expected,('incomplete calibration',variant)
    hashes=json.loads((HERE/'run_hashes.json').read_text())
    assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    if (HERE/'e3/hashes.json').exists():
        for name,digest in json.loads((HERE/'e3/hashes.json').read_text()).items():
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,('E3 code changed',name)
    provenance=json.loads((HERE/'provenance.json').read_text())
    for name,digest in provenance['files'].items():
        assert hashlib.sha256((HERE.parents[1]/name).read_bytes()).hexdigest()==digest,('upstream changed',name)
    for name,digest in provenance.get('installed_mechanics_hashes',{}).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,('installed mechanics changed',name)
    state=json.loads((HERE/'STATE.json').read_text())
    assert state['episode_attempts']<=state['limits']['episode_attempts']
    assert state['benchmark_seconds_used']<=state['limits']['benchmark_seconds']
    assert state.get('active_evaluation') is None
    ledger=[json.loads(s) for s in (HERE/'ledger.jsonl').read_text().splitlines()]
    finished=[r for r in ledger if r['event']=='episode_finished']
    fixtures=[r for r in ledger if r['event']=='unit_fixture_budget_charge']
    assert len(finished)+sum(r['attempts'] for r in fixtures)==state['episode_attempts']
    assert abs(sum(r['elapsed_seconds'] for r in finished)+sum(r['charged_execution_seconds'] for r in fixtures)-state['benchmark_seconds_used'])<.01
    result=dict(status='passed',completed_scored_episodes=episodes,setup_episodes=setup,variance_calibration_episodes=calibration,
        complete_panels=len(groups),logged_llm_calls=calls,logged_output_tokens=tokens,
        stop_reasons=dict(stop_reasons),only_model=MODEL,heldout_observations_generated=0,
        notes=['Every request, response, step and token total reconciles.','All initial fingerprints match across arms and repetitions.',
               'Installed-environment unit fixtures are accounted separately in STATE/ledger.'])
    (HERE/'audit/integrity_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
