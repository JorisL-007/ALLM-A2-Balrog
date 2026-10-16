"""Exercise failure accounting and held-out guard without a server or game."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run


class FailedProcess:
    pid=987654321
    def __init__(self,*args,**kwargs):pass
    def wait(self,timeout=None):return 17


class Opener:
    def open(self,*args,**kwargs):
        return io.StringIO(json.dumps({'data':[{'id':run.MODEL}]}))


class ControllerTests(unittest.TestCase):
    def test_failed_attempt_is_charged_and_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            state=dict(benchmark_stop_unix=10**12,limits=dict(benchmark_seconds=7200,episode_attempts=300,e2_candidates=6),
                       benchmark_seconds_used=0,episode_attempts=0,e2_candidates_started=[],active_processes=[])
            (root/'STATE.json').write_text(json.dumps(state))
            case=json.loads((run.HERE/'development_manifest.json').read_text())[0]['case_id']
            args=['run.py','--variants','naive','--pass-id','fixture','--output',tmp,'--case-ids',case]
            with patch.object(sys,'argv',args),patch.object(run.urllib.request,'build_opener',return_value=Opener()),patch.object(run.subprocess,'Popen',FailedProcess),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit,'Incomplete attempt preserved'):
                    run.main()
            saved=json.loads((root/'STATE.json').read_text())
            self.assertEqual(saved['episode_attempts'],1)
            self.assertGreater(saved['benchmark_seconds_used'],0)
            self.assertIsNone(saved['active_evaluation'])
            failed=list((root/'runs').rglob('incomplete.json'))
            self.assertEqual(len(failed),1)
            self.assertEqual(json.loads(failed[0].read_text())['status'],'incomplete')
            ledger=[json.loads(s) for s in (root/'ledger.jsonl').read_text().splitlines()]
            self.assertEqual([d['event'] for d in ledger],['episode_started','episode_finished'])

    def test_exhausted_budget_spawns_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            state=dict(benchmark_stop_unix=10**12,limits=dict(benchmark_seconds=7200,episode_attempts=300,e2_candidates=6),
                       benchmark_seconds_used=7200,episode_attempts=0,e2_candidates_started=[],active_processes=[])
            (root/'STATE.json').write_text(json.dumps(state))
            args=['run.py','--variants','naive','--pass-id','fixture','--output',tmp]
            with patch.object(sys,'argv',args),patch.object(run.urllib.request,'build_opener',return_value=Opener()),patch.object(run.subprocess,'Popen') as spawn,contextlib.redirect_stdout(io.StringIO()):
                run.main()
                spawn.assert_not_called()

    def test_confirmation_refuses_development_output(self):
        args=['run.py','--mode','confirmation','--variants','naive','--pass-id','fixture']
        with patch.object(sys,'argv',args),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):run.main()


if __name__=='__main__':unittest.main()
