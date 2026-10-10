"""Exercise unattended sequencing/recovery without installing packages or running models."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_gpu


class GPUOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original_cwd = Path.cwd()
        self.addCleanup(os.chdir, self.original_cwd)
        for name in ['src', 'scripts']:
            (self.root / name).mkdir()
        (self.root / 'run_gpu.py').write_text('source')
        (self.root / 'pyproject.toml').write_text('project')
        self.run = self.root / 'runs/gpu'
        self.commands = []
        self.fail = None

    def fake_process(self, command, **kwargs):
        self.commands.append(command)
        stage = 'preflight' if '--worker' in command and command[-1] == 'preflight' else (
            'dataset' if '--worker' in command else next((name for name in
            ['scripts.audit_dataset', 'scripts.train', 'scripts.verify_runtime'] if name in command), 'setup'))
        self.run.mkdir(parents=True, exist_ok=True)
        if stage == 'dataset':
            for name in ['combined.csv', 'combined.provenance.json']:
                (self.run / name).write_text('dataset')
        elif stage == 'scripts.audit_dataset':
            for name in ['combined.clean.csv', 'audit.json']:
                (self.run / name).write_text('audit')
        elif stage == 'scripts.train':
            model = self.run / 'model'
            model.mkdir(exist_ok=True)
            (model / 'training-identity.json').write_text('{}')
            if self.fail == stage:
                checkpoint = model / 'checkpoint-250'
                checkpoint.mkdir(exist_ok=True)
                (checkpoint / 'trainer_state.json').write_text('{}')
            else:
                for name in ['model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
                             'config.json', 'textorigin-manifest.json', 'evaluation.json', 'evaluation.md']:
                    (model / name).write_text('model')
        elif stage == 'scripts.verify_runtime':
            for name in ['runtime-verification.json', 'runtime-verification-offline.json']:
                (self.run / 'model' / name).write_text('{}')
        code = 1 if self.fail == stage else 0
        class Process:
            pid = 12345
            returncode = code
            def poll(self):
                return self.returncode
        return Process()

    def invoke(self, extra=()):
        with patch.object(run_gpu, 'ROOT', self.root), patch('sys.argv', ['run_gpu.py', *extra]), \
             patch.object(run_gpu.venv.EnvBuilder, 'create', side_effect=lambda path: path.mkdir(parents=True, exist_ok=True)), \
             patch.object(run_gpu.subprocess, 'Popen', side_effect=self.fake_process):
            run_gpu.main()

    def test_full_sequence_and_verified_reuse(self):
        self.invoke()
        self.assertTrue(json.loads((self.run / 'status.json').read_text())['complete'])
        self.assertFalse((self.run / 'pipeline.lock').exists())
        self.assertTrue(any('torch==2.13.0' in command for command in self.commands))
        trained = next(command for command in self.commands if 'scripts.train' in command)
        self.assertEqual(trained[trained.index('--mixed-precision') + 1], 'fp16')
        self.assertNotIn('--no-gradient-checkpointing', trained)
        count = len(self.commands)
        self.invoke()
        second_run = self.commands[count:]
        self.assertFalse(any('scripts.train' in command or '--worker' in command and command[-1] == 'dataset'
                             for command in second_run))

    def test_failure_stops_then_checkpoint_resumes(self):
        self.fail = 'scripts.train'
        with self.assertRaises(SystemExit) as caught:
            self.invoke()
        self.assertEqual(caught.exception.code, 1)
        self.assertFalse(json.loads((self.run / 'status.json').read_text())['complete'])
        self.assertFalse(any('scripts.verify_runtime' in command for command in self.commands))
        self.fail = None
        self.invoke()
        trained = [command for command in self.commands if 'scripts.train' in command][-1]
        self.assertIn('--resume-from-checkpoint', trained)
        self.assertTrue(json.loads((self.run / 'status.json').read_text())['complete'])

    def test_cpu_fallback_is_refused(self):
        from types import SimpleNamespace
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict('sys.modules', {'torch': torch}):
            with self.assertRaisesRegex(RuntimeError, 'CPU fallback is disabled'):
                run_gpu.worker(SimpleNamespace(worker='preflight'), self.run)

    def test_active_lock_is_not_removed(self):
        self.run.mkdir(parents=True)
        lock = self.run / 'pipeline.lock'
        lock.write_text('existing-owner')
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertEqual(lock.read_text(), 'existing-owner')
        self.assertEqual(self.commands, [])

    def test_modified_artifact_is_rejected(self):
        self.invoke()
        (self.run / 'combined.csv').write_text('tampered')
        with self.assertRaises(SystemExit):
            self.invoke()
        self.assertIn('artifacts changed', json.loads((self.run / 'status.json').read_text())['error'])

    def test_configuration_change_is_rejected(self):
        self.invoke()
        with self.assertRaises(SystemExit):
            self.invoke(['--epochs', '4'])
        self.assertIn('Settings or source changed', json.loads((self.run / 'status.json').read_text())['error'])

    def test_status_lock_does_not_stop_pipeline(self):
        replace = Path.replace

        def locked_status(source, target):
            if target.name == 'status.json':
                raise PermissionError('status locked')
            return replace(source, target)

        with patch.object(Path, 'replace', autospec=True, side_effect=locked_status), \
             patch.object(run_gpu.time, 'sleep'), patch('sys.stderr') as stderr:
            self.invoke()
        self.assertTrue((self.run / 'runtime-validation.completed.json').exists())
        self.assertFalse((self.run / 'pipeline.lock').exists())
        self.assertEqual(list(self.run.glob('*.tmp')), [])
        self.assertIn('Status may be stale', str(stderr.write.call_args_list))

    def test_status_lock_does_not_mask_stage_failure(self):
        self.fail = 'scripts.train'
        replace = Path.replace

        def locked_status(source, target):
            if target.name == 'status.json':
                raise PermissionError('status locked')
            return replace(source, target)

        with patch.object(Path, 'replace', autospec=True, side_effect=locked_status), \
             patch.object(run_gpu.time, 'sleep'), patch('sys.stderr') as stderr:
            with self.assertRaises(SystemExit) as caught:
                self.invoke()
        self.assertEqual(caught.exception.code, 1)
        self.assertIn('training failed (exit 1)', str(stderr.write.call_args_list))
        self.assertFalse((self.run / 'pipeline.lock').exists())
        self.assertFalse(any('scripts.verify_runtime' in command for command in self.commands))

    def test_save_retries_lock_and_preserves_unrelated_temporary_file(self):
        target = self.root / 'status.json'
        target.write_text('{"old": true}')
        unrelated = self.root / 'status.tmp'
        unrelated.write_text('another writer')
        replace = Path.replace
        attempts = []

        def transient_lock(source, destination):
            attempts.append(source)
            if len(attempts) < 3:
                self.assertEqual(json.loads(target.read_text()), {'old': True})
                raise PermissionError('status locked')
            return replace(source, destination)

        with patch.object(Path, 'replace', autospec=True, side_effect=transient_lock), \
             patch.object(run_gpu.time, 'sleep'):
            run_gpu.save(target, {'complete': True})
        self.assertEqual(len(attempts), 3)
        self.assertEqual(json.loads(target.read_text()), {'complete': True})
        self.assertEqual(unrelated.read_text(), 'another writer')
        self.assertEqual(list(self.root.glob('status-*.tmp')), [])

    def test_critical_metadata_write_failure_still_raises(self):
        target = self.root / 'pipeline-config.json'
        target.write_text('{"old": true}')
        with patch.object(Path, 'replace', side_effect=PermissionError('locked')) as replace, \
             patch.object(run_gpu.time, 'sleep'):
            with self.assertRaises(PermissionError):
                run_gpu.save(target, {'new': True})
        self.assertEqual(replace.call_count, 6)
        self.assertEqual(json.loads(target.read_text()), {'old': True})
        self.assertEqual(list(self.root.glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
