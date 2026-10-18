"""Start or verify a local vLLM server. Stop only the process this runner starts."""
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request


def served_models(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=3) as response:
            return json.load(response)['data']
    except urllib.error.URLError:
        return None


def serving_versions():
    result = subprocess.run(['conda', 'run', '-n', 'vllm', 'python', '-c',
                             'import importlib.metadata as m,json,sys; print(json.dumps(dict('
                             '[(p,m.version(p)) for p in ("vllm","transformers","torch")]+[("python",sys.version.split()[0])])))'],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Cannot inspect the vLLM environment: ' + result.stderr)
    return json.loads(result.stdout)


def ensure_server(case, port, output):
    models = served_models(port)
    if models is not None:
        model = next((m for m in models if m['id'] == case['model']), None)
        if model is None or model.get('max_model_len', case['context']) != case['context']:
            raise SystemExit(f'Port {port} serves a different model/context; use another port or stop that server.')
        return None
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', port)) == 0:
            raise SystemExit(f'Port {port} is occupied by a service without a compatible /v1/models endpoint; use --port with a free port.')
    # Fail promptly when CUDA is unavailable, retaining the preflight output.
    output.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    (output / 'gpu.txt').write_text(gpu.stdout + gpu.stderr)
    if gpu.returncode:
        raise SystemExit(f'GPU unavailable; see {output / "gpu.txt"}')
    command = ['conda', 'run', '--no-capture-output', '-n', 'vllm', 'vllm', 'serve', case.get('weights', case['model']),
               '--host', '127.0.0.1', '--port', str(port), '--served-model-name', case['model'],
               '--gpu-memory-utilization', str(case['gpu_memory_utilization']), '--max-model-len', str(case['context']),
               '--generation-config', 'auto', '--seed', '0']
    command += case.get('server_args', [])
    if case['model_label'] == 'qwen14-awq':
        command += ['--default-chat-template-kwargs', '{"enable_thinking": false}']
    env = dict(os.environ, HF_HUB_OFFLINE='1', VLLM_USE_FLASHINFER_SAMPLER='0', VLLM_USE_DEEP_GEMM='0')
    (output / 'python_version.txt').write_text(json.dumps(serving_versions(), indent=2) + '\n')
    (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (output / 'vllm.log').open('a') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(180):
        models = served_models(port)
        if models and any(m['id'] == case['model'] for m in models):
            return process
        if process.poll() is not None:
            raise SystemExit(f'vLLM exited; see {output / "vllm.log"}')
        time.sleep(2)
    stop_server(process)
    raise SystemExit(f'vLLM startup timed out; see {output / "vllm.log"}')


def stop_server(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
