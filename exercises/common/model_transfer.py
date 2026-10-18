"""Sequential transfer-study orchestration over the ordinary BALROG runner."""
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from exercises.common.profiles import (ROOT, TRANSFER_MODELS, OLMO_MODEL, OLMO_REVISION,
                                      OLMO_GGUF_REVISION, transfer_cases, transfer_pilots)
from exercises.common.run_experiments import (check_manifest, completed_case, episode_command,
                                             run_case, target_directory)
from exercises.common.results import result_files, verify
from exercises.common.server import ensure_server, served_models, stop_server, serving_versions
from exercises.common.transport_audit import read_attempts, spending


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reusable(c):
    if not c["reuse_from"]:
        return False
    source = ROOT / c["reuse_from"]
    required = ("complete.json", "command.json", "llm_calls.jsonl")
    if not all((source / name).exists() for name in required):
        return False
    result = verify(source, c["environment"], c["task"], c["condition"])
    if result is None:
        return False
    meta = json.loads((source / "complete.json").read_text())
    client, agent = result["client"], result["agent"]
    generated = client["generate_kwargs"]
    recorded = json.loads((source / "command.json").read_text())
    command = recorded["command"]
    settings = json.loads((ROOT / "experiment_results/followup_v1/manifest.json").read_text())
    expected = [c["agent"], "reflect" if c["condition"] == "e3" else "planning", 16, 0, 0, 0, 256]
    flags = recorded["environment"]
    return (all(meta[k] == c[k] for k in ("environment", "task", "seed", "condition", "budget", "repeat"))
            and settings["conditions"][c["condition"]] == expected
            and client["model_id"] == c["model"] and generated["temperature"] == 0
            and generated["max_tokens"] == 256 and agent["type"] == c["agent"]
            and agent["max_text_history"] == 16 and agent["max_image_history"] == 0
            and result["num_steps"] <= c["budget"]
            and f'eval.max_steps_per_episode={c["budget"]}' in command
            and 'eval.num_workers=1' in command
            and flags.get("BALROG_CUSTOM_GUIDE", "0") == "0"
            and flags.get("BALROG_CUSTOM_GATE", "0") == "0"
            and flags.get("BALROG_CUSTOM_GOAL_STATE", "0") == "0"
            and (c["environment"] != "crafter" or f'envs.crafter_kwargs.seed={c["seed"]}' in command))


def reference_episode(out, c):
    from analysis.analyze_e2_e3_model_transfer import reference_valid
    target = target_directory(out / c["model_label"], c)
    path = target / "reference.json"
    if path.exists():
        reference = json.loads(path.read_text())
        if reference["source"] != c["reuse_from"] or not reference_valid(reference):
            raise SystemExit(f"Reused evidence changed: {path}")
        return
    source = ROOT / c["reuse_from"]
    files = [p for p in source.rglob("*") if p.is_file()]
    target.mkdir(parents=True)
    path.write_text(json.dumps(dict(source=c["reuse_from"],
                                   evidence_hashes={str(p.relative_to(source)): digest(p) for p in sorted(files)},
                                   protocol="matched frozen follow-up; named variants preserve its policy",
                                   original_manifest_sha256=digest(ROOT / "experiment_results/followup_v1/manifest.json"),
                                   compatibility_replay="research/refactor_verification/saved_call_replay.json"), indent=2) + "\n")


def analyze(out):
    subprocess.run([sys.executable, str(ROOT / "analysis/analyze_e2_e3_model_transfer.py"),
                    "--input", str(out), "--output", str(out / "analysis")], check=True)


def cache_snapshot(repo, revision):
    cache = Path(os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")))
    return cache / ("models--" + repo.replace("/", "--")) / "snapshots" / revision


def download(repo, revision, directory, filename=None, metadata=False):
    command = ["hf", "download", repo]
    if filename:
        command.append(filename)
    command += ["--revision", revision, "--local-dir", str(directory)]
    if metadata:
        command += ["--include", "*.json", "*.jinja", "*.txt"]
    with (directory.parent / (directory.name + "_download.log")).open("a") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1800)
    return directory


def pin_server(c, out, port):
    """Choose OLMo precision only on first startup, then keep it across resumes."""
    location = out / "server" / c["model_label"]
    location.mkdir(parents=True, exist_ok=True)
    setup_file = location / "model_setup.json"
    versions = serving_versions()
    if setup_file.exists():
        setup = json.loads(setup_file.read_text())
        if setup["serving_versions"] != versions:
            raise RuntimeError(f"Serving versions changed; use the pinned environment or a fresh study: {setup_file}")
        for filename, expected in setup["artifact_hashes"].items():
            if not Path(filename).is_file() or digest(Path(filename)) != expected:
                raise RuntimeError(f"Pinned model artifact changed or disappeared: {filename}")
        selected = {**c, **setup["overrides"]}
        return ensure_server(selected, port, location), setup
    if list((out / c["model_label"]).rglob("complete.json")) or list((out / "pilots" / c["model_label"]).rglob("complete.json")):
        raise RuntimeError(f"Model setup is missing after inference began; restore {setup_file}")
    if c["model_label"] != "olmo1b-local":
        if served_models(port) is not None:
            raise RuntimeError(f"First startup requires a free port {port} to pin the Qwen weights")
        gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        (location / "gpu.txt").write_text(gpu.stdout + gpu.stderr)
        if gpu.returncode:
            raise RuntimeError(f"GPU unavailable; see {location / 'gpu.txt'}")
        cache = cache_snapshot(c["model"], "").parent
        if not (cache / "refs/main").is_file():
            raise RuntimeError("Qwen model is not cached; restore the original local model before resuming")
        revision = (cache / "refs/main").read_text().strip()
        weights_dir = cache / "snapshots" / revision
        artifacts = [p for p in weights_dir.rglob("*") if p.is_file()]
        if not list(weights_dir.glob("*.safetensors")):
            raise RuntimeError(f"Qwen weights are missing from {weights_dir}")
        overrides = dict(weights=str(weights_dir), server_args=["--tokenizer", str(weights_dir)])
        hashes = {str(p): digest(p) for p in artifacts}
        process = ensure_server({**c, **overrides}, port, location)
        setup = dict(model=c["model"], precision="FP8", revision=revision, serving_versions=versions, overrides=overrides,
                     artifact_hashes=hashes, server="owned")
        setup_file.write_text(json.dumps(setup, indent=2) + "\n")
        return process, setup
    # Do not attempt downloads/fallbacks for a missing GPU or an occupied port.
    gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    (location / "gpu.txt").write_text(gpu.stdout + gpu.stderr)
    if gpu.returncode:
        raise RuntimeError(f"GPU unavailable; see {location / 'gpu.txt'}")
    if served_models(port) is not None:
        raise RuntimeError(f"OLMo requires a free port {port} on its first startup to pin the selected artifact")
    dependency = subprocess.run(["conda", "run", "-n", "vllm", "python", "-c", "import vllm"], capture_output=True, text=True)
    (location / "dependency_check.txt").write_text(dependency.stdout + dependency.stderr)
    if dependency.returncode:
        raise RuntimeError(f"vLLM unavailable; see {location / 'dependency_check.txt'}")
    tokenizer = cache_snapshot(OLMO_MODEL, OLMO_REVISION)
    if not (tokenizer / "tokenizer.json").exists():
        tokenizer = download(OLMO_MODEL, OLMO_REVISION, location / "tokenizer", metadata=True)
    filename = "OLMo-2-0425-1B-Instruct-Q8_0.gguf"
    weights = cache_snapshot(OLMO_MODEL + "-GGUF", OLMO_GGUF_REVISION) / filename
    if not weights.is_file():
        weights = download(OLMO_MODEL + "-GGUF", OLMO_GGUF_REVISION, location / "gguf", filename) / filename
    overrides = dict(weights=str(weights), server_args=["--tokenizer", str(tokenizer),
                     "--hf-config-path", str(tokenizer), "--dtype", "float16"])
    precision, revision = "Q8_0 GGUF", OLMO_GGUF_REVISION
    try:
        process = ensure_server({**c, **overrides}, port, location / "q8")
    except SystemExit as error:
        (location / "q8_startup_failure.txt").write_text(str(error) + "\n")
        weights_dir = cache_snapshot(OLMO_MODEL, OLMO_REVISION)
        if not list(weights_dir.glob("*.safetensors")) and not list(weights_dir.glob("*.bin")):
            weights_dir = download(OLMO_MODEL, OLMO_REVISION, location / "bf16")
        overrides = dict(weights=str(weights_dir), server_args=["--tokenizer", str(tokenizer), "--dtype", "bfloat16"])
        process = ensure_server({**c, **overrides}, port, location / "bf16_server")
        precision, revision = "BF16", OLMO_REVISION
        weights = weights_dir
    artifacts = [weights] if weights.is_file() else [p for p in weights.rglob("*") if p.is_file() and p.suffix in (".safetensors", ".bin")]
    artifacts += [p for p in tokenizer.iterdir() if p.is_file() and p.suffix in (".json", ".jinja", ".txt")]
    setup = dict(model=c["model"], precision=precision, revision=revision, tokenizer_revision=OLMO_REVISION, serving_versions=versions,
                 overrides=overrides, artifact_hashes={str(p): digest(p) for p in artifacts}, server="owned")
    setup_file.write_text(json.dumps(setup, indent=2) + "\n")
    return process, setup


def failure_is_fatal(target):
    work = target.with_name(target.name + ".working")
    records = read_attempts(work / "api_attempts.jsonl")
    return any(a.get("status_code") in (400, 401, 403, 404) for a in records)


def run_study(args):
    invocation_started = time.monotonic()
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    cases = transfer_cases()
    if args.model:
        cases = [c for c in cases if c["model_label"] == args.model]
    if args.condition:
        cases = [c for c in cases if c["condition"] == args.condition]
    if not cases:
        raise SystemExit("No transfer cases match the filters")
    # Verify the reuse candidates before freezing the new schedule.
    for c in cases:
        if c["reuse_from"] and not reusable(c):
            c["replacement_reason"] = "historical control missing or incompatible"
            c["reuse_from"] = ""
    out = (args.out or ROOT / "experiment_results/refactored_v1/model_transfer_v1").resolve()
    labels = [label for label in TRANSFER_MODELS if any(c["model_label"] == label for c in cases)]
    pilots = [c for label in labels for c in transfer_pilots(label)]
    if args.dry_run:
        for c in pilots + cases:
            print(json.dumps(c, sort_keys=True))
            if c["reuse_from"]:
                print("REFERENCE " + str(ROOT / c["reuse_from"]))
            else:
                branch = out / "pilots" if c["pilot"] else out
                target = target_directory(branch / c["model_label"], c)
                command, flags = episode_command(c, target.with_name(target.name + ".working"), args.port)
                print("cd " + shlex.quote(str(ROOT / "BALROG")) + " && " + shlex.join(["env"] + [f"{k}={v}" for k, v in flags.items()] + command))
        print(json.dumps(dict(new_evaluation=sum(not c["reuse_from"] for c in cases),
                              referenced_controls=sum(bool(c["reuse_from"]) for c in cases),
                              excluded_pilots=len(pilots), hours=args.hours, output=str(out),
                              cloud_target_usd=args.cloud_target_usd, cloud_stop_usd=args.cloud_stop_usd)))
        return
    check_manifest(out, pilots + cases)
    for c in cases:
        if c["reuse_from"]:
            reference_episode(out, c)
    analyze(out)  # Also recovers the old CoT panels before any new inference.
    deadline, done = invocation_started + args.hours * 3600, 0
    status_path = out / "stage_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for label in labels:
        selected = [c for c in cases if c["model_label"] == label and not c["reuse_from"]]
        selected_pilots = [c for c in pilots if c["model_label"] == label]
        worklist = [(out / "pilots", c) for c in selected_pilots] + [(out, c) for c in selected]
        server, consecutive_failures = None, 0
        status[label] = dict(state="running", invocation_started=time.time())
        try:
            if all(completed_case(target_directory(branch / label, c), c) for branch, c in worklist):
                status[label].update(state="complete", missing_evaluation=0)
                continue
            if label == "gpt41mini-cloud" and not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is absent; export it before rerunning the same command")
            for branch, c in worklist:
                if time.monotonic() >= deadline or (args.limit and done >= args.limit):
                    status[label].update(state="invocation limit", new_episodes=done)
                    break
                model_out = branch / label
                target = target_directory(model_out, c)
                if completed_case(target, c):
                    continue
                if label == "gpt41mini-cloud":
                    cost = spending(out)
                    if cost["estimated_usd"] > args.cloud_stop_usd:
                        status[label].update(state="cloud spending limit", **cost)
                        break
                    if cost["estimated_usd"] >= args.cloud_target_usd:
                        print(f"Cloud target reached (${cost['estimated_usd']:.3f}); continuing until the estimate exceeds ${args.cloud_stop_usd:g}.", flush=True)
                elif server is None:
                    # An external matching Qwen server returns None; remember startup separately.
                    if not status[label].get("server_ready"):
                        server, setup = pin_server(c, out, args.port)
                        status[label].update(server_ready=True, model_setup=setup)
                try:
                    state, elapsed = run_case(model_out, c, args.port)
                except RuntimeError as error:
                    consecutive_failures += 1
                    with (out / "failures.jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(model_label=label, case=c, error=str(error), at=time.time())) + "\n")
                    print(str(error), flush=True)
                    if c["pilot"] or failure_is_fatal(target) or consecutive_failures >= 3:
                        raise RuntimeError("Stage stopped after a failed preflight, fatal provider error, or three consecutive failed episodes") from error
                    continue
                done += state == "done"
                consecutive_failures = 0
                print(f"{label}: {'pilot' if c['pilot'] else 'evaluation'} {c['condition']} {c['task']} seed {c['seed']}: {state}, {elapsed:.1f}s", flush=True)
                if c["pilot"] and c is selected_pilots[-1]:
                    metadata = [json.loads((target_directory(out / "pilots" / label, p) / "complete.json").read_text()) for p in selected_pilots]
                    mean_time = sum(m["elapsed_seconds"] / m["num_steps"] for m in metadata) / len(metadata)
                    upper_steps = sum(p["budget"] for p in selected)
                    status[label]["pilot_projected_hours"] = mean_time * upper_steps / 3600
                    status[label]["pilot_cost"] = spending(out / "pilots" / label)
                    status[label]["pilot_projected_evaluation_cost_usd"] = (
                        sum((m["input_tokens"] * 0.40 + m["output_tokens"] * 1.60) / 1e6 / m["num_steps"]
                            for m in metadata) / len(metadata) * upper_steps if label == "gpt41mini-cloud" else 0)
                    print(f"{label}: pilot-based ETA {mean_time * upper_steps / 3600:.2f} hours at full action budgets (later context, system load and retries can change this).", flush=True)
                    if label == "gpt41mini-cloud":
                        print(f"Cloud pilot-based evaluation estimate: ${status[label]['pilot_projected_evaluation_cost_usd']:.2f}; actual latency, episode length, and retries can change this.", flush=True)
            else:
                missing = [c for c in selected if not completed_case(target_directory(out / label, c), c)]
                status[label].update(state="complete" if not missing else "incomplete", missing_evaluation=len(missing))
        except KeyboardInterrupt:
            status[label].update(state="interrupted")
            raise
        except (RuntimeError, SystemExit, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            status[label].update(state="blocked", reason=str(error))
            print(f"{label}: blocked: {error}", flush=True)
        finally:
            stop_server(server)
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            analyze(out)
        if time.monotonic() >= deadline or (args.limit and done >= args.limit):
            break
    print(f"Invocation finished: {done} new episodes. Evidence: {out}. Rerun the same command to resume.", flush=True)
    return 0 if all(status.get(label, {}).get("state") == "complete" for label in labels) else 2
