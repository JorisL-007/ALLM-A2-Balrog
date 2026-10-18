"""Optional request audit for the transfer study; no agent or prompt policy."""
import json
import time
from pathlib import Path

INPUT_USD_PER_MILLION = 0.10
OUTPUT_USD_PER_MILLION = 0.50


def audited_request(client, kwargs, path):
    # Persist the start first: even an interrupted/unknown request remains visible.
    request_id = f"{time.time_ns()}"
    prompt_bound = sum(len(str(m["content"]).encode("utf-8")) + 32 for m in kwargs["messages"])
    maximum_output_tokens = kwargs.get("max_completion_tokens", kwargs.get("max_tokens"))
    bound = (prompt_bound * INPUT_USD_PER_MILLION + maximum_output_tokens * OUTPUT_USD_PER_MILLION) / 1e6
    cloud = kwargs["model"] == "gpt-6-luna"
    started = dict(event="start", request_id=request_id, model=kwargs["model"],
                   started_at=time.time(), maximum_output_tokens=maximum_output_tokens,
                   request=kwargs,
                   uncertain_cost_usd=bound if cloud else 0)

    def append(record):
        with Path(path).open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    append(started)
    begin = time.monotonic()
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as error:
        append(dict(event="error", request_id=request_id, elapsed_seconds=time.monotonic() - begin,
                    error_type=type(error).__name__, status_code=getattr(error, "status_code", None),
                    error=str(error)))
        raise
    usage = response.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    append(dict(event="response", request_id=request_id, elapsed_seconds=time.monotonic() - begin,
                response=response.model_dump(mode="json"), model=response.model,
                usage_available=usage is not None,
                input_tokens=input_tokens, output_tokens=output_tokens,
                cost_usd=(input_tokens * INPUT_USD_PER_MILLION + output_tokens * OUTPUT_USD_PER_MILLION) / 1e6 if cloud else 0))
    return response


def read_attempts(path):
    attempts = {}
    if path.exists():
        for line in path.read_text().splitlines():
            # A killed append may leave one unfinished final line.
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            attempts.setdefault(record["request_id"], {}).update(record)
    return list(attempts.values())


def spending(root):
    records = [attempt for path in root.rglob("api_attempts.jsonl") for attempt in read_attempts(path)]
    known = sum(a.get("cost_usd", 0) for a in records if a.get("event") == "response" and a.get("usage_available"))
    uncertain = sum(a.get("uncertain_cost_usd", 0) for a in records if a.get("event") != "response" or not a.get("usage_available"))
    return dict(known_usd=known, uncertain_usd=uncertain, estimated_usd=known + uncertain,
                attempts=len(records), responses=sum(a.get("event") == "response" for a in records))


def verify_calls(work, context):
    calls = [json.loads(line) for line in (work / "llm_calls.jsonl").read_text().splitlines()]
    used = [a for a in read_attempts(work / "api_attempts.jsonl") if a.get("event") == "response"
            and a["response"].get("choices") and a["response"]["choices"][0]["message"].get("content")]
    return (len(calls) == len(used)
            and all(call["raw_response"].strip() == attempt["response"]["choices"][0]["message"]["content"].strip()
                    and call["input_tokens"] == attempt["input_tokens"]
                    and call["output_tokens"] == attempt["output_tokens"]
                    and (attempt["request"].get("reasoning_effort") == "none"
                         and attempt["request"].get("max_completion_tokens") == 256
                         if attempt["request"]["model"] == "gpt-6-luna" else True)
                    and attempt["input_tokens"] + attempt["maximum_output_tokens"] <= context
                    for call, attempt in zip(calls, used)))
