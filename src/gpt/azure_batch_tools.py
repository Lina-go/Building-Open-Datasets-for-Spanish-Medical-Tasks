#############################
# azure_batch_tools.py
#############################

import os
import json
from openai import AzureOpenAI
from src.gpt.prompting import build_prompt

def write_jsonl(lines, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")


def generate_jsonl(texts, strategy, system_prompt, deployment_name, output_path,
                   temperature=0.0, top_p=1.0, max_tokens=350):
    lines = []
    for i, text in enumerate(texts):
        prompt = build_prompt(text, strategy)
        body = {
            "model": deployment_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        lines.append({
            "custom_id": f"task-{i}",
            "method": "POST",
            "url": "/chat/completions",
            "body": body
        })
    write_jsonl(lines, output_path)
    return output_path


def create_file(client, path):
    with open(path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    return file_obj.id


def create_batch_job(client, input_file_id, endpoint="/chat/completions", completion_window="24h", metadata=None):
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=metadata or {}
    )
    return batch.id


def poll_batch_until_done(client, batch_id, sleep_seconds=30, max_polls=200):
    import time
    polls = 0
    while polls < max_polls:
        info = client.batches.retrieve(batch_id)
        status = info.status
        out_id = getattr(info, "output_file_id", None)
        err_id = getattr(info, "error_file_id", None)
        print(f"Batch {batch_id} -> {status}")
        if status in ("completed", "failed", "cancelled", "expired"):
            return status, out_id, err_id
        polls += 1
        time.sleep(sleep_seconds)
    return "timeout", None, None


def download_bytes(client, file_id):
    stream = client.files.content(file_id)
    return stream.read()


def parse_batch_output(raw_bytes):
    out = {}
    text = raw_bytes.decode("utf-8")
    for line in text.splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        cid = obj.get("custom_id")
        resp = obj.get("response", {})
        body = resp.get("body", {})
        choices = body.get("choices", [])
        content = None
        if choices and "message" in choices[0] and "content" in choices[0]["message"]:
            content = choices[0]["message"]["content"]
        usage = body.get("usage", {})
        if cid is not None:
            out[cid] = {"content": content, "usage": usage, "raw": body}
    return out
