"""Shared response parsing and raw-call logging; no action policy."""

import json
import os
import re
from types import SimpleNamespace


class AgentIO:
    def _field(self, response, name):
        match = re.search(rf"^{name}:\s*(.*)$", response, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else ""


    def _with_action_metadata(self, response, model_action, override_reason):
        return SimpleNamespace(**response._asdict(), model_action=model_action,
                               override_reason=override_reason)


    def _log_call(self, role, messages, response, trigger, model_action="", harness_action="", override_reason=""):
        path = os.environ.get("BALROG_LLM_LOG")
        if path:
            record = {
                "step": self.step_number,
                "role": role,
                "trigger": trigger,
                "messages": [{"role": message.role, "content": message.content} for message in messages],
                "raw_response": response.completion,
                "model_action": model_action,
                "harness_action": harness_action,
                "override_reason": override_reason,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            }
            with open(path, "a") as file:
                file.write(json.dumps(record) + "\n")


