"""Handle OpenAI Batch API mechanics for model-curation stages."""

from __future__ import annotations

from io import BytesIO
import json

from dotenv import load_dotenv
from openai import OpenAI

from pipeline.common.paths import CURATION_WORK_DIR


TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
STATES_DIR = CURATION_WORK_DIR / "model_curation" / "states"


def sort_key_by_custom_id(result: dict) -> tuple[int, str]:
    """Return a stable sort key for numeric-string and nonnumeric custom ids."""
    custom_id = str(result["custom_id"])
    if custom_id.isdecimal():
        return (0, f"{int(custom_id):020d}")
    return (1, custom_id)


class OpenAIClient:
    """Wrap OpenAI Batch API state, submission, and result collection."""

    def __init__(
        self,
        run_name: str,
        model: str,
        prompt: str,
        schema: dict,
        max_output_tokens: int,
        cache_prompt: bool,
    ) -> None:
        """Initialize a named batch client for one model-curation run."""
        load_dotenv()
        self.run_name = run_name
        self.client = OpenAI()
        self.batch_state_path = STATES_DIR / f"{run_name}_batch_state.json"
        self.batch_state_path.parent.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.prompt = prompt
        self.schema = schema
        self.max_output_tokens = max_output_tokens
        self.cache_prompt = cache_prompt

    def _load_batch_state(self) -> dict | None:
        """Load the stored batch state for this run if it exists."""
        if not self.batch_state_path.exists():
            return None

        with self.batch_state_path.open("r", encoding="utf-8") as file:
            batch_state = json.load(file)

        return batch_state or None

    def _save_batch(self, batch) -> None:
        """Persist a raw OpenAI batch object as the current state."""
        batch_dict = batch.model_dump()

        with self.batch_state_path.open("w", encoding="utf-8") as file:
            json.dump(batch_dict, file, ensure_ascii=False, indent=2)

    def clean_batch_state(self) -> None:
        """Delete the stored batch state after successful collection."""
        self.batch_state_path.unlink(missing_ok=True)

    def _prepare_for_new_batch(self) -> None:
        """Ensure there is no active or fatal batch state before submission."""
        batch_state = self.get_current_batch_state()
        if not batch_state:
            return

        status = batch_state["status"]
        if status == "completed":
            self.clean_batch_state()
            return

        if status not in TERMINAL_STATUSES:
            raise RuntimeError("There is a batch being processed. Rerun this script later.")

        raise RuntimeError(
            f"Fatal batch state error: current batch status is {status!r}. "
            f"Check it and when you want to continue, delete {self.batch_state_path} "
            "manually and try again.",
        )

    def get_current_batch_state(self) -> dict | None:
        """Return the current batch state, refreshing active batches from OpenAI."""
        batch_state = self._load_batch_state()
        if not batch_state:
            return None

        if batch_state["status"] not in TERMINAL_STATUSES:
            batch = self.client.batches.retrieve(batch_state["id"])
            self._save_batch(batch)
            batch_state = batch.model_dump()

        status = batch_state["status"]

        if status not in TERMINAL_STATUSES:
            counts = batch_state.get("request_counts")
            return {
                "id": batch_state["id"],
                "status": status,
                "counts": {
                    "total": counts["total"],
                    "completed": counts["completed"],
                    "failed": counts["failed"],
                } if counts else None,
            }

        return self._process_terminal_batch(batch_state)

    def _process_terminal_batch(self, batch_state: dict) -> dict:
        """Normalize a terminal OpenAI batch state for stage runners."""
        status = batch_state["status"]

        result = {
            "id": batch_state["id"],
            "status": status,
            "counts": batch_state.get("request_counts"),
            "output_file_id": batch_state.get("output_file_id"),
            "error_file_id": batch_state.get("error_file_id"),
        }

        if status == "completed":
            result["message"] = "Batch finished successfully."
        elif status == "failed":
            result["message"] = "Batch failed to process."
            result["batch_errors"] = batch_state.get("errors")
        elif status in {"expired", "cancelled"}:
            result["message"] = f"Batch was {status}."

        return result

    def _build_batch_request(self, custom_id: str, user_input: str) -> dict:
        """Build one JSONL request row for the Responses Batch API."""
        body = {
            "model": self.model,
            "input": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": user_input},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": self.run_name,
                    "strict": True,
                    "schema": self.schema,
                },
            },
            "max_output_tokens": self.max_output_tokens,
        }
        if self.cache_prompt:
            body["prompt_cache_key"] = f"amlk-{self.run_name}"

        return {
            "custom_id": str(custom_id),
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }

    def submit_batch(self, batch_inputs: list[tuple[str, str]]) -> str:
        """Submit a new batch from custom-id and user-input pairs."""
        self._prepare_for_new_batch()

        jsonl = ""
        for custom_id, user_input in batch_inputs:
            request = self._build_batch_request(custom_id, user_input)
            jsonl += json.dumps(request, ensure_ascii=False) + "\n"

        file = BytesIO(jsonl.encode("utf-8"))
        file.name = f"{self.run_name}_batch_input.jsonl"

        input_file = self.client.files.create(file=file, purpose="batch")

        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )

        self._save_batch(batch)
        return batch.id

    def get_last_batch_results(self) -> list[dict]:
        """Fetch and parse the result files for the completed current batch."""
        batch_state = self.get_current_batch_state()
        if not batch_state:
            raise ValueError("No batch state file found.")

        if batch_state["status"] != "completed":
            raise ValueError(f"Batch is {batch_state['status']}, not completed.")
        if not batch_state["output_file_id"] and not batch_state["error_file_id"]:
            raise ValueError("Completed batch has no output_file_id or error_file_id.")

        results = []
        if batch_state["output_file_id"]:
            results.extend(
                self._parse_output_file(
                    batch_state["output_file_id"],
                    self._parse_success_output_line,
                ),
            )

        if batch_state["error_file_id"]:
            results.extend(
                self._parse_output_file(
                    batch_state["error_file_id"],
                    self._parse_error_output_line,
                ),
            )

        return sorted(results, key=sort_key_by_custom_id)

    def _parse_output_file(self, file_id: str, parse_line) -> list[dict]:
        """Download one OpenAI output file and parse each nonempty line."""
        file_response = self.client.files.content(file_id)
        return [
            parse_line(line)
            for line in file_response.text.splitlines()
            if line.strip()
        ]

    def _parse_success_output_line(self, line: str) -> dict:
        """Parse one successful batch-output line into a normalized result."""
        item = json.loads(line)
        custom_id = item["custom_id"]
        response = item["response"]
        status_code = response["status_code"]
        body = response["body"]
        if status_code < 200 or status_code >= 300:
            return {
                "custom_id": custom_id,
                "ok": False,
                "error": {
                    "code": f"http_{status_code}",
                    "message": json.dumps(body, ensure_ascii=False),
                },
            }

        if body.get("status") and body["status"] != "completed":
            return {
                "custom_id": custom_id,
                "ok": False,
                "error": {
                    "code": body["status"],
                    "message": json.dumps(
                        body.get("incomplete_details") or body.get("error"),
                        ensure_ascii=False,
                    ),
                },
            }

        try:
            data = json.loads(self._response_output_text(body))
        except (KeyError, IndexError, ValueError) as error:
            return {
                "custom_id": custom_id,
                "ok": False,
                "error": {
                    "code": "parse_error",
                    "message": str(error),
                },
            }

        return {"custom_id": custom_id, "ok": True, "data": data}

    def _parse_error_output_line(self, line: str) -> dict:
        """Parse one error-file line into a normalized failed result."""
        item = json.loads(line)
        return {
            "custom_id": item["custom_id"],
            "ok": False,
            "error": item["error"],
        }

    def _response_output_text(self, body: dict) -> str:
        """Extract the model output text from a Responses API body."""
        for item in body["output"]:
            if item["type"] == "message":
                return item["content"][0]["text"]

        raise ValueError("Could not find model output text in batch response body")
