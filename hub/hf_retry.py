import time
from collections.abc import Callable
from typing import TypeVar

from huggingface_hub.errors import HfHubHTTPError

T = TypeVar("T")


def call_with_retry(
    action: Callable[[], T],
    label: str,
    max_attempts: int = 12,
) -> T:
    attempt = 0
    while True:
        attempt += 1
        try:
            return action()
        except HfHubHTTPError as exc:
            response = exc.response
            if response is None or response.status_code != 429:
                raise
            if attempt >= max_attempts:
                raise
            retry_after = int(response.headers.get("Retry-After", 300))
            wait_s = max(retry_after, 180)
            print(
                f"Hugging Face rate limit on {label}. "
                f"Attempt {attempt}/{max_attempts}. Waiting {wait_s}s."
            )
            time.sleep(wait_s)


def ensure_repo(
    api,
    repo_id: str,
    repo_type: str,
    token: str | None,
) -> None:
    exists = call_with_retry(
        lambda: api.repo_exists(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
        ),
        f"repo_exists {repo_id}",
    )
    if exists:
        return
    call_with_retry(
        lambda: api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            exist_ok=True,
            token=token,
        ),
        f"create_repo {repo_id}",
    )
