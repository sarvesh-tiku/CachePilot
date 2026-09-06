class CachePilotError(Exception):
    """Base class for all CachePilot domain errors."""


class WorkerNotFoundError(CachePilotError):
    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id
        super().__init__(f"worker not found: {worker_id}")


class WorkerAlreadyRegisteredError(CachePilotError):
    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id
        super().__init__(f"worker already registered: {worker_id}")


class NoHealthyWorkersError(CachePilotError):
    def __init__(self) -> None:
        super().__init__("no healthy workers available")


class UnknownPolicyError(CachePilotError):
    def __init__(self, policy: str, known: list[str]) -> None:
        self.policy = policy
        super().__init__(f"unknown scheduler policy {policy!r}; known: {', '.join(known)}")
