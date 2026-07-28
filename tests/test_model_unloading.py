from __future__ import annotations

from visionsort.core.config import AppConfig
from visionsort.runtime.supervisor import RuntimeSupervisor


class _AcknowledgingQueue:
    def __init__(self, store: dict):
        self.store = store
        self.messages: list[dict] = []

    def put(self, message: dict) -> None:
        self.messages.append(message)
        if message["kind"] == "UNLOAD_MODEL":
            self.store[
                f"__model_unloaded__:{message['operation_id']}"
            ] = {
                "kind": "MODEL_UNLOADED",
                "operation_id": message["operation_id"],
                "model_id": message["model_id"],
                "loaded_model_ids": [],
                "memory_cleanup_called": True,
            }


def _supervisor() -> RuntimeSupervisor:
    supervisor = RuntimeSupervisor.__new__(RuntimeSupervisor)
    supervisor.config = AppConfig(
        values={
            "runtime": {
                "model_unload_timeout_seconds": 0.05,
            }
        }
    )
    supervisor.active_model_ids_by_task = {"detection": "model-new"}
    supervisor.active_runtime_models_by_task = {
        "detection": {
            "task": "detection",
            "model_id": "model-new",
            "generation": 2,
            "activated_at": 2.0,
        }
    }
    supervisor.active_source_pipelines = {}
    supervisor.control_flags = {}
    supervisor.rollback_model_holds = set()
    supervisor.loaded_model_ids = {"model-old", "model-new"}
    supervisor.last_model_unload_error = {}
    supervisor.inference_result_store = {}
    supervisor.inference_request_queue = _AcknowledgingQueue(
        supervisor.inference_result_store
    )
    supervisor.drain_inference_results = lambda: None
    return supervisor


def test_referenced_or_inflight_model_is_not_unloaded():
    supervisor = _supervisor()
    supervisor.active_source_pipelines = {
        "fixed-source": [
            {
                "task": "detection",
                "model_id": "model-old",
                "configured_model_id": "model-old",
                "use_active": False,
            }
        ]
    }
    supervisor.control_flags = {
        "__inflight__:request-1": {
            "model_id": "model-old",
            "source_id": "fixed-source",
        }
    }

    result = supervisor.safe_unload_model("model-old")

    assert result["status"] == "REFERENCED"
    assert result["unloaded"] is False
    assert result["references"]["fixed_sources"] == ["fixed-source"]
    assert supervisor.inference_request_queue.messages == []


def test_model_unloads_after_last_reference_and_confirmation():
    supervisor = _supervisor()

    result = supervisor.safe_unload_model("model-old")

    assert result["status"] == "UNLOADED"
    assert result["unloaded"] is True
    assert "model-old" not in supervisor.loaded_model_ids
    assert supervisor.inference_request_queue.messages[-1]["kind"] == (
        "UNLOAD_MODEL"
    )
    assert result["worker"]["memory_cleanup_called"] is True


def test_unload_timeout_keeps_model_loaded():
    supervisor = _supervisor()
    supervisor.control_flags = {
        "__inflight__:request-1": {
            "model_id": "model-old",
            "source_id": "source-1",
        }
    }

    result = supervisor.safe_unload_model("model-old", timeout=0.03)

    assert result["status"] == "TIMEOUT"
    assert result["unloaded"] is False
    assert "model-old" in supervisor.loaded_model_ids
    assert "model-old" in supervisor.last_model_unload_error
