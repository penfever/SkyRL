import os
import asyncio
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
        WeightSyncInitInfo,
    )
from dataclasses import dataclass
from loguru import logger
from http import HTTPStatus
from types import SimpleNamespace
from uuid import uuid4

import ray
import vllm
from loguru import logger
from packaging import version
from vllm import SamplingParams
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
    CompletionResponse,
)
from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.entrypoints.openai.models.serving import BaseModelPath, OpenAIServingModels
from vllm.inputs import TokensPrompt
from vllm.v1.metrics.loggers import LoggingStatLogger
from vllm.lora.request import LoRARequest

from skyrl.backends.skyrl_train.inference_engines.base import (
    InferenceEngineInput,
    InferenceEngineInterface,
    InferenceEngineOutput,
)
from skyrl.backends.skyrl_train.inference_engines.vllm.utils import pop_openai_kwargs

# Backward compatibility: WorkerWrap has moved to inference_servers.vllm_worker
# This alias preserves the old import path for existing scripts/configs.
# TODO (Kourosh): Remove this alias once all references are updated.
from skyrl.backends.skyrl_train.inference_servers.vllm_worker import (
    WorkerWrap,  # noqa: F401, E402
)
from skyrl.backends.skyrl_train.weight_sync import WeightLoader, WeightUpdateRequest


@dataclass
class Logprob:
    logprob: float
    rank: int
    token_id: str


def setup_envvars_for_vllm(kwargs, bundle_indices):
    noset_visible_devices = kwargs.pop("noset_visible_devices")
    mp_cuda_visible_devices = kwargs.pop("mp_cuda_visible_devices", None)
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # TODO(Charlie): may not be needed.

    # When custom all-reduce is disabled (e.g. for TP=2 on H100 where
    # SymmMemCommunicator rendezvous fails), also disable symmetric memory
    # via env var — the engine arg alone doesn't prevent SymmMemCommunicator
    # from being instantiated.
    if kwargs.get("disable_custom_all_reduce"):
        os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
        logger.info("setup_envvars_for_vllm: set VLLM_ALLREDUCE_USE_SYMM_MEM=0 (disable_custom_all_reduce=True)")

    if kwargs.get("distributed_executor_backend") == "mp" and mp_cuda_visible_devices is not None:
        # For mp backend in colocated mode, set CUDA_VISIBLE_DEVICES to the
        # pre-computed GPU IDs for this engine so spawned workers see the
        # correct GPUs (not all GPUs on the node).
        os.environ["CUDA_VISIBLE_DEVICES"] = mp_cuda_visible_devices
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        logger.info(f"mp backend: setting CUDA_VISIBLE_DEVICES={mp_cuda_visible_devices}")
    elif kwargs.get("distributed_executor_backend") in ("ray", "mp"):
        # For ray backend (and non-colocate mp), clear CUDA_VISIBLE_DEVICES
        # so vLLM workers can discover GPUs via their own scheduling.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
    elif noset_visible_devices:
        # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
        # when the distributed_executor_backend is not ray/mp and
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

    num_gpus = kwargs.pop("num_gpus")
    if bundle_indices is not None:
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        logger.info(f"creating LLM with bundle_indices={bundle_indices}")

    # Set NUMA CPU affinity for single-GPU (TP=1) inference actors.
    # For TP>1, affinity is set per-worker via WorkerWrap.set_numa_affinity().
    #
    # When NUMA affinity is enabled, we also disable vLLM V1 multiprocessing.
    # vLLM's V1 engine spawns EngineCore as a separate subprocess using
    # multiprocessing with start_method="spawn" (forced when running inside a
    # Ray actor — see vllm.utils._maybe_force_spawn). Spawned processes do NOT
    # inherit the parent's CPU affinity, so NUMA binding set here would be lost.
    # Disabling V1 multiprocessing forces EngineCore to run in the same process,
    # where our affinity settings take effect.
    executor_backend = kwargs.get("distributed_executor_backend")
    logger.info(
        f"setup_envvars_for_vllm: distributed_executor_backend={executor_backend}, "
        f"SKYRL_ENABLE_NUMA_AFFINITY={os.environ.get('SKYRL_ENABLE_NUMA_AFFINITY', '<unset>')}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}, "
        f"VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '<unset>')}"
    )
    if executor_backend != "ray":
        try:
            from skyrl_train.utils.numa import is_numa_affinity_enabled, set_numa_affinity_for_gpu
            numa_enabled = is_numa_affinity_enabled()
            logger.info(f"setup_envvars_for_vllm: numa_enabled={numa_enabled}")
            if numa_enabled:
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
                logger.info("setup_envvars_for_vllm: set VLLM_ENABLE_V1_MULTIPROCESSING=0 for NUMA affinity")
                cuda_devs = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                if cuda_devs:
                    gpu_ids = [int(x) for x in cuda_devs.split(",")]
                    if len(gpu_ids) == 1:
                        set_numa_affinity_for_gpu(gpu_ids[0])
        except Exception as e:
            logger.warning(f"setup_envvars_for_vllm: NUMA affinity setup failed: {e}")


class BaseVLLMInferenceEngine(InferenceEngineInterface):
    """Base class containing shared logic between sync and async VLLM engines."""

    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        # Redirect infrastructure output to log file before any engine initialization.
        # Done here in the base class so all subclasses get it automatically.
        from skyrl.train.utils.ray_logging import redirect_actor_output_to_file

        redirect_actor_output_to_file()

        setup_envvars_for_vllm(kwargs, bundle_indices)
        vllm_v1_disable_multiproc = kwargs.pop("vllm_v1_disable_multiproc", False)
        logger.info(
            f"BaseVLLMInferenceEngine: vllm_v1_disable_multiproc={vllm_v1_disable_multiproc}, "
            f"vllm.__version__={vllm.__version__}, "
            f"VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '<unset>')}"
        )
        if vllm_v1_disable_multiproc or vllm.__version__ == "0.8.2":
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("BaseVLLMInferenceEngine: set VLLM_ENABLE_V1_MULTIPROCESSING=0")

        # Store common attributes
        self._tp_size = kwargs.get("tensor_parallel_size", 1)
        self._pp_size = kwargs.get("pipeline_parallel_size", 1)
        self._dp_size = kwargs.get("data_parallel_size", 1)
        self._is_lora = kwargs.get("enable_lora", False)

        # Let subclass create the appropriate engine
        self.llm = self._create_engine(*args, **kwargs)

        # Set NUMA affinity for TP>1 workers via collective_rpc
        if self._tp_size > 1 or self._pp_size > 1:
            try:
                self.llm.collective_rpc("set_numa_affinity")
            except Exception:
                pass

        # Weight loader is created by subclass after engine initialization
        self._weight_loader = None

    def tp_size(self):
        return self._tp_size

    def pp_size(self):
        return self._pp_size

    def dp_size(self):
        return self._dp_size

    def _create_engine(self, *args, **kwargs):
        """Abstract method for subclasses to implement engine creation."""
        raise NotImplementedError("Subclasses must implement _create_engine")

    def _preprocess_prompts(self, input_batch: InferenceEngineInput):
        """Common prompt preprocessing logic."""
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        assert (
            prompts is None and prompt_token_ids is not None
        ), "VLLMInferenceEngine only accepts `prompt_token_ids`, not `prompts`."

        sampling_params = (
            SamplingParams(**request_sampling_params) if request_sampling_params is not None else SamplingParams()
        )

        return prompt_token_ids, sampling_params

    def _postprocess_outputs(self, outputs):
        """Common output processing logic."""
        responses: List[str] = []
        stop_reasons: List[str] = []
        response_ids: List[List[int]] = []
        response_logprobs: Optional[List[List[float]]] = []
        rollout_expert_indices: Optional[List[List[List[List[int]]]]] = []
        all_prompt_logprobs: Optional[List] = None

        for output in outputs:
            # TODO(tgriggs): Support n>1 sampling.
            assert (
                len(output.outputs) == 1
            ), "Each prompt should have only one responses. n>1 sampling is supported by copying prompts."
            resp = output.outputs[0]
            responses.append(resp.text)
            stop_reasons.append(resp.finish_reason)
            response_ids.append(resp.token_ids)
            _logprobs = None
            if resp.logprobs:
                _logprobs = []
                for i, token_logprobs in enumerate(resp.logprobs):
                    token_logprobs: Dict[str, Logprob]
                    token_id = resp.token_ids[i]
                    logprob = token_logprobs[token_id].logprob
                    _logprobs.append(logprob)
                    del token_logprobs
            response_logprobs.append(_logprobs)

            _routed_experts = None
            if resp.routed_experts is not None:
                if hasattr(resp.routed_experts, "tolist"):
                    _routed_experts = resp.routed_experts.tolist()
                else:
                    _routed_experts = resp.routed_experts
            rollout_expert_indices.append(_routed_experts)

            # Extract prompt_logprobs if available (used for teacher scoring)
            if hasattr(output, "prompt_logprobs") and output.prompt_logprobs is not None:
                if all_prompt_logprobs is None:
                    all_prompt_logprobs = []
                # Convert vLLM's List[Optional[Dict[int, Logprob]]] to
                # List[Optional[Dict[int, float]]] (extract .logprob from Logprob objects)
                prompt_lps = []
                for pos_logprobs in output.prompt_logprobs:
                    if pos_logprobs is None:
                        prompt_lps.append(None)
                    else:
                        prompt_lps.append({
                            token_id: lp.logprob if hasattr(lp, "logprob") else lp
                            for token_id, lp in pos_logprobs.items()
                        })
                all_prompt_logprobs.append(prompt_lps)

        if len(response_logprobs) and response_logprobs[0] is None:
            response_logprobs = None  # hack: assume uniform sampling params

        if len(rollout_expert_indices) > 0 and rollout_expert_indices[0] is None:
            rollout_expert_indices = None  # hack: assume uniform sampling params

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
            response_ids=response_ids,
            response_logprobs=response_logprobs,
            rollout_expert_indices=rollout_expert_indices,
            prompt_logprobs=all_prompt_logprobs,
        )

    def _get_engine(self):
        """Get the underlying engine for RPC calls."""
        return self.llm.engine if hasattr(self.llm, "engine") else self.llm

    @staticmethod
    def _get_unfinished_request_ids(output_processor) -> list:
        """Get unfinished request IDs suitable for abort/abort_request calls.

        In vllm 0.16.0+, request_states is keyed by internal IDs (with a random suffix),
        while abort() expects external IDs by default. We use external_req_ids when
        available and fall back to request_states keys for older vllm versions.
        """
        if hasattr(output_processor, "external_req_ids"):
            return list(output_processor.external_req_ids.keys())
        return list(output_processor.request_states.keys())

    def reset_prefix_cache(self):
        """Reset the prefix cache. Subclasses override for async version."""
        return self.llm.llm_engine.reset_prefix_cache()

    async def pause_generation(self, clear_cache: bool = False) -> None:
        raise NotImplementedError("pause_generation is only supported for AsyncVLLMInferenceEngine.")

    async def resume_generation(self) -> None:
        raise NotImplementedError("resume_generation is only supported for AsyncVLLMInferenceEngine.")


class VLLMInferenceEngine(BaseVLLMInferenceEngine):
    """Synchronous VLLM engine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_loader = VLLMWeightLoader(self.llm, is_async=False)

    def _create_engine(self, *args, **kwargs):
        # Pipeline parallelism requires AsyncLLMEngine
        if kwargs.get("pipeline_parallel_size", 1) > 1:
            raise ValueError(
                "Pipeline parallelism is only supported with AsyncVLLMInferenceEngine. "
                "Please set `generator.async_engine=true` in your config."
            )
        # Pop enable_ray_prometheus_stats - only supported for async engine
        enable_ray_prometheus_stats = kwargs.pop("enable_ray_prometheus_stats", False)
        if enable_ray_prometheus_stats:
            logger.warning(
                "enable_ray_prometheus_stats is only supported with AsyncVLLMInferenceEngine. "
                "Set `generator.async_engine=true` to enable Ray Prometheus stats logging."
            )
        return vllm.LLM(*args, **kwargs)

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        prompt_token_ids, sampling_params = self._preprocess_prompts(input_batch)

        # Check if LoRA is enabled and create LoRA requests
        lora_requests = None
        if self._is_lora:
            lora_int_ids = list(self.llm.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                batch_size = len(prompt_token_ids)
                # dummy_lora_path for placeholder (actual loading done in add_lora())
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/dummy_lora_path")
                ] * batch_size

        outputs = await asyncio.to_thread(
            self.llm.generate,
            prompts=[TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids],
            sampling_params=sampling_params,
            lora_request=lora_requests,
        )

        return self._postprocess_outputs(outputs)

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Only supported in AsyncVLLMInferenceEngine."""
        raise NotImplementedError("`chat_completion` is only supported in AsyncVLLMInferenceEngine.")

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Only supported in AsyncVLLMInferenceEngine."""
        raise NotImplementedError("`completion` is only supported in AsyncVLLMInferenceEngine.")

    async def wake_up(self, *args: Any, **kwargs: Any):
        await asyncio.to_thread(self.llm.wake_up, tags=kwargs.get("tags", None))

    async def sleep(self, *args: Any, **kwargs: Any):
        engine = self._get_engine().llm_engine
        output_processor = engine.output_processor
        if output_processor.has_unfinished_requests():
            logger.warning(
                "Calling sleep() with unfinished requests in vLLM engine. This is unexpected since all "
                "generation should be done before sleep() is called. Check for potential failures or "
                "dangling requests in your Generator/Env. Aborting all unfinished requests."
            )
            unfinished_request_ids = self._get_unfinished_request_ids(output_processor)
            await asyncio.to_thread(engine.abort_request, unfinished_request_ids)

        level = 1 if self._is_lora else kwargs.get("level", 2)
        await asyncio.to_thread(self.llm.sleep, level=level)

    async def init_weight_update_communicator(self, init_info: "WeightSyncInitInfo"):
        import pickle

        engine = self._get_engine()
        # Pickle the init_info to preserve type through collective_rpc
        pickled_init_info = pickle.dumps(init_info)
        return await asyncio.to_thread(
            engine.collective_rpc,
            "init_weight_update_communicator",
            args=(pickled_init_info,),
        )

    async def _load_lora_from_disk(self, lora_path: str):
        """Load LoRA adapters from disk using vLLM's native add_lora method."""
        lora_id = int(time.time_ns() % 0x7FFFFFFF)
        lora_request = LoRARequest(lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=lora_path)
        result = self.llm.llm_engine.add_lora(lora_request)
        return result

    async def update_named_weights(self, request: WeightUpdateRequest):
        from skyrl.backends.skyrl_train.weight_sync import LoraLoadRequest

        # Handle LoRA disk loading request
        if isinstance(request, LoraLoadRequest):
            return await self._load_lora_from_disk(request.lora_path)

        if not len(request):
            raise ValueError("Weight update request must not be empty")

        # Use the weight loader to coordinate weight transfer
        return await self._weight_loader.load_weights(request)

    async def teardown(self):
        await self._teardown_weight_receiver()

    async def reset_prefix_cache(self):
        return await asyncio.to_thread(self.llm.llm_engine.reset_prefix_cache)

    async def _teardown_weight_receiver(self):
        engine = self._get_engine()
        return await asyncio.to_thread(engine.collective_rpc, "teardown_weight_receiver")

class V1LoggingStatLoggerFixed(LoggingStatLogger):
    """
    A fixed version of LoggingStatLogger that actually logs during the record method.
    The log method is otherwise not called in the VLLM codebase.

    Also stores aggregated stats in a class-level registry for programmatic access
    (used by VLLMStatsCallback to bypass Ray log-to-driver unreliability).

    Stats are accumulated throughout a step:
    - Request counts (running, waiting): track peak and median values
    - Throughput metrics: track peak and median values observed during active periods
    - Cache metrics: track peak and median usage
    """

    # Class-level registry mapping engine IDs to their accumulated stats
    _stats_registry: Dict[int, Dict[str, Any]] = {}
    _registry_lock = threading.Lock()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.log_interval = 5
        self._engine_id: Optional[int] = None

    def set_engine_id(self, engine_id: int) -> None:
        """Set the engine ID for this stat logger instance."""
        self._engine_id = engine_id

    def record(self, *args: Any, **kwargs: Any) -> None:
        # Call parent with original arguments - important to preserve vLLM's calling convention
        super().record(*args, **kwargs)

        # Accumulate stats in registry if engine ID is set
        if self._engine_id is not None:
            # Extract scheduler_stats from vLLM v1 API:
            # vLLM calls record(scheduler_stats, iteration_stats, ...) with positional args
            # or record(scheduler_stats=..., iteration_stats=...) with keyword args
            scheduler_stats = None
            if args:
                scheduler_stats = args[0]
            elif "scheduler_stats" in kwargs:
                scheduler_stats = kwargs["scheduler_stats"]

            current_running = 0
            current_waiting = 0
            current_cache_usage = 0.0
            current_prefix_hit = 0.0

            if scheduler_stats is not None:
                current_running = getattr(scheduler_stats, "num_running_reqs", 0)
                current_waiting = getattr(scheduler_stats, "num_waiting_reqs", 0)
                current_cache_usage = getattr(scheduler_stats, "kv_cache_usage", 0.0) * 100.0  # Convert to percentage

                # Extract prefix cache hit rate from prefix_cache_stats
                prefix_cache_stats = getattr(scheduler_stats, "prefix_cache_stats", None)
                if prefix_cache_stats is not None:
                    hits = getattr(prefix_cache_stats, "hits", 0)
                    misses = getattr(prefix_cache_stats, "misses", 0)
                    total = hits + misses
                    current_prefix_hit = (hits / total * 100.0) if total > 0 else 0.0

            # Extract iteration_stats (second positional arg) for per-request latency data
            iteration_stats = None
            if len(args) > 1:
                iteration_stats = args[1]
            elif "iteration_stats" in kwargs:
                iteration_stats = kwargs["iteration_stats"]

            # Collect per-request latency samples from finished requests
            finished_prefill_times: List[float] = []
            finished_decode_times: List[float] = []
            finished_e2e_latencies: List[float] = []
            finished_queued_times: List[float] = []
            finished_ttfts: List[float] = []
            finished_num_preempted = 0
            if iteration_stats is not None:
                # Time-to-first-token samples from this iteration
                ttft_iter = getattr(iteration_stats, "time_to_first_tokens_iter", None)
                if ttft_iter:
                    finished_ttfts.extend(ttft_iter)
                # Preemption count
                finished_num_preempted = getattr(iteration_stats, "num_preempted_reqs", 0)
                # Per-request stats from completed requests
                for req in getattr(iteration_stats, "finished_requests", []):
                    prefill_t = getattr(req, "prefill_time", 0.0)
                    decode_t = getattr(req, "decode_time", 0.0)
                    e2e_t = getattr(req, "e2e_latency", 0.0)
                    queued_t = getattr(req, "queued_time", 0.0)
                    if prefill_t > 0:
                        finished_prefill_times.append(prefill_t)
                    if decode_t > 0:
                        finished_decode_times.append(decode_t)
                    if e2e_t > 0:
                        finished_e2e_latencies.append(e2e_t)
                    if queued_t > 0:
                        finished_queued_times.append(queued_t)

            # Throughput is computed by parent class LoggingStatLogger after super().record()
            # These are stored as instance attributes
            current_prompt_tp = getattr(self, "last_prompt_throughput", 0.0) or 0.0
            current_gen_tp = getattr(self, "last_generation_throughput", 0.0) or 0.0

            is_active = current_running > 0 or current_waiting > 0

            with V1LoggingStatLoggerFixed._registry_lock:
                existing = V1LoggingStatLoggerFixed._stats_registry.get(self._engine_id)

                if existing is None:
                    # Initialize with sample lists for median calculation
                    V1LoggingStatLoggerFixed._stats_registry[self._engine_id] = {
                        # Sample lists for computing median (only active samples)
                        "_samples_prompt_tp": [current_prompt_tp] if is_active else [],
                        "_samples_gen_tp": [current_gen_tp] if is_active else [],
                        "_samples_running": [current_running] if is_active else [],
                        "_samples_waiting": [current_waiting] if is_active else [],
                        "_samples_cache": [current_cache_usage] if is_active else [],
                        "_samples_prefix_hit": [current_prefix_hit] if is_active else [],
                        # Per-request latency samples (accumulated from finished requests)
                        "_samples_prefill_time": list(finished_prefill_times),
                        "_samples_decode_time": list(finished_decode_times),
                        "_samples_e2e_latency": list(finished_e2e_latencies),
                        "_samples_queued_time": list(finished_queued_times),
                        "_samples_ttft": list(finished_ttfts),
                        "_total_preempted": finished_num_preempted,
                        # Peak values
                        "_peak_prompt_tp": current_prompt_tp,
                        "_peak_gen_tp": current_gen_tp,
                        "_peak_running": current_running,
                        "_peak_waiting": current_waiting,
                        "_peak_cache": current_cache_usage,
                        "_peak_prefix_hit": current_prefix_hit,
                        # Counters
                        "_num_samples": 1,
                        "_num_active_samples": 1 if is_active else 0,
                        "timestamp": time.time(),
                    }
                else:
                    # Update peak values
                    existing["_peak_prompt_tp"] = max(existing["_peak_prompt_tp"], current_prompt_tp)
                    existing["_peak_gen_tp"] = max(existing["_peak_gen_tp"], current_gen_tp)
                    existing["_peak_running"] = max(existing["_peak_running"], current_running)
                    existing["_peak_waiting"] = max(existing["_peak_waiting"], current_waiting)
                    existing["_peak_cache"] = max(existing["_peak_cache"], current_cache_usage)
                    existing["_peak_prefix_hit"] = max(existing["_peak_prefix_hit"], current_prefix_hit)

                    # Accumulate per-request latency samples
                    existing["_samples_prefill_time"].extend(finished_prefill_times)
                    existing["_samples_decode_time"].extend(finished_decode_times)
                    existing["_samples_e2e_latency"].extend(finished_e2e_latencies)
                    existing["_samples_queued_time"].extend(finished_queued_times)
                    existing["_samples_ttft"].extend(finished_ttfts)
                    existing["_total_preempted"] += finished_num_preempted

                    # Append to sample lists (only for active samples to get meaningful medians)
                    if is_active:
                        existing["_samples_prompt_tp"].append(current_prompt_tp)
                        existing["_samples_gen_tp"].append(current_gen_tp)
                        existing["_samples_running"].append(current_running)
                        existing["_samples_waiting"].append(current_waiting)
                        existing["_samples_cache"].append(current_cache_usage)
                        existing["_samples_prefix_hit"].append(current_prefix_hit)
                        existing["_num_active_samples"] += 1

                    existing["_num_samples"] += 1
                    existing["timestamp"] = time.time()

        now = time.monotonic()
        if now - self.last_log_time > self.log_interval:
            self.log()
            self.last_log_time = now

    @staticmethod
    def _compute_median(samples: List[float]) -> float:
        """Compute median of a list of samples."""
        if not samples:
            return 0.0
        sorted_samples = sorted(samples)
        n = len(sorted_samples)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_samples[mid - 1] + sorted_samples[mid]) / 2.0
        return sorted_samples[mid]

    @classmethod
    def get_stats_by_engine_id(cls, engine_id: int, reset: bool = True) -> Optional[Dict[str, Any]]:
        """Get the accumulated stats for a given engine ID.

        Args:
            engine_id: The engine ID to get stats for.
            reset: If True, reset the accumulated stats after reading (default True).
                   This ensures each training step gets fresh stats.

        Returns:
            Dict with accumulated stats, or None if no stats recorded yet.
            Includes peak values, median values, and computed averages.
        """
        with cls._registry_lock:
            stats = cls._stats_registry.get(engine_id)
            if stats is None:
                return None

            # Compute medians from sample lists
            median_prompt_tp = cls._compute_median(stats["_samples_prompt_tp"])
            median_gen_tp = cls._compute_median(stats["_samples_gen_tp"])
            median_running = cls._compute_median(stats["_samples_running"])
            median_waiting = cls._compute_median(stats["_samples_waiting"])
            median_cache = cls._compute_median(stats["_samples_cache"])
            median_prefix_hit = cls._compute_median(stats["_samples_prefix_hit"])

            # Compute means from sample lists
            num_active = stats["_num_active_samples"]
            if num_active > 0:
                mean_prompt_tp = sum(stats["_samples_prompt_tp"]) / num_active
                mean_gen_tp = sum(stats["_samples_gen_tp"]) / num_active
            else:
                mean_prompt_tp = 0.0
                mean_gen_tp = 0.0

            # Compute per-request latency statistics
            prefill_samples = stats["_samples_prefill_time"]
            decode_samples = stats["_samples_decode_time"]
            e2e_samples = stats["_samples_e2e_latency"]
            queued_samples = stats["_samples_queued_time"]
            ttft_samples = stats["_samples_ttft"]

            def _mean(s: List[float]) -> float:
                return sum(s) / len(s) if s else 0.0

            def _p90(s: List[float]) -> float:
                if not s:
                    return 0.0
                sorted_s = sorted(s)
                idx = int(len(sorted_s) * 0.9)
                return sorted_s[min(idx, len(sorted_s) - 1)]

            result = {
                # Peak values
                "peak_prompt_throughput": stats["_peak_prompt_tp"],
                "peak_generation_throughput": stats["_peak_gen_tp"],
                "peak_running_reqs": stats["_peak_running"],
                "peak_waiting_reqs": stats["_peak_waiting"],
                "peak_gpu_cache_usage_perc": stats["_peak_cache"],
                "peak_prefix_cache_hit_rate": stats["_peak_prefix_hit"],
                # Median values
                "median_prompt_throughput": median_prompt_tp,
                "median_generation_throughput": median_gen_tp,
                "median_running_reqs": median_running,
                "median_waiting_reqs": median_waiting,
                "median_gpu_cache_usage_perc": median_cache,
                "median_prefix_cache_hit_rate": median_prefix_hit,
                # Mean values
                "mean_prompt_throughput": mean_prompt_tp,
                "mean_generation_throughput": mean_gen_tp,
                # Per-request latency stats (seconds)
                "latency_prefill_mean": _mean(prefill_samples),
                "latency_prefill_median": cls._compute_median(prefill_samples),
                "latency_prefill_p90": _p90(prefill_samples),
                "latency_decode_mean": _mean(decode_samples),
                "latency_decode_median": cls._compute_median(decode_samples),
                "latency_decode_p90": _p90(decode_samples),
                "latency_e2e_mean": _mean(e2e_samples),
                "latency_e2e_median": cls._compute_median(e2e_samples),
                "latency_e2e_p90": _p90(e2e_samples),
                "latency_queued_mean": _mean(queued_samples),
                "latency_queued_median": cls._compute_median(queued_samples),
                "latency_queued_p90": _p90(queued_samples),
                "latency_ttft_mean": _mean(ttft_samples),
                "latency_ttft_median": cls._compute_median(ttft_samples),
                "latency_ttft_p90": _p90(ttft_samples),
                "latency_num_finished_requests": len(e2e_samples),
                "total_preempted_reqs": stats["_total_preempted"],
                # Legacy field names for backwards compatibility (use peak values)
                "avg_prompt_throughput": stats["_peak_prompt_tp"],
                "avg_generation_throughput": stats["_peak_gen_tp"],
                "num_running_reqs": stats["_peak_running"],
                "num_waiting_reqs": stats["_peak_waiting"],
                "gpu_cache_usage_perc": stats["_peak_cache"],
                "prefix_cache_hit_rate": stats["_peak_prefix_hit"],
                # Metadata
                "timestamp": stats["timestamp"],
                "num_samples": stats["_num_samples"],
                "num_active_samples": stats["_num_active_samples"],
            }

            if reset:
                # Reset for next step
                del cls._stats_registry[engine_id]

            return result

class AsyncVLLMInferenceEngine(BaseVLLMInferenceEngine):
    """Asynchronous VLLM engine."""

    def __init__(self, *args, **kwargs):
        # Generate unique engine ID before calling super().__init__() which calls _create_engine
        self._stats_engine_id = id(self)
        super().__init__(*args, **kwargs)
        self._weight_loader = VLLMWeightLoader(self.llm, is_async=True)

    def _create_stat_logger_factory(self):
        """Create a factory that produces stat loggers with the engine ID set."""
        engine_id = self._stats_engine_id

        def factory(*args, **kwargs):
            logger_instance = V1LoggingStatLoggerFixed(*args, **kwargs)
            logger_instance.set_engine_id(engine_id)
            return logger_instance

        return factory

    def _create_engine(self, *args, **kwargs):
        openai_kwargs = pop_openai_kwargs(kwargs)

        # Store sampling params for OpenAI-style requests (Harbor rollouts)
        self._openai_sampling_params = openai_kwargs.pop("openai_sampling_params", {})
        if self._openai_sampling_params:
            logger.warning(
                f"OpenAI API sampling params overridden: "
                f"temperature={self._openai_sampling_params.get('temperature', 1.0)}, "
                f"top_p={self._openai_sampling_params.get('top_p', 1.0)}, "
                f"top_k={self._openai_sampling_params.get('top_k', -1)}"
            )

        # Logging kwargs
        enable_ray_prometheus_stats = kwargs.pop("enable_ray_prometheus_stats", False)
        enable_log_requests = kwargs.pop("enable_log_requests", False)
        max_log_len = kwargs.pop("max_log_len", None)
        custom_chat_template_path = kwargs.pop("custom_chat_template_chat_completion_path", None)

        # Use factory to inject engine ID into stat logger
        stat_loggers = [self._create_stat_logger_factory()]

        if version.parse(vllm.__version__) >= version.parse("0.10.0"):
            engine_args = vllm.AsyncEngineArgs(enable_log_requests=enable_log_requests, **kwargs)
        else:
            engine_args = vllm.AsyncEngineArgs(disable_log_requests=not enable_log_requests, **kwargs)

        # Add Ray Prometheus stat loggers if enabled
        if enable_ray_prometheus_stats:
            ray_loggers = self._create_ray_prometheus_stat_loggers()
            if ray_loggers:
                stat_loggers.extend(ray_loggers)

        # Stagger engine startup to avoid TOCTOU port collisions (EADDRINUSE).
        # vLLM's get_open_port() queries a free port then releases the socket;
        # if multiple engines on the same node call it simultaneously, they can
        # get the same port.  A random delay desynchronises the calls.
        import random
        _stagger = random.uniform(1.5, 3.0)
        logger.info(f"Engine startup stagger: sleeping {_stagger:.2f}s to avoid port collisions")
        time.sleep(_stagger)

        engine = vllm.AsyncLLMEngine.from_engine_args(engine_args, stat_loggers=stat_loggers)

        # Adapted from https://github.com/volcengine/verl/blob/e90f18c40aa639cd25092b78a5ff7e2d2508c088/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L327
        model_config = engine.model_config
        model_path = kwargs.get("model")
        # Use served_model_name if provided (from generator.served_model_name config),
        # otherwise fall back to model_path. This allows using a different model name
        # in HTTP endpoint requests than the actual model path.
        # See: https://github.com/NovaSky-AI/SkyRL/pull/238#discussion_r2326561295
        served_model_name = kwargs.get("served_model_name", None)
        model_name = served_model_name if served_model_name is not None else model_path

        base_model_paths = [BaseModelPath(name=model_name, model_path=model_path)]

        # vllm >= 0.11.2 removed model_config from OpenAI serving APIs
        is_new_api = version.parse(vllm.__version__) >= version.parse("0.11.2")
        legacy_kwargs = {}
        if is_new_api:
            models = OpenAIServingModels(engine, base_model_paths)
        else:
            models = OpenAIServingModels(engine, model_config, base_model_paths)
            legacy_kwargs["model_config"] = model_config

        # Build request logger for debugging (off by default).
        # Enable via: generator.engine_init_kwargs.enable_log_requests=true
        # Optionally limit logged chars: generator.engine_init_kwargs.max_log_len=256
        request_logger = None
        if enable_log_requests:
            from vllm.entrypoints.logger import RequestLogger

            request_logger = RequestLogger(max_log_len=max_log_len)

        # Load custom chat template if provided
        chat_template = openai_kwargs.pop("chat_template", None)
        if custom_chat_template_path:
            with open(custom_chat_template_path, "r") as f:
                chat_template = f.read()
            logger.info(f"Initializing OpenAIServingChat with custom_chat_template read from: {custom_chat_template_path}")

        self.openai_serving_chat = OpenAIServingChat(
            engine_client=engine,
            models=models,
            response_role="assistant",
            request_logger=request_logger,
            chat_template=chat_template,
            chat_template_content_format="auto",
            **legacy_kwargs,
            **openai_kwargs,
        )

        # TODO(Charlie): revisit kwargs `return_tokens_as_token_ids`,
        # `enable_prompt_tokens_details`, `enable_force_include_usage`.
        self.openai_serving_completion = OpenAIServingCompletion(
            engine_client=engine,
            models=models,
            request_logger=request_logger,
            **legacy_kwargs,
        )
        return engine

    def _create_ray_prometheus_stat_loggers(self):
        """Create Ray Prometheus stat loggers for vLLM metrics.

        Returns stat_loggers in the format expected by vLLM's from_engine_args().
        For vLLM v1 (0.9.0+), this returns a list of StatLoggerFactory callables.
        For older versions where the v1 API is not available, this returns `None`.

        See: https://docs.vllm.ai/en/latest/api/vllm/v1/metrics/ray_wrappers/
        """
        try:
            # Try vLLM v1 API first (0.9.0+)
            from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger

            logger.info("Enabling RayPrometheusStatLogger for vLLM inference engine metrics")
            # For v1, stat_loggers is a list of factory callables
            return [RayPrometheusStatLogger]
        except ImportError:
            logger.warning(
                "RayPrometheusStatLogger not available in this vLLM version. "
                "For Ray-integrated metrics, upgrade to vLLM >= 0.9.0. "
                "Stat logging will be disabled."
            )
            return None

    async def _load_lora_from_disk(self, lora_path: str):
        """Load LoRA adapters from disk using vLLM's native add_lora method."""
        lora_id = int(time.time_ns() % 0x7FFFFFFF)
        lora_request = LoRARequest(lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=lora_path)
        result = await self.llm.add_lora(lora_request)
        return result

    async def _collect_outputs(self, prompt_token_ids, request_id: str, sampling_params: SamplingParams):
        """Collect outputs for a single prompt."""
        # Check if LoRA is enabled and create LoRA request
        final_output = None
        lora_request = None

        if self._is_lora:
            lora_int_ids = list(await self.llm.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                # dummy_lora_path for placeholder (actual loading done in add_lora())
                lora_request = LoRARequest(
                    lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/dummy_lora_path"
                )

        async for request_output in self.llm.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        ):
            final_output = request_output

        return final_output

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        """Generate responses using vLLM's async engine."""
        prompt_token_ids, sampling_params = self._preprocess_prompts(input_batch)

        tasks = []
        for prompt in prompt_token_ids:
            # Schedule the collection of outputs for each prompt.
            # Avoid duplicate request_ids
            request_id = str(uuid4().hex)
            task = asyncio.create_task(self._collect_outputs(prompt, request_id, sampling_params))
            tasks.append(task)
        outputs = await asyncio.gather(*tasks)

        return self._postprocess_outputs(outputs)

    async def wake_up(self, *args: Any, **kwargs: Any):
        await self.llm.wake_up(tags=kwargs.get("tags", None))

    async def sleep(self, *args: Any, **kwargs: Any):
        engine = self._get_engine()
        output_processor = engine.output_processor
        # make sure that the engine is alive
        engine.engine_core.ensure_alive()
        if output_processor.has_unfinished_requests():
            logger.warning(
                "Calling sleep() with unfinished requests in vLLM engine. This is unexpected since all "
                "generation should be done before sleep() is called. Check for potential failures or "
                "dangling requests in your Generator/Env. Aborting all unfinished requests."
            )
            unfinished_request_ids = self._get_unfinished_request_ids(output_processor)
            await engine.abort(unfinished_request_ids)

        # TODO(team): remove once vllm fixes this
        # otherwise waking it up will output gibberish: https://github.com/vllm-project/vllm/issues/17103
        await self.reset_prefix_cache()
        level = 1 if self._is_lora else kwargs.get("level", 2)
        await self.llm.sleep(level=level)

    async def init_weight_update_communicator(self, init_info: "WeightSyncInitInfo"):
        import pickle

        engine = self._get_engine()
        # Pickle the init_info to preserve type through collective_rpc
        pickled_init_info = pickle.dumps(init_info)
        return await engine.collective_rpc(
            "init_weight_update_communicator",
            args=(pickled_init_info,),
        )

    async def update_named_weights(self, request: WeightUpdateRequest):
        from skyrl.backends.skyrl_train.weight_sync import LoraLoadRequest

        # Check for LoRA disk loading request
        if isinstance(request, LoraLoadRequest):
            return await self._load_lora_from_disk(request.lora_path)

        if not len(request):
            raise ValueError("Weight update request must not be empty")

        # Use the weight loader to coordinate weight transfer
        return await self._weight_loader.load_weights(request)

    async def teardown(self):
        await self._teardown_weight_receiver()

    async def reset_prefix_cache(self):
        engine = self._get_engine()
        await engine.reset_prefix_cache()

    async def _teardown_weight_receiver(self):
        engine = self._get_engine()
        return await engine.collective_rpc("teardown_weight_receiver")

    # ----------------------------------------
    # Methods for handling OpenAI API requests
    # ----------------------------------------

    async def _handle_openai_request(self, request_payload: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """Handle OpenAI API request."""
        assert endpoint in ["/chat/completions", "/completions"]

        body = request_payload.get("json", {})
        headers = request_payload.get("headers", {})

        # Apply configured sampling params from generator config.
        # Harbor requests may include their own sampling params; we override
        # with the SkyRL generator config so rollout exploration is consistent.
        sp = getattr(self, "_openai_sampling_params", {})
        body.update({
            "temperature": sp.get("temperature", 1.0),
            "top_p": sp.get("top_p", 1.0),
            "top_k": sp.get("top_k", -1),
            "min_p": sp.get("min_p", 0.0),
        })

        # 1. Build request
        try:
            if endpoint == "/chat/completions":
                request = ChatCompletionRequest(**body)
            else:
                request = CompletionRequest(**body)
            assert request.stream is False, "Streaming is not supported in SkyRL yet, please set stream to False."
        except Exception as e:
            if version.parse(vllm.__version__) >= version.parse("0.10.0"):
                return ErrorResponse(
                    error=ErrorInfo(
                        message=str(e),
                        type=HTTPStatus.BAD_REQUEST.phrase,
                        code=HTTPStatus.BAD_REQUEST.value,
                    ),
                ).model_dump()
            else:
                return ErrorResponse(
                    message=str(e),
                    type=HTTPStatus.BAD_REQUEST.phrase,
                    code=HTTPStatus.BAD_REQUEST.value,
                ).model_dump()

        # 2. Call vllm engine
        try:
            # Create a minimal request-like object with attributes used by vLLM
            minimal_request = _MinimalRequest(headers)
            if endpoint == "/chat/completions":
                generator = await self.openai_serving_chat.create_chat_completion(request, minimal_request)
                assert isinstance(generator, (ChatCompletionResponse, ErrorResponse))
            else:
                generator = await self.openai_serving_completion.create_completion(request, minimal_request)
                assert isinstance(generator, (CompletionResponse, ErrorResponse))
            return generator.model_dump()

        except Exception as e:
            # Handle it here so we can surface the error from a ray worker.

            # Determine appropriate HTTP status code based on error message to mimic vllm serve error
            # handling. Here, we handle context length errors, which should return 400 according to
            # vllm serve error handling, so that downstream users can handle these properly rather
            # than seeing a 500 SkyRL INTERNAL_SERVER_ERROR. For instance, LiteLLM can wraps them as
            # BadRequestError, enabling Harbor to detect ContextLengthExceededError.
            # NOTE(Charlie): This is hacky. With the refactored inference stack, we
            # should be able to directly reuse the error handling from the served vllm.
            error_message = str(e).lower()
            is_context_length_error = "context length" in error_message or "maximum input length" in error_message

            if is_context_length_error:
                http_status = HTTPStatus.BAD_REQUEST
            else:
                http_status = HTTPStatus.INTERNAL_SERVER_ERROR

            if version.parse(vllm.__version__) >= version.parse("0.10.0"):
                return ErrorResponse(
                    error=ErrorInfo(
                        message=str(e),
                        type=http_status.phrase,
                        code=http_status.value,
                    ),
                ).model_dump()
            else:
                return ErrorResponse(
                    message=str(e),
                    type=http_status.phrase,
                    code=http_status.value,
                ).model_dump()

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible HTTP endpoint for handling `/chat/completions` in Python vLLM engine.

        Accepts a JSON-serializable payload: {"json": <request-body>, "headers": <headers-dict>}.
        Constructs a minimal request-like object for vLLM's openai_serving_chat.
        Returns a plain dict, either a ChatCompletionResponse or an ErrorResponse, both defined
        in vllm.entrypoints.openai.protocol.
        """
        return await self._handle_openai_request(request_payload, endpoint="/chat/completions")

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible HTTP endpoint for handling `/completions` in Python vLLM engine.

        Accepts a JSON-serializable payload: {"json": <request-body>, "headers": <headers-dict>}.
        Constructs a minimal request-like object for vLLM's openai_serving_completion.
        Returns a plain dict, either a CompletionResponse or an ErrorResponse, both defined
        in vllm.entrypoints.openai.protocol.
        """
        return await self._handle_openai_request(request_payload, endpoint="/completions")

    async def get_stats(self) -> Dict[str, Any]:
        """Get accumulated vLLM engine statistics for the current step.

        Returns a dict with the following keys:
        - peak_*: Peak values observed during the step
        - median_*: Median values across active samples
        - mean_*: Mean values across active samples
        - num_samples: Total number of stat samples collected
        - num_active_samples: Number of samples with active requests
        - timestamp: Unix timestamp of last sample
        - engine_id: Unique identifier for this engine instance

        Note: Stats are reset after reading to provide fresh stats per training step.

        Used by VLLMStatsCallback to collect and aggregate stats across engines.
        """
        # Reset=True ensures each training step gets fresh stats
        stats = V1LoggingStatLoggerFixed.get_stats_by_engine_id(self._stats_engine_id, reset=True)
        if stats is None:
            # Return empty stats if no data recorded yet
            stats = {
                # Peak values
                "peak_prompt_throughput": 0.0,
                "peak_generation_throughput": 0.0,
                "peak_running_reqs": 0,
                "peak_waiting_reqs": 0,
                "peak_gpu_cache_usage_perc": 0.0,
                "peak_prefix_cache_hit_rate": 0.0,
                # Median values
                "median_prompt_throughput": 0.0,
                "median_generation_throughput": 0.0,
                "median_running_reqs": 0.0,
                "median_waiting_reqs": 0.0,
                "median_gpu_cache_usage_perc": 0.0,
                "median_prefix_cache_hit_rate": 0.0,
                # Mean values
                "mean_prompt_throughput": 0.0,
                "mean_generation_throughput": 0.0,
                # Per-request latency stats
                "latency_prefill_mean": 0.0,
                "latency_prefill_median": 0.0,
                "latency_prefill_p90": 0.0,
                "latency_decode_mean": 0.0,
                "latency_decode_median": 0.0,
                "latency_decode_p90": 0.0,
                "latency_e2e_mean": 0.0,
                "latency_e2e_median": 0.0,
                "latency_e2e_p90": 0.0,
                "latency_queued_mean": 0.0,
                "latency_queued_median": 0.0,
                "latency_queued_p90": 0.0,
                "latency_ttft_mean": 0.0,
                "latency_ttft_median": 0.0,
                "latency_ttft_p90": 0.0,
                "latency_num_finished_requests": 0,
                "total_preempted_reqs": 0,
                # Legacy field names
                "avg_prompt_throughput": 0.0,
                "avg_generation_throughput": 0.0,
                "num_running_reqs": 0,
                "num_waiting_reqs": 0,
                "gpu_cache_usage_perc": 0.0,
                "prefix_cache_hit_rate": 0.0,
                # Metadata
                "num_samples": 0,
                "num_active_samples": 0,
                "timestamp": time.time(),
            }
        stats["engine_id"] = self._stats_engine_id
        return stats

    async def abort_generation(self) -> None:
        """
        Abort all running and waiting requests, which make the ongoing requests return the
        already-generated tokens with a stop_reason of "abort".
        """
        engine = self._get_engine()
        await engine.pause_generation(mode="keep", clear_cache=False)
        logger.info("abort_generation (via pause_generation mode='keep') finished")

    async def pause_generation(self, clear_cache: bool = False) -> None:
        """Pause generation using vLLM's native keep mode, freezing in-flight requests."""
        engine = self._get_engine()
        await engine.pause_generation(mode="keep", clear_cache=clear_cache)
        logger.info("pause_generation(mode='keep') finished")

    async def resume_generation(self) -> None:
        """Resume generation after a keep-mode pause."""
        engine = self._get_engine()
        await engine.resume_generation()
        logger.info("resume_generation() finished")


class _MinimalRequest:
    """
    Minimal request-like object for vLLM's openai_serving_chat and openai_serving_completion.

    We cannot use the original user Request object because it cannot be serialized and hence
    cannot be a ray method argument. Instead we take the original request's headers and
    reconstruct an instance of _MinimalRequest to mimic the FastAPI Request object.

    The fields depend on what vLLM accesses internally.
    """

    def __init__(self, headers):
        self.headers = headers  # Expect a mapping with .get support
        self.state = SimpleNamespace()  # vLLM sets raw_request.state.request_metadata


class VLLMWeightLoader(WeightLoader):
    """Loads weights into vLLM engine, managing RPC coordination.

    This loader encapsulates the collective_rpc calls to workers.
    Workers create the appropriate receiver locally for the actual weight transfer.
    """

    def __init__(self, engine: Any, is_async: bool = False) -> None:
        """Initialize the loader.

        Args:
            engine: The vLLM engine (LLM or AsyncLLMEngine).
            is_async: Whether this is for AsyncVLLMInferenceEngine.
        """
        self._engine = engine.engine if hasattr(engine, "engine") else engine
        self._is_async = is_async

    async def load_weights(self, request: WeightUpdateRequest) -> None:
        """Load weights by coordinating RPC to workers.

        Sends the request to workers via collective_rpc. Workers create
        the receiver locally and use it to receive and load weights.

        Args:
            request: Weight update request.
        """
        import pickle

        # Pickle the request to preserve type through collective_rpc
        pickled_request = pickle.dumps(request)

        if self._is_async:
            await self._engine.collective_rpc(
                "load_weights",
                args=(pickled_request,),
            )
        else:
            await asyncio.to_thread(
                self._engine.collective_rpc,
                "load_weights",
                args=(pickled_request,),
            )


VLLMRayActor = ray.remote(VLLMInferenceEngine)
AsyncVLLMRayActor = ray.remote(AsyncVLLMInferenceEngine)
