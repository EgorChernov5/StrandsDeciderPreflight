"""HTTP server exposing the System One API.

The path and the request and response shapes follow the public Jev API documentation.
JevBench's typesafe adapter runs against this server unchanged (evaluation/jevbench.md).
Compatibility with the Jev API itself is not verified.
"""

# ruff: noqa: RUF002

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import __version__
from .infer import EngineConfig, SystemOneEngine, load_mlx
from .modeling import StrandsDeciderModel
from .schema import SystemOneRequest, SystemOneResponse

_engine: SystemOneEngine | None = None
_engine_lock = threading.Lock()

PREFLIGHT_CONTRACT_VERSION = "semantic-validation-preflight-v1"
PROMPT_CONTRACT = "strands_decider_prompting_v1"
SERIALIZER_REVISION = "strands_decider_http_to_model_v1"
BACKEND_CONTRACT = "strands_decider.http.v1.systemone"

_TOKENIZER_ASSET_NAMES = {
    "added_tokens.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}


def _needs_serialisation(device: str) -> bool:
    """MPS aborts the process with a Metal command-buffer assertion (issue #9) when
    two overlapping threads hit the same device. CUDA and CPU are thread-safe for
    inference, and MLX serialises on its own scheduler."""
    return str(device).startswith("mps")


def get_engine() -> SystemOneEngine:
    if _engine is None:  # pragma: no cover - guarded by create_app
        raise RuntimeError("engine not initialised")
    return _engine


def _files_fingerprint(root: Path, files: list[Path]) -> str | None:
    """
    Вычисляет SHA-256 содержимого файлов с учётом их относительных имён.

    Parameters:
        root (Path): Корневой каталог набора файлов.
        files (list[Path]): Файлы для хеширования.

    Returns:
        str | None: SHA-256 в hex или None для пустого набора.

    Fallbacks:
        Пустой набор возвращает None.
    """
    selected = sorted({path.absolute() for path in files if path.is_file()})
    if not selected:
        return None
    digest = hashlib.sha256()
    for path in selected:
        name = path.relative_to(root.absolute()).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _model_fingerprint(checkpoint_path: str) -> str | None:
    """
    Вычисляет fingerprint фактически загружаемых частей локального checkpoint.

    Parameters:
        checkpoint_path (str): Разрешённый каталог checkpoint.

    Returns:
        str | None: SHA-256 конфигурации, head, adapter и provenance.

    Fallbacks:
        Возвращает None, если каталог не содержит model assets.
    """
    root = Path(checkpoint_path).resolve()
    names = {
        "head.safetensors",
        "hobson_config.json",
        "provenance.json",
        "slot_head.pt",
        "strands_decider_config.json",
    }
    files = [root / name for name in names]
    lora = root / "lora"
    if lora.is_dir():
        files.extend(path for path in lora.rglob("*") if path.is_file())
    return _files_fingerprint(root, files)


def _tokenizer_fingerprint(checkpoint_path: str, tokenizer: object) -> str | None:
    """
    Вычисляет SHA-256 tokenizer assets, из которых создан tokenizer runtime.

    Parameters:
        checkpoint_path (str): Разрешённый каталог checkpoint.
        tokenizer (object): Фактически загруженный tokenizer.

    Returns:
        str | None: Детерминированный SHA-256 tokenizer assets.

    Fallbacks:
        Возвращает None, если ни один tokenizer asset не найден.
    """
    root = Path(checkpoint_path).resolve()
    names = set(_TOKENIZER_ASSET_NAMES)
    vocab_files = getattr(tokenizer, "vocab_files_names", None) or {}
    names.update(Path(value).name for value in vocab_files.values() if value)
    init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
    names.update(
        Path(value).name
        for key, value in init_kwargs.items()
        if (key == "tokenizer_file" or key.endswith("_file")) and isinstance(value, str)
    )
    return _files_fingerprint(root, [root / name for name in names])


def _hub_commit(checkpoint_path: str) -> str | None:
    """
    Извлекает immutable commit SHA из разрешённого пути Hub snapshot.

    Parameters:
        checkpoint_path (str): Путь, возвращённый snapshot download.

    Returns:
        str | None: Сорокасимвольный commit SHA или None.

    Fallbacks:
        Непроверяемая структура пути возвращает None.
    """
    path = Path(checkpoint_path).resolve()
    if path.parent.name == "snapshots" and re.fullmatch(r"[0-9a-fA-F]{40}", path.name):
        return path.name.lower()
    return None


def _runtime_device_dtype(engine: SystemOneEngine) -> tuple[str | None, str | None]:
    """
    Определяет устройство и dtype по фактически загруженному runtime.

    Parameters:
        engine (SystemOneEngine): Загруженный inference engine.

    Returns:
        tuple[str | None, str | None]: Устройство и dtype модели.

    Fallbacks:
        Для runtime без проверяемых параметров возвращает None.
    """
    model = engine.model
    runtime_device = getattr(model, "runtime_device", None)
    runtime_dtype = getattr(model, "runtime_dtype", None)
    if runtime_device is not None or runtime_dtype is not None:
        return runtime_device, runtime_dtype

    module = getattr(model, "torso", model)
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return None, None
    totals: dict[tuple[str, str], int] = {}
    for parameter in parameters():
        if getattr(parameter, "is_floating_point", lambda: False)():
            key = (str(parameter.device), str(parameter.dtype).removeprefix("torch."))
            totals[key] = totals.get(key, 0) + int(parameter.numel())
    if not totals:
        return None, None

    target = str(getattr(getattr(engine, "cfg", None), "device", ""))
    on_target = {
        key: count
        for key, count in totals.items()
        if key[0] == target or key[0].split(":", 1)[0] == target.split(":", 1)[0]
    }
    candidates = on_target or totals
    device, dtype = max(candidates, key=candidates.__getitem__)
    return device, dtype


def _preflight_contract(engine: SystemOneEngine) -> dict[str, object]:
    """
    Собирает доказуемый preflight-контракт загруженного engine.

    Parameters:
        engine (SystemOneEngine): Полностью загруженный inference engine.

    Returns:
        dict[str, object]: Поля preflight и диагностика недоказуемых значений.

    Fallbacks:
        Недоказуемые значения равны None и сопровождаются preflight_errors.
    """
    model = engine.model
    tokenizer = getattr(model, "tokenizer", None)
    checkpoint_path = getattr(model, "checkpoint_path", None)
    hub_id = getattr(model, "checkpoint_hub_id", None)
    errors: list[str] = []

    model_fingerprint = (
        _model_fingerprint(checkpoint_path) if checkpoint_path and not hub_id else None
    )
    model_id = hub_id or (f"local-sha256:{model_fingerprint}" if model_fingerprint else None)
    if hub_id:
        model_revision = _hub_commit(checkpoint_path) if checkpoint_path else None
    else:
        model_revision = f"sha256:{model_fingerprint}" if model_fingerprint else None

    tokenizer_fingerprint = (
        _tokenizer_fingerprint(checkpoint_path, tokenizer)
        if checkpoint_path and tokenizer is not None
        else None
    )
    tokenizer_revision = f"sha256:{tokenizer_fingerprint}" if tokenizer_fingerprint else None

    device, dtype = _runtime_device_dtype(engine)
    special_tokens = None
    prompt_reserve = None
    if tokenizer is not None:
        tokens = getattr(tokenizer, "all_special_tokens", None)
        if tokens is not None:
            special_tokens = sorted({str(token) for token in tokens})
        reserve = getattr(tokenizer, "num_special_tokens_to_add", None)
        if callable(reserve):
            prompt_reserve = int(reserve(pair=False))

    fields = {
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "native_window": getattr(getattr(model, "config", None), "max_length", None),
        "special_tokens": special_tokens,
        "prompt_reserve": prompt_reserve,
        "device": device,
        "dtype": dtype,
    }
    for name, value in fields.items():
        if value is None:
            errors.append(f"{name}: значение невозможно доказать по загруженному runtime")

    return {
        "preflight_contract_version": PREFLIGHT_CONTRACT_VERSION,
        **fields,
        "prompt_contract": PROMPT_CONTRACT,
        "serializer_revision": SERIALIZER_REVISION,
        "backend_contract": BACKEND_CONTRACT,
        "preflight_errors": errors,
    }


def create_app(
    checkpoint: str,
    *,
    revision: str | None = None,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    attn_implementation: str | None = None,
    strict_window: bool = False,
    max_batch: int = 32,
    vision: bool = False,
    max_batch_tokens: int | None = None,
) -> FastAPI:
    global _engine

    app = FastAPI(
        title="strands-decider System One",
        version=__version__,
        description="Typed, calibrated answers. Choice, Score and Noul over one state.",
    )

    # Identify the response's `model` by the checkpoint served, so multiple servers
    # on the same host cannot be confused. HF repo ids ("org/name") collapse to `name`.
    resolved_name = model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint

    config = EngineConfig(
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=resolved_name,
        strict_window=strict_window,
        max_batch=max_batch,
        max_batch_tokens=max_batch_tokens,
    )
    if vision:
        if device == "mlx":
            raise ValueError("--vision runs on torch devices (cuda, mps, cpu), not mlx")
        from .vision import load_vision_engine

        _engine = load_vision_engine(
            checkpoint,
            config,
            revision=revision,
            attn_implementation=attn_implementation,
        )
    elif device == "mlx":
        _engine = (
            load_mlx(checkpoint, config, revision=revision)
            if revision is not None
            else load_mlx(checkpoint, config)
        )
    else:
        model = StrandsDeciderModel.load(
            checkpoint, revision=revision, attn_implementation=attn_implementation
        )
        _engine = SystemOneEngine(model, config)

    eng = get_engine()
    preflight = _preflight_contract(eng)

    # SageMaker and AgentCore Runtime call /ping and /invocations; each is the same
    # handler as the path above it.
    @app.get("/health")
    @app.get("/ping")
    def health() -> dict:
        return {
            "status": "ok",
            "model": eng.cfg.model_name,
            "checkpoint": checkpoint,
            "base_model": eng.model.config.base_model,
            "max_length": eng.model.config.max_length,
            "temperature": eng.model.config.temperature,
            "device": preflight["device"],
            "prefix_cache": eng.cfg.use_prefix_cache,
            "vision": vision,
            **preflight,
        }

    guard: contextlib.AbstractContextManager
    guard = _engine_lock if _needs_serialisation(device) else contextlib.nullcontext()

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    @app.post("/invocations", response_model=SystemOneResponse)
    def systemone(request: SystemOneRequest) -> JSONResponse:
        eng = get_engine()
        started = time.perf_counter()
        try:
            with guard:
                response = eng.evaluate(request)
        except ValueError as exc:
            # Malformed permutation, etc. -- caller error.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = response.model_dump()
        payload["latency_ms"] = round(elapsed_ms, 2)
        return JSONResponse(content=payload)

    return app


def serve(
    checkpoint: str,
    *,
    revision: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    model_name: str | None = None,
    strict_window: bool = False,
    max_batch: int = 32,
    vision: bool = False,
    max_batch_tokens: int | None = None,
) -> None:
    import uvicorn

    app = create_app(
        checkpoint,
        revision=revision,
        device=device,
        use_prefix_cache=use_prefix_cache,
        model_name=model_name,
        strict_window=strict_window,
        max_batch=max_batch,
        vision=vision,
        max_batch_tokens=max_batch_tokens,
    )
    # Single worker: the model owns the GPU, and forking more would just duplicate it.
    uvicorn.run(app, host=host, port=port, workers=1)
