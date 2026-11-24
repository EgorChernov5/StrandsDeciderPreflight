"""Проверки preflight-контракта без загрузки реальной модели."""

# ruff: noqa: RUF002

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from strands_decider import modeling, server


class _FakeTokenizer:
    """
    Имитирует загруженный tokenizer с намеренно нестабильным исходным порядком.

    Attributes:
        all_special_tokens (list[str]): Специальные токены tokenizer.
        vocab_files_names (dict[str, str]): Имена загруженных tokenizer assets.
        init_kwargs (dict[str, str]): Параметры загрузки tokenizer.

    Fallbacks:
        Не предусмотрены.
    """

    def __init__(self) -> None:
        self.all_special_tokens = ["<z>", "<pad>", "<a>", "<z>"]
        self.vocab_files_names = {"tokenizer_file": "tokenizer.json"}
        self.init_kwargs = {"tokenizer_file": "tokenizer.json"}

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Возвращает число токенов, автоматически добавляемых сериализатором.

        Parameters:
            pair (bool): Признак парного ввода. По умолчанию: False.

        Returns:
            int: Два обязательных токена для одиночного ввода.

        Fallbacks:
            Для парного ввода используется то же значение.
        """
        return 2


class _FakeConfig:
    """
    Хранит фактическую конфигурацию fake checkpoint.

    Attributes:
        base_model (str): Идентификатор базовой модели.
        max_length (int): Нативное окно модели.
        temperature (float): Температура калибровки.

    Fallbacks:
        Не предусмотрены.
    """

    base_model = "base/model"
    max_length = 4096
    temperature = 0.75


class _FakeModel:
    """
    Представляет загруженную модель с реальными torch device и dtype.

    Attributes:
        config (_FakeConfig): Конфигурация модели.
        tokenizer (_FakeTokenizer): Загруженный tokenizer.
        torso (torch.nn.Module): Fake torso для проверки runtime.
        checkpoint_path (str): Разрешённый путь checkpoint.
        checkpoint_hub_id (str | None): Hub id либо None для локальной модели.

    Fallbacks:
        Не предусмотрены.
    """

    def __init__(self, checkpoint_path: str, hub_id: str | None) -> None:
        self.config = _FakeConfig()
        self.tokenizer = _FakeTokenizer()
        self.torso = torch.nn.Linear(2, 2).to(dtype=torch.float64)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_hub_id = hub_id


class _FakeEngine:
    """
    Хранит fake model и прежнюю конфигурацию server engine.

    Attributes:
        model (_FakeModel): Загруженная fake model.
        cfg (object): Конфигурация HTTP runtime.

    Fallbacks:
        Не предусмотрены.
    """

    def __init__(self, model: _FakeModel, cfg: object) -> None:
        self.model = model
        self.cfg = cfg


def _write_checkpoint_assets(path: Path) -> None:
    """
    Создаёт минимальные model и tokenizer assets для preflight.

    Parameters:
        path (Path): Каталог fake checkpoint.

    Returns:
        None: Функция записывает assets в переданный каталог.

    Fallbacks:
        Не предусмотрены.
    """
    path.mkdir(parents=True)
    (path / "strands_decider_config.json").write_text("{}", encoding="utf-8")
    (path / "head.safetensors").write_bytes(b"head")
    (path / "tokenizer.json").write_text('{"version":"1"}', encoding="utf-8")
    (path / "tokenizer_config.json").write_text('{"model_max_length":4096}', encoding="utf-8")
    (path / "special_tokens_map.json").write_text('{"pad_token":"<pad>"}', encoding="utf-8")


def _client(
    monkeypatch: object,
    checkpoint: str,
    model: _FakeModel,
    revision: str | None = None,
) -> tuple[TestClient, list[tuple[str, str | None]]]:
    """
    Создаёт TestClient с fake loaded runtime.

    Parameters:
        monkeypatch (object): Pytest fixture для замены загрузчиков.
        checkpoint (str): Аргумент checkpoint команды serve.
        model (_FakeModel): Фактически загруженная fake model.
        revision (str | None): Переданная Hub revision. По умолчанию: None.

    Returns:
        tuple[TestClient, list[tuple[str, str | None]]]: Клиент и журнал загрузки.

    Fallbacks:
        Не предусмотрены.
    """
    calls: list[tuple[str, str | None]] = []

    def load(_cls: type, _checkpoint: str, **kwargs: object) -> _FakeModel:
        calls.append((_checkpoint, kwargs.get("revision")))
        return model

    monkeypatch.setattr(server.StrandsDeciderModel, "load", classmethod(load))
    monkeypatch.setattr(server, "SystemOneEngine", _FakeEngine)
    client = TestClient(server.create_app(checkpoint, revision=revision, device="cpu"))
    return client, calls


def test_health_returns_complete_preflight_and_preserves_old_fields(
    tmp_path: Path, monkeypatch: object
) -> None:
    """Полный health содержит новый контракт и прежние диагностические поля."""
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint_assets(checkpoint)
    client, _ = _client(monkeypatch, str(checkpoint), _FakeModel(str(checkpoint), None))

    health = client.get("/health").json()

    assert health["preflight_contract_version"] == "semantic-validation-preflight-v1"
    assert health["prompt_contract"] == "strands_decider_prompting_v1"
    assert health["serializer_revision"] == "strands_decider_http_to_model_v1"
    assert health["backend_contract"] == "strands_decider.http.v1.systemone"
    assert health["native_window"] == 4096
    assert health["prompt_reserve"] == 2
    assert health["device"] == "cpu"
    assert health["dtype"] == "float64"
    assert health["preflight_errors"] == []
    assert health["special_tokens"] == ["<a>", "<pad>", "<z>"]
    assert health["model_id"].startswith("local-sha256:")
    assert health["model_revision"].startswith("sha256:")
    assert health["tokenizer_revision"].startswith("sha256:")
    assert health["status"] == "ok"
    assert health["model"] == "checkpoint"
    assert health["checkpoint"] == str(checkpoint)
    assert health["base_model"] == "base/model"
    assert health["max_length"] == 4096
    assert health["temperature"] == 0.75
    assert health["prefix_cache"] is True
    assert health["vision"] is False


def test_ping_is_the_same_contract_as_health(tmp_path: Path, monkeypatch: object) -> None:
    """Алиас ping возвращает полностью идентичный health payload."""
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint_assets(checkpoint)
    client, _ = _client(monkeypatch, str(checkpoint), _FakeModel(str(checkpoint), None))

    assert client.get("/ping").json() == client.get("/health").json()


def test_hub_revision_is_resolved_sha_and_revision_reaches_loader(
    tmp_path: Path, monkeypatch: object
) -> None:
    """Health сообщает SHA snapshot, а загрузчик получает явно заданную revision."""
    sha = "216d1ccb46a9e9b036eb9187b248db98480f77d4"
    checkpoint = tmp_path / "models--org--model" / "snapshots" / sha
    _write_checkpoint_assets(checkpoint)
    model = _FakeModel(str(checkpoint), "org/model")

    client, calls = _client(monkeypatch, "org/model", model, revision="release-tag")

    health = client.get("/health").json()
    assert calls == [("org/model", "release-tag")]
    assert health["model_id"] == "org/model"
    assert health["model_revision"] == sha


def test_hub_revision_reaches_snapshot_download(tmp_path: Path, monkeypatch: object) -> None:
    """Явная revision передаётся в snapshot download без изменения значения."""
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    calls: list[tuple[str, str | None]] = []

    def download(repo_id: str, revision: str | None = None) -> str:
        calls.append((repo_id, revision))
        return str(snapshot)

    monkeypatch.setattr(modeling, "snapshot_download", download)

    assert modeling.checkpoint_dir("org/model", "pinned") == str(snapshot)
    assert calls == [("org/model", "pinned")]


def test_tokenizer_revision_fingerprints_only_tokenizer_assets(tmp_path: Path) -> None:
    """Fingerprint меняется от tokenizer asset и не меняется от model head."""
    checkpoint = tmp_path / "checkpoint"
    _write_checkpoint_assets(checkpoint)
    tokenizer = _FakeTokenizer()

    first = server._tokenizer_fingerprint(str(checkpoint), tokenizer)
    (checkpoint / "head.safetensors").write_bytes(b"changed model")
    assert server._tokenizer_fingerprint(str(checkpoint), tokenizer) == first
    (checkpoint / "tokenizer.json").write_text('{"version":"2"}', encoding="utf-8")
    assert server._tokenizer_fingerprint(str(checkpoint), tokenizer) != first
    assert first is not None and len(first) == hashlib.sha256().digest_size * 2


def test_unprovable_hub_revision_is_null_with_diagnostic(
    tmp_path: Path, monkeypatch: object
) -> None:
    """Непроверяемый путь Hub не подменяется branch, tag или именем каталога."""
    checkpoint = tmp_path / "downloaded-model"
    _write_checkpoint_assets(checkpoint)
    client, _ = _client(
        monkeypatch,
        "org/model",
        _FakeModel(str(checkpoint), "org/model"),
        revision="main",
    )

    health = client.get("/health").json()
    assert health["model_revision"] is None
    assert any(error.startswith("model_revision:") for error in health["preflight_errors"])


def test_special_tokens_have_deterministic_order() -> None:
    """Порядок special_tokens не зависит от порядка и дублей tokenizer runtime."""
    tokenizer = _FakeTokenizer()
    tokenizer.all_special_tokens = ["<pad>", "<z>", "<a>", "<pad>"]
    model = type(
        "Model",
        (),
        {
            "tokenizer": tokenizer,
            "config": _FakeConfig(),
            "checkpoint_path": None,
            "checkpoint_hub_id": None,
            "torso": torch.nn.Linear(1, 1),
        },
    )()
    engine = type("Engine", (), {"model": model})()

    assert server._preflight_contract(engine)["special_tokens"] == ["<a>", "<pad>", "<z>"]
