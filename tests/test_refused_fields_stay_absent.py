"""A field the loader refused is never presented, echoed or acted on as a value
(#1261, round 2).

``get`` and ``update`` load a file whose type will not fully load with
``partial=True``: the declared type, built without the fields its loader
refuses (``entry.load_problems``). A dataclass cannot lack an attribute, so the
model holds the type's default for each refused field (``rank: 0``, ``status:
proposed``). Nothing downstream may tell that default from the file's value:

* the read leaves the refused field out and names it in ``load_problems``
  (CLI, REST and MCP);
* a whole read sent back as an update changes no byte of the file;
* no templated move runs on the model (``backlog/{status}`` moved a file whose
  ``status`` was refused into ``backlog/proposed/``);
* validation, hooks and in-process callers never see the default as a value.

Why a filter at the choke points and not an absent attribute: the partial
model has three construction sites (``test_the_partial_model_has_three_...``),
all in ``KBService``; each consumer of it is closed in this file. An attribute
removed from the instance falls back to the class default, and a model that
raises on access would break every generic field walker.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from pyrite.config import KBConfig

#: shape -> (frontmatter lines, refused field)
SHAPES = {
    "inf": ("rank: .inf\n", "rank"),
    "bad-int": ("rank: abc\n", "rank"),
    "bad-enum": ("status: weird\n", "status"),
}
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _a_loader_that_refuses_a_status(monkeypatch):
    """No in-tree loader raises on a status value (they coerce or keep it), so
    the enum shape is a loader that does, as a plugin's validator can: it
    refuses ``status: weird`` the way ``backlog_item`` refuses ``rank: .inf``.
    The in-process surfaces (CLI runner, TestClient, MCP dispatcher) all run
    this class."""
    from pyrite_software_kb.entry_types import BacklogItemEntry

    original = BacklogItemEntry.from_frontmatter.__func__

    def from_frontmatter(cls, meta, body=""):
        if meta.get("status") == "weird":
            raise ValueError("status must be one of: proposed, done")
        return original(cls, meta, body)

    monkeypatch.setattr(BacklogItemEntry, "from_frontmatter", classmethod(from_frontmatter))


def _file(extra: str, entry_id: str = "x") -> str:
    return f"---\ntype: backlog_item\nid: {entry_id}\ntitle: T\n{extra}---\n\nZebra body.\n"


@pytest.fixture
def make_world(tmp_path):
    """A software-KB world; ``files`` (KB-relative path -> text) exist before
    the world is built, so a ``kb.yaml`` among them is the schema it loads."""
    import contextlib

    from tests._world import build_world

    stack = contextlib.ExitStack()
    count = {"n": 0}

    def make(files=None):
        count["n"] += 1
        kb = tmp_path / f"sw{count['n']}"
        for rel, text in (files or {}).items():
            (kb / rel).parent.mkdir(parents=True, exist_ok=True)
            (kb / rel).write_text(text, encoding="utf-8")
        w = build_world(
            tmp_path / f"w{count['n']}", kbs=[KBConfig(name="sw", path=kb, kb_type="software")]
        )
        stack.callback(w.close)
        stack.enter_context(w.pointed())
        return w

    with stack:
        yield make


@pytest.fixture
def world(make_world):
    return make_world()


def _cli_get(world, entry_id):
    got = world.cli("get", entry_id, "-k", "sw", "--format", "json")
    assert got.exit_code == 0, got.output
    return json.loads(got.stdout)


def _rest_get(world, entry_id):
    resp = world.client.get(f"/api/entries/{entry_id}", params={"kb": "sw"})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _mcp_get(world, entry_id):
    res = world.mcp()._dispatch_tool("kb_get", {"entry_id": entry_id, "kb_name": "sw"})
    assert "entry" in res, res
    return res["entry"]


SURFACES = {"cli": _cli_get, "rest": _rest_get, "mcp": _mcp_get}


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("shape", SHAPES)
def test_a_refused_field_is_absent_from_every_read_and_named(world, shape, surface):
    extra, field = SHAPES[shape]
    (world.kb_path("sw") / "x.md").write_text(_file(extra), encoding="utf-8")

    data = SURFACES[surface](world, "x")

    assert data["entry_type"] == "backlog_item"
    # REST spells a field a read does not carry as null; never a value.
    assert data.get(field) is None, f"{field} presented as {data[field]!r}"
    assert field not in (data.get("metadata") or {})
    assert [p["field"] for p in data["load_problems"]] == [field]
    # The same shape as the update warning: field, rule, severity, message.
    assert set(data["load_problems"][0]) == {"field", "rule", "severity", "message"}


@pytest.mark.parametrize("shape", SHAPES)
def test_a_full_echo_of_a_read_leaves_the_files_bytes_alone(world, shape):
    extra, _ = SHAPES[shape]
    path = world.kb_path("sw") / "x.md"
    path.write_text(_file(extra), encoding="utf-8")
    before = path.read_bytes()
    read = _mcp_get(world, "x")

    res = world.mcp()._dispatch_tool("kb_update", {**read, "kb_name": "sw", "entry_id": "x"})

    assert "error" not in res, res
    assert path.read_bytes() == before
    assert "load_problems" in res.get("ignored", [])


def test_a_templated_move_never_runs_on_the_partial_model(make_world):
    world = make_world(
        {
            "kb.yaml": "name: sw\ntypes:\n  backlog_item:\n    subdirectory: 'backlog/{status}'\n",
            "backlog/weird/x.md": _file("status: weird\n"),
        }
    )
    kb = world.kb_path("sw")
    path = kb / "backlog" / "weird" / "x.md"
    before = path.read_text(encoding="utf-8")

    result = world.cli("update", "x", "-k", "sw", "-f", "title=Y")

    assert result.exit_code == 0, result.output
    assert path.read_text(encoding="utf-8") == before.replace("title: T", "title: Y")
    assert sorted(p.relative_to(kb).as_posix() for p in kb.rglob("x.md")) == ["backlog/weird/x.md"]


def test_an_in_process_caller_is_never_handed_a_model_with_a_default_for_a_value(world):
    # `update_entry` / `apply_operations` return an Entry callers read fields
    # from; a result that still has a refused field is refused, not returned.
    from pyrite.services.kb_service import KBService
    from pyrite.storage.database import PyriteDB

    path = world.kb_path("sw") / "x.md"
    path.write_text(_file("rank: .inf\n"), encoding="utf-8")
    svc = KBService(world.config, PyriteDB(world.config.settings.index_path))
    before = path.read_bytes()

    from pyrite.exceptions import ValidationError

    with pytest.raises(ValidationError, match="rank"):
        svc.update_entry("x", "sw", title="Y")
    assert path.read_bytes() == before
    # A repair is the one write that may go through.
    entry = svc.update_entry("x", "sw", rank=3)
    assert entry.load_problems == []
    assert "rank: 3" in path.read_text(encoding="utf-8")


@pytest.mark.control(
    reason="a ratchet: the sites already number three; it fails when a fourth is added"
)
def test_the_partial_model_has_three_construction_sites():
    # A new consumer of a partial model must be read against this file's rules.
    sites = []
    for src in sorted((REPO / "pyrite").rglob("*.py")) + sorted(
        (REPO / "extensions").rglob("src/**/*.py")
    ):
        tree = ast.parse(src.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(
                k.arg == "partial" and isinstance(k.value, ast.Constant) and k.value.value is True
                for k in node.keywords
            ):
                sites.append(
                    f"{src.relative_to(REPO)}:{node.func.attr if hasattr(node.func, 'attr') else '?'}"
                )
    assert [s.split(":")[0] for s in sites] == ["pyrite/services/kb_service.py"] * 3, sites


def test_a_refused_field_spelled_under_metadata_is_named_and_absent(world):
    (world.kb_path("sw") / "x.md").write_text(
        _file("metadata:\n  rank: .inf\n  keep: yes\n"), encoding="utf-8"
    )

    data = _cli_get(world, "x")

    assert [p["field"] for p in data["load_problems"]] == ["metadata.rank"]
    assert "rank" not in data
    assert data["metadata"] == {"keep": "yes"}


def test_the_update_warning_is_the_read_problem(world):
    path = world.kb_path("sw") / "x.md"
    path.write_text(_file("rank: .inf\n"), encoding="utf-8")
    problems = _cli_get(world, "x")["load_problems"]

    result = world.cli("update", "x", "-k", "sw", "-f", "title=Y", "--format", "json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["warnings"] == problems
