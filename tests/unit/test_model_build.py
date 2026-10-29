from pathlib import Path

import pytest

from amigo.model import Model, _BUILT_MODULE_DIGESTS, _write_text_if_changed


def test_generated_source_is_not_rewritten_when_contents_are_unchanged(tmp_path):
    source_path = tmp_path / "generated.cpp"
    source_path.write_text("int main() {}", encoding="utf-8")
    original_mtime = source_path.stat().st_mtime_ns

    _write_text_if_changed(source_path, "int main() {}")

    assert source_path.stat().st_mtime_ns == original_mtime


def test_generated_source_is_rewritten_when_contents_change(tmp_path):
    source_path = tmp_path / "generated.cpp"

    _write_text_if_changed(source_path, "old")
    _write_text_if_changed(source_path, "new")

    assert source_path.read_text(encoding="utf-8") == "new"


def test_model_reuses_identical_compiled_module_in_the_same_process(
    monkeypatch, tmp_path
):
    module_name = "cached_module_test"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(_BUILT_MODULE_DIGESTS, module_name, raising=False)
    build_calls = []

    def generate_cpp(self):
        Path(f"{module_name}.h").write_text("header", encoding="utf-8")
        Path(f"{module_name}.cpp").write_text("source", encoding="utf-8")

    def build_module(self, source_dir, build_dir, debug=False):
        build_calls.append(module_name)

    monkeypatch.setattr(Model, "generate_cpp", generate_cpp)
    monkeypatch.setattr(Model, "_build_module", build_module)

    first = Model(module_name)
    first.build_module(comm=None, source_dir=tmp_path, build_dir=tmp_path / "build")
    second = Model(module_name)
    second.build_module(comm=None, source_dir=tmp_path, build_dir=tmp_path / "build")

    assert build_calls == [module_name]
    assert first._built and second._built


def test_model_rejects_changed_source_for_a_loaded_module(monkeypatch, tmp_path):
    module_name = "changed_module_test"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(_BUILT_MODULE_DIGESTS, module_name, raising=False)
    source = ["first"]

    def generate_cpp(self):
        Path(f"{module_name}.h").write_text("header", encoding="utf-8")
        Path(f"{module_name}.cpp").write_text(source[0], encoding="utf-8")

    monkeypatch.setattr(Model, "generate_cpp", generate_cpp)
    monkeypatch.setattr(Model, "_build_module", lambda *args, **kwargs: None)

    Model(module_name).build_module(
        comm=None, source_dir=tmp_path, build_dir=tmp_path / "build"
    )
    source[0] = "changed"

    with pytest.raises(RuntimeError, match="unique module name"):
        Model(module_name).build_module(
            comm=None, source_dir=tmp_path, build_dir=tmp_path / "build"
        )
