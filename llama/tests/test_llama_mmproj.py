#!/usr/bin/env python3
import contextlib
import importlib.util
import io
import os
import tempfile
import unittest

LLAMA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import sys

if LLAMA_DIR not in sys.path:
    sys.path.insert(0, LLAMA_DIR)

from llama_mmproj import (  # noqa: E402
    find_mmproj,
    is_mmproj_filename,
    iter_chat_ggufs,
    llama_switch_command,
    resolve_model_filename,
)


def _load_switch():
    path = os.path.join(LLAMA_DIR, "llama-switch.py")
    spec = importlib.util.spec_from_file_location("llama_switch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(b"GGUF")
    return path


UNCEN_GGUF = "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf"
UNCEN_MMPROJ = "mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
FARA_GGUF = "Fara1.5-27B-Q4_K_M.gguf"
FARA_MMPROJ = "Fara1.5-27.mmproj-q8_0.gguf"
BASE_QWEN_GGUF = "Qwen3.8-27B-Q4_K_M.gguf"
TEXT_GGUF = "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"


class MmprojFilenameTests(unittest.TestCase):
    def test_detects_prefix_and_infix(self):
        self.assertTrue(is_mmproj_filename(UNCEN_MMPROJ))
        self.assertTrue(is_mmproj_filename(FARA_MMPROJ))
        self.assertTrue(is_mmproj_filename("mmproj-BF16.gguf"))

    def test_chat_weights_are_not_mmproj(self):
        self.assertFalse(is_mmproj_filename(UNCEN_GGUF))
        self.assertFalse(is_mmproj_filename(FARA_GGUF))
        self.assertFalse(is_mmproj_filename(BASE_QWEN_GGUF))


class DiscoveryTests(unittest.TestCase):
    def test_uncensored_same_dir_unsloth_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _touch(
                os.path.join(
                    tmp,
                    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF",
                    UNCEN_GGUF,
                )
            )
            mmproj = _touch(
                os.path.join(
                    tmp,
                    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF",
                    UNCEN_MMPROJ,
                )
            )
            self.assertEqual(find_mmproj(model, models_dir=tmp), mmproj)

    def test_generic_mmproj_bf16_in_same_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _touch(os.path.join(tmp, "Qwen3.8-27B-GGUF", BASE_QWEN_GGUF))
            mmproj = _touch(os.path.join(tmp, "Qwen3.8-27B-GGUF", "mmproj-BF16.gguf"))
            self.assertEqual(find_mmproj(model, models_dir=tmp), mmproj)

    def test_fara_infix_mmproj_same_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _touch(os.path.join(tmp, "Fara1.5-27B-GGUF", FARA_GGUF))
            mmproj = _touch(os.path.join(tmp, "Fara1.5-27B-GGUF", FARA_MMPROJ))
            self.assertEqual(find_mmproj(model, models_dir=tmp), mmproj)

    def test_text_only_when_no_mmproj(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _touch(os.path.join(tmp, "Qwen3.6-35B-A3B-GGUF", TEXT_GGUF))
            self.assertIsNone(find_mmproj(model, models_dir=tmp))

    def test_does_not_attach_uncensored_mmproj_to_base_qwen(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = _touch(os.path.join(tmp, "Qwen3.8-27B-GGUF", BASE_QWEN_GGUF))
            _touch(
                os.path.join(
                    tmp,
                    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF",
                    UNCEN_GGUF,
                )
            )
            _touch(
                os.path.join(
                    tmp,
                    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF",
                    UNCEN_MMPROJ,
                )
            )
            self.assertIsNone(find_mmproj(base, models_dir=tmp))

    def test_prefers_same_dir_over_other_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            fara = _touch(os.path.join(tmp, "Fara1.5-27B-GGUF", FARA_GGUF))
            fara_mm = _touch(os.path.join(tmp, "Fara1.5-27B-GGUF", FARA_MMPROJ))
            _touch(
                os.path.join(
                    tmp,
                    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF",
                    UNCEN_MMPROJ,
                )
            )
            self.assertEqual(find_mmproj(fara, models_dir=tmp), fara_mm)

    def test_flat_gguf_finds_conventional_unsloth_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _touch(os.path.join(tmp, UNCEN_GGUF))
            mmproj = _touch(
                os.path.join(
                    tmp,
                    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF",
                    UNCEN_MMPROJ,
                )
            )
            self.assertEqual(find_mmproj(model, models_dir=tmp), mmproj)

    def test_mirror_env_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            primary = os.path.join(tmp, "fast")
            mirror = os.path.join(tmp, "mirror")
            rel = os.path.join(
                "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF", UNCEN_GGUF
            )
            model = _touch(os.path.join(primary, rel))
            mmproj = _touch(
                os.path.join(
                    mirror,
                    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF",
                    UNCEN_MMPROJ,
                )
            )
            old = os.environ.get("LLAMA_MODELS_DIR_MIRROR")
            os.environ["LLAMA_MODELS_DIR_MIRROR"] = mirror
            try:
                self.assertEqual(find_mmproj(model, models_dir=primary), mmproj)
            finally:
                if old is None:
                    os.environ.pop("LLAMA_MODELS_DIR_MIRROR", None)
                else:
                    os.environ["LLAMA_MODELS_DIR_MIRROR"] = old

    def test_known_map_when_mmproj_is_only_under_models_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Weights live in a nested quant dir; mmproj uses the documented
            # Unsloth relative path from KNOWN_MMPROJ_BY_GGUF.
            model = _touch(os.path.join(tmp, "quant", UNCEN_GGUF))
            mmproj = _touch(
                os.path.join(
                    tmp,
                    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF",
                    UNCEN_MMPROJ,
                )
            )
            self.assertEqual(find_mmproj(model, models_dir=tmp), mmproj)

    def test_prefers_bf16_over_q8_in_same_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _touch(
                os.path.join(
                    tmp,
                    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF",
                    UNCEN_GGUF,
                )
            )
            q8 = _touch(
                os.path.join(
                    tmp,
                    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF",
                    "mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf",
                )
            )
            bf16 = _touch(
                os.path.join(
                    tmp,
                    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF",
                    UNCEN_MMPROJ,
                )
            )
            self.assertEqual(find_mmproj(model, models_dir=tmp), bf16)
            self.assertNotEqual(find_mmproj(model, models_dir=tmp), q8)


class CatalogTests(unittest.TestCase):
    def test_mmproj_files_are_not_chat_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            chat = _touch(os.path.join(tmp, "Fara1.5-27B-GGUF", FARA_GGUF))
            _touch(os.path.join(tmp, "Fara1.5-27B-GGUF", FARA_MMPROJ))
            _touch(
                os.path.join(
                    tmp,
                    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF",
                    UNCEN_MMPROJ,
                )
            )
            found = list(iter_chat_ggufs(tmp))
            self.assertEqual(found, [chat])
            self.assertTrue(all(not is_mmproj_filename(os.path.basename(p)) for p in found))

    def test_alias_qwen38_uncensored(self):
        installed = [UNCEN_GGUF, FARA_GGUF, BASE_QWEN_GGUF]
        self.assertEqual(
            resolve_model_filename("qwen3.8-uncensored", installed), UNCEN_GGUF
        )
        self.assertEqual(resolve_model_filename("fara1.5", installed), FARA_GGUF)
        self.assertEqual(
            resolve_model_filename(
                "external::litellm::Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf",
                installed,
            ),
            UNCEN_GGUF,
        )


class SwitchArgvTests(unittest.TestCase):
    def test_proxy_passes_mmproj_flag(self):
        self.assertEqual(
            llama_switch_command(UNCEN_GGUF, "/tmp/" + UNCEN_MMPROJ),
            ["llama-switch", UNCEN_GGUF, "--mmproj", "/tmp/" + UNCEN_MMPROJ],
        )
        self.assertEqual(llama_switch_command(TEXT_GGUF, None), ["llama-switch", TEXT_GGUF])

    def test_llama_server_cmd_appends_mmproj(self):
        switch = _load_switch()
        cmd = switch.build_llama_server_cmd(
            "/models/" + UNCEN_GGUF,
            mmproj_path="/models/" + UNCEN_MMPROJ,
            extra_args=["-ngl", "99"],
            api_key_file="/no/such/key",
            llama_server_bin="llama-server",
        )
        self.assertIn("--mmproj", cmd)
        self.assertEqual(cmd[cmd.index("--mmproj") + 1], "/models/" + UNCEN_MMPROJ)
        self.assertEqual(cmd[cmd.index("-m") + 1], "/models/" + UNCEN_GGUF)

    def test_text_only_omits_mmproj(self):
        switch = _load_switch()
        cmd = switch.build_llama_server_cmd(
            "/models/" + TEXT_GGUF,
            mmproj_path=None,
            extra_args=["-ngl", "99"],
            api_key_file="/no/such/key",
            llama_server_bin="llama-server",
        )
        self.assertNotIn("--mmproj", cmd)

    def test_strips_stale_mmproj_from_extra_args(self):
        switch = _load_switch()
        stripped = switch.strip_mmproj_args(
            ["-ngl", "99", "--mmproj", "/old/fara.mmproj.gguf", "-fa", "on"]
        )
        self.assertEqual(stripped, ["-ngl", "99", "-fa", "on"])

    def test_dry_run_uncensored_includes_mmproj(self):
        switch = _load_switch()
        with tempfile.TemporaryDirectory() as tmp:
            _touch(
                os.path.join(
                    tmp,
                    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF",
                    UNCEN_GGUF,
                )
            )
            mmproj = _touch(
                os.path.join(
                    tmp,
                    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF",
                    UNCEN_MMPROJ,
                )
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = switch.main(
                    [
                        "qwen3.8-uncensored",
                        "--models-dir",
                        tmp,
                        "--dry-run",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "8081",
                    ]
                )
            self.assertEqual(rc, 0)
            printed = buf.getvalue()
            self.assertIn("--mmproj", printed)
            self.assertIn(mmproj, printed)

    def test_dry_run_fara_still_gets_mmproj(self):
        switch = _load_switch()
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "Fara1.5-27B-GGUF", FARA_GGUF))
            mmproj = _touch(os.path.join(tmp, "Fara1.5-27B-GGUF", FARA_MMPROJ))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = switch.main(["fara1.5", "--models-dir", tmp, "--dry-run"])
            self.assertEqual(rc, 0)
            self.assertIn("--mmproj", buf.getvalue())
            self.assertIn(mmproj, buf.getvalue())

    def test_refuses_mmproj_as_chat_model(self):
        switch = _load_switch()
        with tempfile.TemporaryDirectory() as tmp:
            mmproj = _touch(os.path.join(tmp, UNCEN_MMPROJ))
            with self.assertRaises(SystemExit):
                switch.resolve_gguf(mmproj, models_dir=tmp)


if __name__ == "__main__":
    unittest.main()
