from __future__ import annotations

import os
import stat
import dill
import pytest

from mass2.core import safe_dill


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Redirect default_config_dir to a clean temporary directory for every test."""
    config_dir = tmp_path / "mass2_config"
    config_dir.mkdir()
    monkeypatch.setattr(safe_dill, "default_config_dir", lambda app_name="mass2": config_dir)
    return config_dir


def _sample_recipe_func(x, scale=2.5):
    return x * scale + 1.0


def test_key_creation_and_permissions(isolated_config):
    key_path = isolated_config / "hmac_signing.key"
    assert not key_path.exists()

    key1 = safe_dill.get_or_create_signing_key()
    assert len(key1) == 32
    assert key_path.exists()

    # Subsequent calls must return the exact same key
    key2 = safe_dill.get_or_create_signing_key()
    assert key1 == key2

    # Check owner-only permissions on POSIX systems
    if os.name == "posix":
        mode = stat.S_IMODE(key_path.stat().st_mode)
        assert mode == 0o600


def test_dumps_and_loads_roundtrip_function(isolated_config):
    payload_obj = {
        "name": "calibration_step",
        "func": _sample_recipe_func,
        "lambda": lambda arr: arr[::2],
    }

    blob = safe_dill.dumps_signed(payload_obj)
    assert blob.startswith(safe_dill._MAGIC_HEADER)

    restored = safe_dill.loads_signed(blob)
    assert restored["name"] == "calibration_step"
    assert restored["func"](4.0) == pytest.approx(11.0)
    assert restored["lambda"]([10, 20, 30, 40]) == [10, 30]


def test_dump_and_load_file_roundtrip(tmp_path, isolated_config):
    recipe_file = tmp_path / "recipe.dill"
    safe_dill.dump_signed(_sample_recipe_func, recipe_file)

    loaded_fn = safe_dill.load_signed(recipe_file)
    assert loaded_fn(2.0, scale=3.0) == pytest.approx(7.0)


def test_tampered_payload_rejected_by_default(isolated_config):
    blob = bytearray(safe_dill.dumps_signed({"threshold": 100}))
    # Flip a byte inside the serialized payload region (past the 40-byte header)
    blob[-1] ^= 0xFF
    tampered = bytes(blob)

    with pytest.raises(safe_dill.UntrustedPayloadError, match="verification failed"):
        safe_dill.loads_signed(tampered)


def test_tampered_mac_rejected_by_default(isolated_config):
    blob = bytearray(safe_dill.dumps_signed({"threshold": 100}))
    # Flip a byte inside the 32-byte HMAC digest (bytes 8..40)
    blob[10] ^= 0xFF
    tampered = bytes(blob)

    with pytest.raises(safe_dill.UntrustedPayloadError, match="verification failed"):
        safe_dill.loads_signed(tampered)


def test_legacy_unsigned_dill_blocked_then_allowed(tmp_path, isolated_config):
    legacy_file = tmp_path / "legacy_unsigned.dill"
    legacy_file.write_bytes(dill.dumps({"legacy": True, "fn": _sample_recipe_func}))

    with pytest.raises(safe_dill.UntrustedPayloadError, match="unsigned dill payload"):
        safe_dill.load_signed(legacy_file)

    restored = safe_dill.load_signed(legacy_file, allow_untrusted=True)
    assert restored["legacy"] is True
    assert restored["fn"](2.0) == pytest.approx(6.0)


def test_foreign_key_rejected_unless_installed_or_opted_in(tmp_path, isolated_config):
    # Simulate a lab acquisition computer with its own separate signing key
    lab_key_file = tmp_path / "daq_fridge_1.key"
    safe_dill.get_or_create_signing_key(key_path=lab_key_file)

    shared_recipe = tmp_path / "fridge1_recipe.dill"
    safe_dill.dump_signed(_sample_recipe_func, shared_recipe, key_path=lab_key_file)

    # 1. Fails on researcher's machine before installing the lab key
    with pytest.raises(safe_dill.UntrustedPayloadError):
        safe_dill.load_signed(shared_recipe)

    # 2. Explicit allow_untrusted=True works as an immediate override
    fn_override = safe_dill.load_signed(shared_recipe, allow_untrusted=True)
    assert fn_override(2.0) == pytest.approx(6.0)

    # 3. Installing the lab key into trusted_keys/ allows seamless loading without flags
    installed_path = safe_dill.install_trusted_key(lab_key_file, name="daq-fridge-1")
    assert installed_path.name == "daq-fridge-1.key"

    fn_trusted = safe_dill.load_signed(shared_recipe)
    assert fn_trusted(2.0) == pytest.approx(6.0)


def test_multiple_trusted_keys_all_accepted(tmp_path, isolated_config):
    # Ensure local primary key exists too
    safe_dill.get_or_create_signing_key()

    recipes = []
    for idx in range(3):
        inst_key = tmp_path / f"instrument_{idx}.key"
        safe_dill.get_or_create_signing_key(key_path=inst_key)

        recipe_path = tmp_path / f"recipe_{idx}.dill"
        safe_dill.dump_signed(f"payload_{idx}", recipe_path, key_path=inst_key)
        recipes.append(recipe_path)

        safe_dill.install_trusted_key(inst_key)

    # Primary key + 3 installed keys = 4 total active keys
    assert len(safe_dill.list_trusted_keys()) == 4

    for idx, recipe_path in enumerate(recipes):
        assert safe_dill.load_signed(recipe_path) == f"payload_{idx}"


def test_install_trusted_key_validation_and_overwrite(tmp_path, isolated_config):
    # Non-existent file
    with pytest.raises(FileNotFoundError):
        safe_dill.install_trusted_key(tmp_path / "does_not_exist.key")

    # Wrong byte length (e.g., user accidentally passed a recipe file instead of a key)
    bad_key = tmp_path / "not_a_key.key"
    bad_key.write_bytes(b"too short")
    with pytest.raises(ValueError, match="Expected a 32-byte HMAC key"):
        safe_dill.install_trusted_key(bad_key)

    # Valid key installation + duplicate protection + overwrite + path traversal safety
    valid_key = tmp_path / "source.key"
    safe_dill.get_or_create_signing_key(key_path=valid_key)

    dest = safe_dill.install_trusted_key(valid_key, name="../evil_traversal_name")
    assert dest.parent == isolated_config / "trusted_keys"
    assert dest.name == "evil_traversal_name.key"

    if os.name == "posix":
        assert stat.S_IMODE(dest.stat().st_mode) == 0o600

    with pytest.raises(FileExistsError):
        safe_dill.install_trusted_key(valid_key, name="evil_traversal_name")

    # Overwriting with new key bytes succeeds
    new_key = tmp_path / "rotated.key"
    safe_dill.get_or_create_signing_key(key_path=new_key)
    dest_overwritten = safe_dill.install_trusted_key(new_key, name="evil_traversal_name", overwrite=True)
    assert dest_overwritten.read_bytes() == new_key.read_bytes()
