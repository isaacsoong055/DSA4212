"""The BO archive round-trips: manifest hashes match, content is byte-identical, zip is deterministic."""
import hashlib
import zipfile

import pandas as pd
import pytest

from archive import archive_bo


def test_archive_roundtrip(tmp_path):
    raw = tmp_path / "raw" / "primary_far_w0.3"
    raw.mkdir(parents=True)
    (raw / "bo__bo_default__seed1000.csv").write_text("a,b\n1,2\n3,4\n")
    (raw / "cmaes__cma_default__seed1000.csv").write_text("a,b\n9,9\n")  # not archived
    z1, m1 = archive_bo("t", raw_dir=tmp_path / "raw", out_dir=tmp_path / "out")
    man = pd.read_csv(m1)
    assert list(man["file"]) == ["primary_far_w0.3/bo__bo_default__seed1000.csv"] and man["rows"][0] == 2
    with zipfile.ZipFile(z1) as z:
        data = z.read(man["file"][0])
    assert hashlib.sha256(data).hexdigest() == man["sha256"][0] and data == b"a,b\n1,2\n3,4\n"
    assert z1.name == "raw_bo_t.zip" and m1.name == "raw_bo_t_manifest.csv"
    z2, _ = archive_bo("t2", raw_dir=tmp_path / "raw", out_dir=tmp_path / "out2")
    assert z1.read_bytes() == z2.read_bytes()  # deterministic
    with pytest.raises(FileExistsError):       # an existing archive is never rewritten
        archive_bo("t", raw_dir=tmp_path / "raw", out_dir=tmp_path / "out")
