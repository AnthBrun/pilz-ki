import json

from pilzki import prepare, vocab


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def make_data(tmp_path):
    d = tmp_path / "pilzdaten"
    d.mkdir()
    write_jsonl(d / "index_arts.jsonl", [{"slug": s, "name": s} for s in ("A", "B", "C")])
    write_jsonl(d / "filter_tags.jsonl", [
        {"slug": "A", "tags": ["lamellen", "rot"]},
        {"slug": "B", "tags": ["kugel"]},
        {"slug": "C", "tags": ["holz"]},
    ])
    write_jsonl(d / "species.jsonl", [
        {"slug": "A", "merkmale": {}, "bilddateien": [f"daten/details/A-Dateien/image{i}.jpg" for i in range(10)] + ["x.gif"]},
        {"slug": "B", "merkmale": {"Lamellen/Leisten": "..."}, "bilddateien": ["daten/details/B-Dateien/image1.jpg", "daten/details/B-Dateien/shared.jpg"]},
        {"slug": "C", "merkmale": {}, "bilddateien": ["daten/details/C-Dateien/shared.jpg", "daten/details/C-Dateien/dup1.jpg", "daten/details/C-Dateien/dup2.jpg"]},
        {"slug": "Meta", "merkmale": {}, "bilddateien": ["daten/details/Meta-Dateien/image1.jpg"]},
    ])
    (d / "originals.json").write_text("{}")
    sha = tmp_path / "sha.txt"
    lines = [f"{i:x}{0:063x}  ./daten/details/A-Dateien/image{i}.jpg" for i in range(10)]
    lines += [f"{'b' * 64}  ./daten/details/B-Dateien/image1.jpg", f"{'5' * 64}  ./daten/details/B-Dateien/shared.jpg",
              f"{'5' * 64}  ./daten/details/C-Dateien/shared.jpg", f"{'d' * 64}  ./daten/details/C-Dateien/dup1.jpg",
              f"{'d' * 64}  ./daten/details/C-Dateien/dup2.jpg"]
    sha.write_text("\n".join(lines) + "\n")
    return d, sha


def test_build(tmp_path):
    d, sha = make_data(tmp_path)
    out = tmp_path / "out"
    report = prepare.build(d, None, sha, out, holdout_frac=0.0)
    species = {s["slug"]: s for s in json.loads((out / "species.json").read_text())}
    assert set(species) == {"A", "B", "C"}  # meta pages are not species
    assert "hutpilz" in species["A"]["tags"]  # derived from lamellen
    assert {"leisten", "hutpilz"} <= set(species["B"]["tags"])  # derived from merkmale key
    rows = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    assert report["dropped"] == {"shared_between_species": 2, "duplicate_within_species": 1}
    assert len(rows) == 10 + 1 + 1
    assert len({r["id"] for r in rows}) == len(rows)
    for slug in "ABC":  # every species keeps a training photo
        assert any(r["slug"] == slug and r["split"] == "train" for r in rows)


def test_vocab_tags_unique_and_requires_known():
    assert len(vocab.TAGS) == len(set(vocab.TAGS))
    for g in vocab.GROUPS:
        assert g.requires is None or g.requires in vocab.TAGS


def test_scientific_genus():
    from pilzki.labels import scientific_genus

    assert scientific_genus("BOLETUS EDULIS") == "BOLETUS"
    assert scientific_genus("  Amanita muscaria var. aureola") == "AMANITA"
    assert scientific_genus("") == ""
    assert scientific_genus("Kombucha – Pilz (ASCOMYCOTA)") == "KOMBUCHA"
