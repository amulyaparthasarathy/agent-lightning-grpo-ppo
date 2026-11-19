"""Scope/byte receipt regressions using synthetic filesystem copies only."""
import copy
import hashlib
import json
import shutil
import unittest

import test_preservation_manifest as fixtures
from scripts import backup_qualification as gate
from scripts import preservation_manifest as pm


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PreservationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        data = {name: {"sha256": hashlib.sha256(name.encode()).hexdigest()} for name in gate.DATA_FILES}
        for name in gate.DATA_FILES:
            f.put(gate.DATA + name, name.encode())
        f.put(gate.DATA + "data_manifest.json", json.dumps({"files": data}).encode())
        model = {**gate.EXPECTED_BASE, "weights_files": ["model-00001-of-00001.safetensors"],
                 "weight_bytes": len(b"prepared fixture"),
                 "prepared_tokenizer_config_sha256": hashlib.sha256(b"prepared fixture").hexdigest(),
                 "template_sha256": hashlib.sha256(b"prepared fixture").hexdigest()}
        f.put(gate.BASE + "config.json", json.dumps({"architectures": [gate.EXPECTED_BASE["architecture"]]}).encode())
        f.put(gate.BASE + "model_manifest.json", json.dumps(model).encode())
        f.put(gate.BASE + "model.safetensors.index.json", json.dumps({"weight_map": {"x": model["weights_files"][0]}}).encode())
        f.selection["provenance_records"] = list(gate.ORIGINAL_LINEAGE)
        for path, commit in gate.ORIGINAL_LINEAGE.items():
            f.put(path, json.dumps({"request":{"source":{"commit":commit}}}).encode())
        f.selection["provenance"]["source_identity"] = "fixture-volume"
        destination = f.parent / "destination"
        shutil.copytree(f.root, destination)
        self.sources = {"experiment": f.scan()}
        self.destinations = {"experiment": f.scan(destination, "fixture-readback")}
        # Separate explicit namespaces, even for these tiny fixture roots.
        for name, path in (("repository", "repository"), ("validation", "submissions")):
            selection = {"provenance": dict(f.selection["provenance"], source_identity="fixture-" + name),
                         "selections": [{"group": "evidence", "path": path}]}
            self.sources[name] = pm.scan(f.root, selection, "synthetic_observation", "fixture-" + name, fixtures.STAMP)
            self.destinations[name] = pm.scan(destination, selection, "synthetic_observation", "readback-" + name, fixtures.STAMP)
        self.scope = {"schema_version": 1, "approval_id": "fixture-only", "approved_by": "fixture-owner",
                      "remote_execution_approved": "PENDING", "primary_namespace": "experiment",
                      "repository_namespace": "repository", "validation_namespace": "validation",
                      "expected_base": gate.EXPECTED_BASE,
                      "retained_notices": [gate.BASE + "LICENSE", gate.BASE + "Notice"],
                      "required_git_commits": ["fixture-execution", "fixture-closeout"], "namespaces": {},
                      "destination": {"repo_id": "fixture/private", "private": True,
                                      "revision": "a" * 40, "snapshot_prefix": "snapshots/fixture"}}
        for name, manifest in self.sources.items():
            self.scope["namespaces"][name] = {
                "approved_files": [r["path"] for r in manifest["files"]],
                "required_files": [r["path"] for r in manifest["files"]],
                "source_identity": manifest["provenance"]["source_identity"],
                "expected_provenance": copy.deepcopy(manifest["provenance"]),
                "selections": manifest["selections"], "reviewed_exclusions": manifest["exclusions"],
                "expected_sha256": {r["path"]: r["sha256"] for r in manifest["files"]}}
        bundle = self.sources["repository"]["files"][0]
        self.retrieval = {"destination": self.scope["destination"], "method": "destination_bytes",
                          "fresh_process": True, "original_source_mounted": False,
                          "verifier_id": "fixture-process", "receipt_location": "fixture-independent-store",
                          "git_bundle": {"verified": True, "restored_commits": self.scope["required_git_commits"],
                                         "path": bundle["path"], "sha256": bundle["sha256"]}}
        self.scope["namespaces"]["experiment"]["run_lineage"] = gate.ORIGINAL_LINEAGE
        self.bind()

    def bind(self):
        self.retrieval.update(scope_sha256=gate.digest(self.scope),
                              source_manifests={n: gate.digest(m) for n, m in self.sources.items()},
                              destination_manifests={n: gate.digest(m) for n, m in self.destinations.items()})

    def check(self):
        return gate.qualify(self.scope, self.sources, self.destinations, self.retrieval, fixtures.STAMP)

    def test_complete_synthetic_scope_passes_checks_but_never_real_backup(self):
        result = self.check()
        self.assertTrue(result["scope_checks_passed"], result["failures"])
        self.assertFalse(result["backup_verified"])
        self.scope["remote_execution_approved"] = "YES"
        self.bind()
        self.assertFalse(self.check()["backup_verified"])

    def test_required_file_missing_on_both_sides_even_with_matching_rows(self):
        for path in (pm.CHECKPOINT + "/actor/model_world_size_1_rank_0.pt",
                     gate.BASE + "model-00001-of-00001.safetensors", gate.DATA + "eval_main.jsonl"):
            with self.subTest(path=path):
                sources, destinations = copy.deepcopy(self.sources), copy.deepcopy(self.destinations)
                for collection in (sources, destinations):
                    collection["experiment"]["files"] = [r for r in collection["experiment"]["files"] if r["path"] != path]
                self.assertTrue(pm.compare(sources["experiment"], destinations["experiment"], fixtures.STAMP)["content_match"])
                result = gate.qualify(self.scope, sources, destinations, self.retrieval, fixtures.STAMP)
                self.assertFalse(result["scope_checks_passed"])
                self.assertFalse(result["backup_verified"])
                self.assertTrue(any("coverage" in x or "required native" in x for x in result["failures"]))

    def test_missing_scope_exclusions_unknown_hashes_and_receipt_binding(self):
        mutations = [
            lambda: self.sources["experiment"].update(missing_selections=["runs/failed"]),
            lambda: self.sources["experiment"]["exclusions"].append({"path": "runs/failed", "reason": "unreviewed"}),
            lambda: self.sources["experiment"]["files"][0].update(sha256=None),
            lambda: self.retrieval.update(original_source_mounted=True),
            lambda: self.retrieval.update(method="copied_source_metadata"),
            lambda: self.retrieval.update(destination={"repo_id": "wrong"}),
            lambda: self.retrieval["git_bundle"].update(sha256="b" * 64),
            lambda: self.scope["namespaces"]["experiment"].update(expected_sha256={}),
        ]
        for change in mutations:
            original = copy.deepcopy((self.sources, self.destinations, self.scope, self.retrieval))
            change()
            self.assertFalse(self.check()["scope_checks_passed"])
            self.sources, self.destinations, self.scope, self.retrieval = original

    def test_wrong_missing_base_revision_and_provenance_fail(self):
        for value in (None, "wrong"):
            original = copy.deepcopy(self.sources)
            self.sources["experiment"]["checked_metadata"][gate.BASE+"model_manifest.json"]["revision"] = value
            self.bind()
            self.assertFalse(self.check()["scope_checks_passed"])
            self.sources = original
        self.sources["experiment"]["provenance"]["lab_execution_commit"] = "wrong"
        self.bind()
        self.assertTrue(any("provenance identity" in reason for reason in self.check()["failures"]))

    def test_actual_native_fsdp_metadata_capitalization(self):
        f = self.fixture
        f.put(pm.CHECKPOINT+"/actor/fsdp_config.json", b'{"FSDP_version":1,"world_size":1}')
        self.assertTrue(f.scan()["required_file_findings"]["structurally_complete"])
        f.put(pm.CHECKPOINT+"/actor/fsdp_config.json", b'{"FSDP_version":1,"fsdp_version":2,"world_size":1}')
        self.assertFalse(f.scan()["required_file_findings"]["structurally_complete"])

    def test_platform_root_requires_exact_target_and_member_links_stay_forbidden(self):
        f = self.fixture
        alias = f.parent / "logical-data"
        alias.symlink_to(f.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "approved runtime target"):
            f.scan(alias)
        with self.assertRaisesRegex(ValueError, "differs"):
            pm.scan(alias, f.selection, "synthetic_observation", "fixture", fixtures.STAMP, f.parent)
        result = pm.scan(alias, f.selection, "synthetic_observation", "fixture", fixtures.STAMP, f.root)
        self.assertTrue(all(not r["path"].startswith("/") for r in result["files"]))
        self.assertEqual(result["runtime_root_binding"]["logical"], str(alias))
        (f.root / "runs/escape").symlink_to(f.parent)
        with self.assertRaises(ValueError):
            pm.scan(alias, f.selection, "synthetic_observation", "fixture", fixtures.STAMP, f.root)

    def test_replication_requires_explicit_step20_selection_and_own_marker(self):
        f = self.fixture
        (f.root / "runs/train-01").rename(f.root / "runs/replication-01")
        selection = copy.deepcopy(f.selection)
        selection["native_checkpoint"] = pm.CHECKPOINT.replace("train-01", "replication-01")
        for item in selection["selections"]:
            item["path"] = item["path"].replace("train-01", "replication-01")
        result = pm.scan(f.root, selection, "synthetic_observation", "fixture", fixtures.STAMP)
        self.assertTrue(result["required_file_findings"]["structurally_complete"])
        (f.root / pm.MARKER.replace("train-01", "replication-01")).write_text("15")
        result = pm.scan(f.root, selection, "synthetic_observation", "fixture", fixtures.STAMP)
        self.assertFalse(result["required_file_findings"]["structurally_complete"])


if __name__ == "__main__":
    unittest.main()
