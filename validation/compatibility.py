"""Fail-closed, version-qualified audit of the image-02 dependency exceptions.

These exceptions permit *diagnostics*, not a clean dependency closure or a
successful RL training claim. Only the specific six image-02 conflicts qualify.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

UPSTREAM_COMMIT = "cbc56e41e0eab643aa71d195e92e5bba652fa683"
EXPECTED_PYTHON = "3.12"
EXPECTED_MACHINE = "x86_64"

# Explicit environment identity: major frameworks + every offending package and
# its dependency. The CUDA build suffix on torch is intentionally not stripped.
EXPECTED_VERSIONS = {
    "agentlightning": "1.0.3",
    "verl": "0.8.0",
    "vllm": "0.22.0",
    "torch": "2.11.0+cu130",
    "flash-attn": "2.8.3",
    "autogen-agentchat": "0.7.5",
    "autogen-ext": "0.7.5",
    "weave": "0.54.1",
    "ml-dtypes": "0.6.0",
    "opencv-python-headless": "5.0.0.93",
    "xgrammar": "0.2.5.post1",
    "google-api-core": "2.44.0",
    "tensorboard": "2.21.0",
    "proto-plus": "1.29.0",
    "numpy": "1.26.4",
    # PyArrow 26 rejects NumPy 1.x at import time even though uv pip check
    # cannot infer this from published Requires-Dist metadata.
    "pyarrow": "25.0.1",
    "datasets": "5.2.0",
    "transformers": "5.22.2",
    "protobuf": "5.29.6",
}

# Exact UV 0.12.13 diagnostics observed in image-02. Requirement text and
# installed target version are compared without dropping environment markers.
REVIEWED_CONFLICTS = frozenset({
    ("ml-dtypes", "numpy>=2.0.0", "1.26.4"),
    ("opencv-python-headless", "numpy>=2 ; python_full_version >= '3.9'", "1.26.4"),
    ("google-api-core", "protobuf>=6.33.5,<8.0.0", "5.29.6"),
    ("tensorboard", "protobuf>=6.31.1,<8.0.0", "5.29.6"),
    ("xgrammar", "transformers>=4.38.0,<5", "5.22.2"),
    ("proto-plus", "protobuf>=6.33.5,<8.0.0", "5.29.6"),
})

# uv pip check can prepend an interpreter banner in the remote worker.
# Accept exactly this pinned venv banner, not arbitrary extra output.
_VENV_HEADER = re.compile(
    r"Using Python 3\.12\.\d+ environment at: /opt/agent-lightning/\.venv"
)
_CHECKED = re.compile(r"Checked \d+ packages in \d+(?:\.\d+)?(?:ms|s)")
_FOUND = re.compile(r"Found (\d+) incompatibilit(?:y|ies)")
_CONFLICT = re.compile(
    r"The package `([^`]+)` requires `([^`]+)`, but `([^`]+)` is installed"
)


def classify_conflicts(
    returncode: int,
    report: str,
    installed_versions: Mapping[str, str],
    *,
    python_minor: str,
    machine: str,
    source_commit: str,
) -> dict:
    """Return an audit record. Unknown output or version drift fails closed."""
    normalized = {name.lower().replace("_", "-"): version
                  for name, version in installed_versions.items()}
    drift = {name: {"expected": expected, "actual": normalized.get(name)}
             for name, expected in EXPECTED_VERSIONS.items()
             if normalized.get(name) != expected}
    identity = {
        "python_minor": python_minor,
        "machine": machine,
        "upstream_commit": source_commit,
        "versions_checked": dict(EXPECTED_VERSIONS),
        "version_drift": drift,
    }
    identity_ok = (python_minor == EXPECTED_PYTHON
                   and machine == EXPECTED_MACHINE
                   and source_commit == UPSTREAM_COMMIT
                   and not drift)
    lines = [line.strip() for line in report.splitlines() if line.strip()]
    # The build-time uv check may omit this line; the explicit --python invocation
    # in the remote worker prints it. Preserve strictness for all other banners.
    if lines and _VENV_HEADER.fullmatch(lines[0]):
        lines = lines[1:]
    parsed = []
    parse_ok = (len(lines) >= 2 and _CHECKED.fullmatch(lines[0]) is not None)
    closure_consistent = False
    if parse_ok and returncode == 0:
        parse_ok = len(lines) == 2 and lines[1] == "All installed packages are compatible"
        closure_consistent = bool(parse_ok)
    elif parse_ok and returncode == 1:
        count = _FOUND.fullmatch(lines[1])
        parsed = [match.groups() for line in lines[2:]
                  if (match := _CONFLICT.fullmatch(line))]
        parse_ok = bool(count and int(count.group(1)) == len(parsed)
                        and len(lines) == len(parsed) + 2
                        and len(set(parsed)) == len(parsed))
    else:
        parse_ok = False

    observed = set(parsed)
    # Only a *complete* exact match (or a legitimately clean environment) can pass.
    reviewed_exactly = bool(parse_ok and returncode == 1
                            and observed == REVIEWED_CONFLICTS)
    accepted = bool(identity_ok and (closure_consistent or reviewed_exactly))
    return {
        "accepted": accepted,
        "dependency_closure_consistent": closure_consistent,
        "reviewed_exception_qualified": bool(accepted and reviewed_exactly),
        "qualification": ("dependency_clean" if accepted and closure_consistent else
                          "diagnostic_with_reviewed_dependency_exceptions" if accepted else
                          "rejected"),
        "uv_pip_check_returncode": returncode,
        "report_parsed": bool(parse_ok),
        "environment_identity": identity,
        "identity_matches": bool(identity_ok),
        "observed_conflicts": [
            {"package": package, "requirement": requirement, "installed": version,
             "package_version": normalized.get(package)}
            for package, requirement, version in sorted(observed)
        ],
        "reviewed_dependency_conflicts": len(observed) if reviewed_exactly else 0,
        "unreviewed_or_changed_conflicts": [list(row) for row in sorted(observed - REVIEWED_CONFLICTS)],
        "missing_reviewed_conflicts": [list(row) for row in sorted(REVIEWED_CONFLICTS - observed)]
           if returncode != 0 else [],
    }
