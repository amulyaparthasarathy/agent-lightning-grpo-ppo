"""Shared image recipe; no apps, Volumes, registries or Secrets are declared here."""
from pathlib import Path
import json
import shlex
import modal
from modal_build import build_training_stack, finalize_training_stack

LOCAL = Path(__file__).resolve().parent
BUNDLE = Path('/opt/agent-lightning-lab/experiment')
CUDA_IMAGE = "nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d"
RUNTIME_FILES = (
    "ppo_contract.py", "ppo_math.py", "ppo_runtime.py", "ppo_source_lock.json",
    "check_gpu.py", "launch.py", "export_artifact.py", "observed_calc_agent.py", "prepare_model.py",
    "prepare_data.py", "summarize.py", "requirements-agent.txt", "LICENSE-AgentLightning.txt",
    "templates/tool_chat_template_llama3.1_json.jinja", "templates/PROVENANCE.json",
    "templates/LICENSE-vllm.txt",
)

def training_image():
    image = (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
        .apt_install("build-essential", "git", "curl", "unzip", "pkg-config", "ca-certificates")
        .uv_pip_install("uv==0.12.13", uv_version="0.12.13")
        # Present before either builder is discovered; no experiment/app source here.
        .add_local_file(LOCAL / "modal_build.py", "/root/modal_build.py", copy=True)
        .add_local_file(LOCAL / "scripts/install_modal_training.sh", "/opt/agl-build/install_modal_training.sh", copy=True)
        .run_function(build_training_stack, cpu=8, memory=65536, timeout=7200, include_source=False)
        # Changes below this line reuse a successfully built FlashAttention layer.
        .add_local_file(LOCAL / "scripts/finalize_modal_training.sh", "/opt/agl-build/finalize_modal_training.sh", copy=True)
        .add_local_file(LOCAL / "experiment/requirements-agent.txt", "/opt/agl-build/requirements-agent.txt", copy=True)
        .run_function(finalize_training_stack, cpu=4, memory=16384, timeout=1800, include_source=False)
    )
    # Only named runtime inputs are copied; credentials, local venvs, and datasets are not uploaded.
    for relative in RUNTIME_FILES:
        image = image.add_local_file(LOCAL / "experiment" / relative, str(BUNDLE / relative), copy=True)
    image = image.add_local_file(LOCAL / 'modal_image.py', '/root/modal_image.py', copy=True)
    return image

def matched_image(reader=False):
    image = training_image()
    wheel = json.loads((LOCAL/"experiment/ppo_source_lock.json").read_text())["transformers_wheel"]
    if reader:
        from modal_reader_build import install_matched_transformers
        image = image.add_local_file(LOCAL/'modal_reader_build.py', '/root/modal_reader_build.py', copy=True)
        image = image.run_function(install_matched_transformers, args=(wheel['url'], wheel['digests']['sha256']),
                                   cpu=(4, 4), memory=(16384, 16384), timeout=900, include_source=False)
    else:
        image = image.run_commands(
            "uv pip install --python /opt/agent-lightning/.venv/bin/python --no-deps --no-build "
            + shlex.quote(wheel["url"]+"#sha256="+wheel["digests"]["sha256"]))
    for path in sorted((LOCAL/"experiment").rglob("*")):
        if (path.is_file() and path.suffix in (".py", ".json", ".jinja", ".txt")
                and "__pycache__" not in path.parts
                and str(path.relative_to(LOCAL/"experiment")) not in RUNTIME_FILES):
            image = image.add_local_file(path, str(BUNDLE/path.relative_to(LOCAL/"experiment")), copy=True)
    return image
