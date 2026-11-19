"""Audit the real training image without training. See VALIDATION.md."""
from pathlib import Path
import modal
from modal_app import image as training_image, PYTHON
from validation_support import SECRET_NAME, VOLUME_NAME, execute, request, submitted, require_detached, source_identity

LOCAL = Path(__file__).resolve().parent
app = modal.App("agent-lightning-validation-training")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
credentials = modal.Secret.from_name(SECRET_NAME, required_keys=["HF_TOKEN", "WANDB_API_KEY"])
# Reuse the exact existing image recipe. Do not reimplement its dependency stack.
image = (training_image
         .add_local_file(LOCAL / "validation/runtime.py", "/opt/validation/runtime.py")
         .add_local_file(LOCAL / "validation/compatibility.py", "/opt/validation/compatibility.py")
         .add_local_python_source("validation_support", "modal_app"))


@app.function(image=image, secrets=[credentials], volumes={"/validation": volume},
              cpu=4, memory=16384, timeout=1800, retries=0, max_containers=1,
              single_use_containers=True)
def image_check(req: dict):
    return execute(req, str(PYTHON), volume, timeout_seconds=1500)


@app.function(image=image, secrets=[credentials], volumes={"/validation": volume},
              gpu="H100!", cpu=4, memory=16384, timeout=1200, retries=0,
              max_containers=1, single_use_containers=True)
def gpu_check(req: dict):
    return execute(req, str(PYTHON), volume, timeout_seconds=900)


@app.local_entrypoint()
def main(name: str, kind: str = "image", wandb_entity: str = "amulyap-stanford-university"):
    if kind not in {"image", "gpu"}:
        raise ValueError("Use --kind image or --kind gpu")
    require_detached()
    req = request(kind, name, wandb_entity)
    req["source"] = source_identity()
    call = (image_check if kind == "image" else gpu_check).spawn(req)
    submitted(app, call, req)
