"""CPU-only W&B/Hugging Face/disconnection test. See VALIDATION.md."""
from pathlib import Path
import sys
import modal
from validation_support import SECRET_NAME, VOLUME_NAME, execute, request, submitted, require_detached, source_identity

LOCAL = Path(__file__).resolve().parent
app = modal.App("agent-lightning-validation-services")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
credentials = modal.Secret.from_name(SECRET_NAME, required_keys=["HF_TOKEN", "WANDB_API_KEY"])
# Deliberately independent of modal_app: this test cannot trigger the CUDA build.
# Familiar client versions from the CS224R runtime; this is NOT the training image.
image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install("wandb==0.24.2", "huggingface-hub==0.36.2")
         .add_local_file(LOCAL / "validation/runtime.py", "/opt/validation/runtime.py")
         .add_local_python_source("validation_support"))


@app.function(image=image, secrets=[credentials], volumes={"/validation": volume},
              cpu=1, memory=2048, timeout=1800, retries=0, max_containers=1,
              single_use_containers=True)
def services(req: dict):
    return execute(req, sys.executable, volume, timeout_seconds=1500)


@app.local_entrypoint()
def main(name: str, wandb_entity: str = "amulyap-stanford-university", hold_seconds: int = 180):
    require_detached()
    req = request("services", name, wandb_entity, hold_seconds)
    req["source"] = source_identity()
    call = services.spawn(req)
    submitted(app, call, req)
