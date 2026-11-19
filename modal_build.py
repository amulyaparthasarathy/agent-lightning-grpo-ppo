"""Importable image-build entrypoints; stdlib only, no application dependencies.

Modal imports the defining module before calling a run_function build step.
Keep this module independent of modal_app and the not-yet-copied runtime bundle.
"""


def build_training_stack():
    import subprocess
    subprocess.run(["bash", "/opt/agl-build/install_modal_training.sh"], check=True)


def finalize_training_stack():
    import subprocess
    subprocess.run(["bash", "/opt/agl-build/finalize_modal_training.sh"], check=True)
