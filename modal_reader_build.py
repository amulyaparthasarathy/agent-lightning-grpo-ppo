"""Bounded reader-only final layer; kept after the shared expensive CUDA layers."""


def install_matched_transformers(wheel_url, sha256):
    import subprocess
    subprocess.run(['uv', 'pip', 'install', '--python', '/opt/agent-lightning/.venv/bin/python',
                    '--no-deps', '--no-build', wheel_url+'#sha256='+sha256], check=True, timeout=850)
