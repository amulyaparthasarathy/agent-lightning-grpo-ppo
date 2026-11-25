"""Bounded reader-only final layer; kept after the shared expensive CUDA layers."""


def install_matched_transformers(wheel_url, sha256):
    import subprocess
    # Restore the exact CPython 3.12/Linux x86_64 wheels admitted by the
    # reader's dependency gate, even when the upstream cold build selects newer releases.
    reviewed_wheels = (
        'https://files.pythonhosted.org/packages/c7/f9/7d76c1eae866f5d4636401b31b6d6dd90e4b4ced1fa7cfdfcca9c60e4bd3/'
        'ml_dtypes-0.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
        '#sha256=3b4a480aa8fd54a1805b8ac10f3f91763926a74f73c0c364c10f9231854f4170',
        'https://files.pythonhosted.org/packages/9b/21/f6ef335f6e65724aa78b8d792b48d40a48c381715f1e62f5a5049e09d07e/'
        'opencv_python_headless-5.0.0.93-cp37-abi3-manylinux_2_28_x86_64.whl'
        '#sha256=ed709fdf9aa0bd1f2ed8549e71d19449b03a675bb581eb292285f6861953be37',
    )
    subprocess.run(['uv', 'pip', 'install', '--python', '/opt/agent-lightning/.venv/bin/python',
                    '--no-deps', '--no-build', wheel_url+'#sha256='+sha256, *reviewed_wheels], check=True, timeout=850)
