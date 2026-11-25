# A02 CPU preflight and reader image correction

A02 prepared and verified fresh assets from `d66b5d112587f36ca1a2708577d9218f7db39e29`. Preparation plan `97afeb534d1b4c2140a805ba36b6238e366dd703e8000218b3fd50aad81112d4` produced asset-bound plan `3d03a28a35fb903d380262a0254c3ebbd73f203fa891ac2f9c6820b5a3466e2d`. The CPU preflight failed the existing dependency check. The prepare and preflight calls were terminal, both apps stopped and the owned-container count reached zero. There was no launch. The four stages recorded zero episodes, rewards, updates and checkpoints.

The cold image selected two newer transitive wheels. The recorded package inventory contained `ml-dtypes 0.7.0` and `opencv-python-headless 5.1.0.95`. The reviewed exception requires `0.6.0` and `5.0.0.93` respectively. All six conflicts were still present, but the version check correctly rejected both substitutions. The historical warm image had the reviewed versions.

The final reader image layer installs the exact CPython 3.12/Linux x86-64 wheels by URL and SHA-256 alongside the pinned Transformers wheel. The installer uses the training interpreter, binary wheels and disabled dependency resolution. The build correction leaves the dependency classifier and scientific runtime intact.

For local reproduction, `AGL_READER_WHEELS` names a directory containing the four public Linux wheels for the failed and reviewed versions plus the pinned Transformers wheel. Run with the installed reader client and uv.

```bash
PYTHONPATH=.:tests python -m unittest -v test_reader_dependency_wheels test_dependency_exceptions
bash scripts/test_reader_offline.sh
```

The focused suite ran 17 tests with all passing. It installed the actual failed wheels in a temporary venv, reproduced rejection, verified the production overlay's wheel hashes, installed the reviewed wheels, and reran the unchanged classifier. The local host staged Linux x86-64 target wheels explicitly. This exercises the metadata used by preflight, without running their accelerator code. A fresh provider preflight is still required.
