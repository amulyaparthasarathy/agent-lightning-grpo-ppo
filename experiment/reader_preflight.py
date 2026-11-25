"""Paid CPU preflight in the pinned image. No model weights loaded or GPU used."""
import json
import os
from pathlib import Path
import sys
import importlib.metadata
import importlib.util
import platform
import subprocess

try:
    from . import reader_assets as assets, reader_contract as c, ppo_runtime, launch
except ImportError:
    import reader_assets as assets
    import reader_contract as c
    import ppo_runtime
    import launch


def task_probe(plan, run, tokenizer):
    """The selected task's real probe in this image. Unknown tasks fail closed.

    For recovery-readiness-v1 the bounded prompts are also measured with the
    tokenizer and template actually served, against the declared context limits.
    """
    task = plan['profile']['task']
    if task == 'asset-status-v1':
        try:
            from .asset_probe import check
        except ImportError:
            from asset_probe import check
        return check(run/'asset-preflight.json'), None
    if task == 'recovery-readiness-v1':
        try:
            from . import recovery_probe, recovery_prompt
        except ImportError:
            import recovery_probe
            import recovery_prompt
        receipt = recovery_probe.check(run/'recovery-preflight.json')
        if tokenizer.chat_template != recovery_prompt.prepare_model.effective_template():
            raise ValueError('Served template differs from the rendered template')
        budget = recovery_prompt.budget(lambda text: len(tokenizer.encode(text, add_special_tokens=False)))
        assets.write_new(run/'prompt-budget.json', budget)
        if not budget['within_limits']:
            raise ValueError('Bounded recovery prompts exceed the declared context limits')
        return receipt, budget
    if task != 'calculator-v1':
        raise ValueError('Unknown task; no preflight probe')
    sys.path.insert(0, '/opt/agent-lightning/examples/calc_x')
    os.chdir('/opt/agent-lightning/examples/calc_x')
    from observed_calc_agent import scalar_are_results_same
    if not scalar_are_results_same('2', '2', 1e-2) or scalar_are_results_same('3', '2', 1e-2):
        raise ValueError('Calculator verifier sanity failed')
    try:
        from .reader_calculator import check
    except ImportError:
        from reader_calculator import check
    return check(run/'calculator-preflight.json'), None


def main(plan, run):
    c.validate(plan)
    # Match preparation's canonical Volume paths while retaining nested-symlink rejection.
    scope = 'qual-train-dev' if c.modes.phase(plan) == c.modes.RECOVERY_LEARNING else 'all'
    actual = assets.inspect(*(assets.owned_path('/data', plan['profile']['paths'][key])
                              for key in ('model', 'data', 'staging')), task=plan['profile']['task'], scope=scope)
    if actual != plan['assets']:
        raise ValueError('Preflight assets differ from plan')
    head = subprocess.check_output(['git', '-C', '/opt/agent-lightning', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', '/opt/agent-lightning', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if head != launch.AGL_COMMIT or dirty:
        raise ValueError('Task/verifier training checkout is not the pinned clean source')
    backend = ppo_runtime.source_identity()
    matched_lock = assets.read(Path(__file__).with_name('matched_source_lock.json'))
    for package, files in matched_lock.items():
        root = Path(importlib.util.find_spec(package).origin).parent
        if any(c.file_hash(root/name) != sha for name, sha in files.items()):
            raise ValueError('Matched backend source guard differs')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from validation.compatibility import classify_conflicts, EXPECTED_VERSIONS
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    assets.write_new(run/'runtime-packages.json', {'python': sys.executable,
        'distributions': sorted([{'name': d.metadata['Name'], 'version': d.version,
                                  'location': str(d.locate_file(''))} for d in importlib.metadata.distributions()],
                                key=lambda d: (d['name'] or '', d['version'], d['location']))})
    check = subprocess.run(['uv', 'pip', 'check', '--python', sys.executable], capture_output=True, text=True, timeout=60)
    dependency = classify_conflicts(check.returncode, check.stdout+check.stderr, versions,
        python_minor='.'.join(platform.python_version_tuple()[:2]), machine=platform.machine(), source_commit=launch.AGL_COMMIT)
    assets.write_new(run/'dependencies.json', dependency)
    if not dependency['accepted']:
        raise ValueError('Pinned package identity or reviewed dependency exceptions differ')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(Path('/data')/plan['profile']['paths']['model']), local_files_only=True, trust_remote_code=False)
    if tokenizer.bos_token_id != 128000 or tokenizer.eos_token_id != 128009 or tokenizer.chat_template != assets.prepare_model.effective_template():
        raise ValueError('Loaded tokenizer differs')
    from huggingface_hub import HfApi
    info = HfApi().model_info(c.MODEL, revision=c.BASE_REVISION)
    if info.sha != c.BASE_REVISION:
        raise ValueError('Gated model access/revision not verified')
    import wandb
    # Read project access; actual write/finish is still required independently at every stage.
    api = wandb.Api(timeout=30)
    projects = api.projects(entity=plan['profile']['wandb']['entity'])
    if not any(p.name == plan['profile']['wandb']['project'] for p in projects):
        raise ValueError('W&B project must already exist and be readable')
    calculator, budget = task_probe(plan, run, tokenizer)
    assets.write_new(run/'preflight.json', {'status': 'passed', 'plan_sha256': c.digest(plan),
        'backend': backend, 'dependencies': dependency, 'packages': versions, 'tokenizer_verified': True, 'model_access': True,
        'wandb_project_readable': True, 'task_probe': calculator, 'task': plan['task_identity'],
        **({'prompt_budget': budget} if budget else {}),
        'model_inference': 'unqualified', 'training': 'unqualified', 'reload': 'unqualified'})


if __name__ == '__main__':
    main(assets.read(sys.argv[1]), Path(sys.argv[2]))
