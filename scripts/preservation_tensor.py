"""Isolated tensor subprocess; no Hub authentication or transport in this interpreter."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

SAFE_GATES={'native/base config semantics differ','native model hash mismatch',
            'installed merger source differs from reviewed public v0.8.0; inspect before conversion',
            'only FSDP1/world-size-1 supported','fresh conversion output required',
            'prepared base identity mismatch','unequal tied aliases',
            'prepared tokenizer IDs differ from frozen Llama recipe','native token IDs differ from VERL tokenizer override',
            'prepared token configuration differs','ambiguous RoPE representation','ambiguous RoPE theta',
            'native/base generation semantics differ','generation stop tokens differ','installed VERL token override source differs'}


def invoke(python, action, arguments, timeout):
    from scripts.preservation_transfer import require
    require(action in {'convert', 'clean_load'}, 'unknown tensor action')
    with tempfile.TemporaryDirectory(prefix='tensor-request-') as scratch:
        request, receipt = Path(scratch)/'request.json', Path(scratch)/'result.json'
        request.write_text(json.dumps({'action':action, 'arguments':arguments}))
        # Tensor code reads only verified local files. Keep publishing/training
        # credentials out of this subprocess and force offline model resolution.
        env = {k:v for k,v in os.environ.items() if k not in {'HF_TOKEN','WANDB_API_KEY'}}
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME=str(Path(scratch)/'cache'))
        child = subprocess.run([str(python), '-m', 'scripts.preservation_tensor',
                                '--input', str(request), '--output', str(receipt)],
                               env=env, timeout=timeout, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        require(receipt.is_file(), 'tensor subprocess failed before result')
        result = json.loads(receipt.read_text())
        diagnostic=result.get('diagnostic')
        detail=diagnostic if diagnostic in SAFE_GATES else result.get('error_type','unknown')
        require(child.returncode == 0 and result['success'], 'tensor subprocess failed: '+detail)
        return result['result']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    try:
        request = json.loads(Path(args.input).read_text())
        if request['action'] == 'convert':
            from scripts.convert_fsdp1 import convert
            value = convert(**request['arguments'])
        elif request['action'] == 'clean_load':
            from scripts.preservation_release import clean_load
            value = clean_load(**request['arguments'])
        else:
            raise ValueError('unknown tensor action')
        result = {'success':True, 'result':value}
    except Exception as exc:
        # Never emit arbitrary exception strings, environment or traceback locals.
        result = {'success':False, 'error_type':type(exc).__name__}
        # Known fixed gate messages identify preservation failures without
        # serializing arbitrary SDK/Torch exceptions or checkpoint key contents.
        if type(exc).__name__=='GateFailure' and str(exc) in SAFE_GATES:
            result['diagnostic']=str(exc)
    with Path(args.output).open('x') as f: json.dump(result, f, allow_nan=False)
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
