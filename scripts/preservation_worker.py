"""Remote child process: sanitized terminal result, no credential-bearing logs."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
from scripts.preservation_transfer import GateFailure


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    logging.disable(logging.CRITICAL)
    os.environ['HF_DEBUG']='0'
    request=json.loads(Path(a.input).read_text())
    stage='bootstrap'
    try:
        from scripts.preservation_phases import execute
        from scripts.preservation_http import configure
        phase=next(p for p in request['plan']['phases'] if p['name']==request['phase'])
        configure(phase['reservation']['egress_bytes']-64*1024**2)
        from huggingface_hub import HfApi
        api=HfApi(token=os.environ['HF_TOKEN'])
        stage='execute'
        value=execute(request['phase'],request['plan'],request['ledger'],api,
                      lambda v: print(json.dumps({'progress':v}),flush=True))
        result={'success':True,'result':value}
    except Exception as exc:
        # SDK exceptions may contain signed URLs/headers. Never serialize str(exc).
        result={'success':False,'result':{'error_type':type(exc).__name__,'phase':request['phase'],'stage':stage,
                'diagnostic':str(exc) if isinstance(exc,GateFailure) else 'Inspect retained phase records; exception text suppressed'}}
    with Path(a.output).open('x') as f:json.dump(result,f,allow_nan=False)
    return 0 if result['success'] else 1

if __name__=='__main__':
    raise SystemExit(main())
