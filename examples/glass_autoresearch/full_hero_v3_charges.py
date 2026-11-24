"""Read-only Modal gross resource charges for terminated V3 sandboxes, never launches a job."""
import argparse
import json
from datetime import datetime,timedelta
from decimal import Decimal
from pathlib import Path

import modal


def read(p): return json.loads(p.read_bytes())


def collect(stage,root):
    p=root/f'h100-{stage}-1'; ident=read(p/'provider-id.json'); term=read(p/'termination.json')
    if ident['sandbox_id']!=term['sandbox_id'] or not isinstance(term['terminated_exit_code'],int):
        raise RuntimeError('Terminal receipt and sandbox identity must match')
    sid=ident['sandbox_id']
    sb=modal.Sandbox.from_id(sid); tags={'study':'hero-full-v3','cell':stage+'-1'}
    if sb.get_tags()!=tags or sb.poll() is None: raise RuntimeError('Unknown object or stage not terminal')
    from full_hero_sandbox import APP_NAME
    app=modal.App.lookup(APP_NAME,environment_name='main',create_if_missing=False)
    if app.app_id!=ident['app_id']: raise RuntimeError('Unexpected billing app identity')
    lo=datetime.fromisoformat(ident['at_utc']).replace(minute=0,second=0,microsecond=0)
    hi=datetime.fromisoformat(term['at_utc']).replace(minute=0,second=0,microsecond=0)+timedelta(hours=1)
    records=list(app.sandboxes.stats(since=lo,until=hi,group_by_tags=['cell'],tags=tags)); skus={}; total=Decimal(0)
    for r in records:
        if r.tags!= {'cell':stage+'-1'}: raise RuntimeError('Unexpected aggregation scope')
        total+=r.cost
        for k,v in r.cost_by_resource.items(): skus[k]=skus.get(k,Decimal(0))+v
    if not skus.get('H100',0)>0: raise RuntimeError('No attributed H100 amount published yet; keep complete cap reserved')
    return dict(scope='Modal App.sandboxes.stats; terminated, attributed gross, not an invoice or a credit. Never add App and Sandbox lines.',sandbox_id=sid,app_id=app.app_id,
                interval_hour_utc=[lo.isoformat(),hi.isoformat()],tags=tags,reported_usd={k:str(v) for k,v in skus.items()},total_metered_usd=str(total),
                h100_usd=str(skus['H100']),h100_equivalent_device_s=float(skus['H100']*3600/Decimal('3.95')),whole_create_to_terminate_s=term['whole_wall_s'],
                missing_egress_note='No additionally metered network SKU unless present in reported_usd; exclusive gross cap still reserved.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['main','refit'],required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(collect(a.stage,a.root),indent=2))
