"""Read-only H100 admission evidence for the current HERO v3. No paid execution path.

Use the Modal 1.6.4 interpreter. The only local input is the *allocation*
metadata of the historical, successfully completed archive decoder job; its
image was built separately from the mounted inputs. Do not supply a dataset.
Outputs omit tokens, member names, unrelated app names and account balances.
"""
import argparse
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import sys

import modal
from google.protobuf.empty_pb2 import Empty
from modal._utils.function_utils import parse_gpu_config
from modal.client import _Client
from modal.exception import PermissionDeniedError
from modal_proto import api_pb2 as api


IMAGE_ID = 'im-1aUX0dAUmmZQ583Ohg7Icu'
REQUIRED_PACKAGES = {'torch': '2.5.1+cu124', 'numpy': '1.26.4', 'scipy': '1.14.1'}
HOURS = Decimal(300) / Decimal(3600)
RATE_KEYS = ('gpu_hour_cost_h100', 'cpu_hour_cost_sandbox', 'mem_gib_hour_cost_sandbox',
             'egress_gib_cost', 'volume_storage_gib_month_cost')
APP_NAME = 'rsmi-hero-full-v3'


def digest(raw):
    return sha256(raw).hexdigest()


def cli(*args):
    command = [str(Path(sys.executable).with_name('modal')), *args, '--json']
    return json.loads(subprocess.check_output(command, timeout=20))


async def resources(prior_app):
    # Fixed, read-only RPCs. In particular, no ImageGetOrCreate, AppGetOrCreate,
    # SandboxCreate, scheduling, secret lookup, image publish or content reads.
    stub = (await _Client.from_env())._stub
    identity = await stub.WorkspaceNameLookup(Empty())
    envs = await stub.EnvironmentList(Empty())
    environments = []
    for e in envs.items:
        try:
            b = await stub.EnvironmentGetBudget(api.EnvironmentGetBudgetRequest(environment_id=e.environment_id))
        except PermissionDeniedError as error:
            budget = {'read': 'permission_denied', 'detail': str(error)}
        else:
            budget = {'read': 'success', 'scope': 'compute_only',
                      'cycle_budget_dollars': b.cycle_budget_dollars if b.HasField('cycle_budget_dollars') else None}
        items = cli('app', 'list', '--env', e.name)
        environments.append({'name_sha256': digest(e.name.encode()), 'environment_budget': budget,
                             'named_images': len(cli('image', 'names', 'list', '--env', e.name)),
                             'visible_apps': len(items),
                             'visible_study_apps': sum(a['description'] == APP_NAME for a in items),
                             'deployed_nonstudy_apps': sum(a['state'] == 'deployed' and a['description'] != APP_NAME for a in items)})
    layout = await stub.AppGetLayout(api.AppGetLayoutRequest(app_id=prior_app))
    if IMAGE_ID not in {o.object_id for o in layout.app_layout.objects}:
        raise ValueError('historical app does not attest expected image ID')
    image = await stub.ImageFromId(api.ImageFromIdRequest(image_id=IMAGE_ID))
    chain = await stub.ImageBuildChainGet(api.ImageBuildChainGetRequest(image_id=IMAGE_ID))
    packages = {k: image.metadata.python_packages.get(k) for k in REQUIRED_PACKAGES}
    if image.image_id != IMAGE_ID or packages != REQUIRED_PACKAGES:
        raise ValueError('prebuilt image ID or package metadata differs')
    return identity.workspace_name or identity.username, environments, {'immutable_id': image.image_id, 'packages': packages,
                          'python': image.metadata.python_version_info.split(' | ')[0],
                          'workdir': image.metadata.workdir, 'build_steps': len(chain.build_steps),
                          'exists_in_probed_workspace': True}


def probe(allocation):
    if version('modal') != '1.6.4':
        raise ValueError('re-audit SDK semantics before using a different Modal version')
    raw = allocation.read_bytes()
    prior = json.loads(raw)
    if set(prior) != {'app_id', 'call_id'} or not re.fullmatch(r'ap-[A-Za-z0-9]+', prior['app_id']):
        raise ValueError('expected two-field historical allocation metadata')
    profiles = cli('profile', 'list')
    active = [p for p in profiles if p['active']]
    if len(active) != 1:
        raise ValueError('cannot attribute authenticated probe to one active profile')
    ws = modal.Workspace.from_context()
    workspace, environments, image = asyncio.run(resources(prior['app_id']))
    if workspace != active[0]['workspace']:
        raise ValueError('current credentials and selected workspace differ')
    owners = [m for m in cli('workspace', 'members', 'list') if m['name'] == active[0]['name']]
    billing = ws.billing.summary(cycle='2026-11')
    rates = ws.billing.rates()
    gpu = parse_gpu_config('H100!')
    h = rates['gpu_hour_cost_h100']; c = rates['cpu_hour_cost_sandbox']; m = rates['mem_gib_hour_cost_sandbox']
    # Account hourly rates are more precise than rounded public seconds.
    # Illustrations only: the size of returned files is not a NIC byte limit.
    e = rates['egress_gib_cost'] * Decimal(1) / Decimal(1024)
    second = lambda cost: str(cost.quantize(Decimal('.00000001')))
    return {'schema': 'hero-full-v3/modal-h100-read-only/v1',
            'source_class': 'authenticated control-plane metadata only; neither physical nor synthetic observations',
            'utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'probe_sha256': digest(Path(__file__).read_bytes()), 'sdk_version': version('modal'),
            'workspace_sha256': digest(workspace.encode()), 'configured_profile_count': len(profiles),
            'active_profile_name_matches_member_role': [o['role'] for o in owners],
            'environments': environments,
            'billing_cycle': '2026-11', 'workspace_metered_gross_exceeds_25': billing.metered_cost > 25,
            'workspace_metered_noncompute_present': any(v > 0 for k, v in billing.metered_cost_breakdown.items() if k.lower().replace(' ', '_') in ('network_egress', 'volumes')),
            'workspace_precredit_usage_limit_returned_by_summary': False,
            'allocation_metadata_sha256': digest(raw), 'preexisting_image': image,
            'local_gpu_request_encoding': {'gpu_type': gpu.gpu_type, 'count': gpu.count},
            'authenticated_tariff_usd': {k: str(rates[k]) for k in RATE_KEYS},
            'illustrative_300s_one_core_four_gib_not_invoice_or_cap': {
                'compute_usd': second(HOURS * (h + c + 4*m)),
                'compute_plus_1MiB_egress_usd': second(HOURS * (h + c + 4*m) + e),
                'triple_GPU_rate_sensitivity_plus_1MiB_usd': second(HOURS * (3*h + c + 4*m) + e)},
            'paid_calls_issued_by_probe': 0, 'new_sandbox_id': None, 'device_or_availability_observed': False,
            'admission': 'NOT_RELEASED; read-only command cannot authorize or launch'}


if __name__ == '__main__':
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument('--prior-allocation', required=True, type=Path)
    print(json.dumps(probe(args.parse_args().prior_allocation), indent=2, allow_nan=False))
