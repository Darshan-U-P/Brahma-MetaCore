from __future__ import annotations
import argparse, hashlib, json, random, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / 'checkpoints' / 'milestone2_v02' / 'best_validation.pt'
DEFAULT_OUTPUT = ROOT / 'reports' / 'metacore_construction_stage_a_v01'

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def build_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model

def main():
    p = argparse.ArgumentParser(description='Validate residual-stage transfer mechanics; no training.')
    p.add_argument('--source-checkpoint', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--stage', choices=['layer1','layer2','layer3','layer4'], default='layer2')
    p.add_argument('--seed', type=int, default=9001)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    a = p.parse_args()
    if a.batch_size < 1: p.error('--batch-size must be positive')
    src, out = a.source_checkpoint.resolve(), a.output_dir.resolve()
    if not src.is_file(): raise FileNotFoundError(src)
    if out.exists(): raise FileExistsError(f'Refusing to overwrite: {out}')
    if a.device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    device = torch.device('cuda' if a.device == 'cuda' or (a.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    seed_all(a.seed); started = time.perf_counter(); source_hash = sha256(src)
    ckpt = torch.load(src, map_location='cpu', weights_only=True)
    expected_arch = 'torchvision_resnet18_cifar10_stem'
    if ckpt.get('architecture') != expected_arch: raise ValueError(f"Unexpected architecture: {ckpt.get('architecture')!r}")
    source_state = ckpt.get('model_state_dict')
    if not isinstance(source_state, dict): raise KeyError('Missing model_state_dict')
    source_model = build_model(); source_model.load_state_dict(source_state, strict=True)
    target = build_model(); prefix = a.stage + '.'
    stage_state = {k:v for k,v in source_state.items() if k.startswith(prefix)}
    target_state = target.state_dict()
    target_keys = {k for k in target_state if k.startswith(prefix)}
    if not stage_state or set(stage_state) != target_keys: raise RuntimeError('Source/target stage keys differ')
    copied = []
    for key, value in stage_state.items():
        if target_state[key].shape != value.shape: raise RuntimeError(f'Shape mismatch: {key}')
        target_state[key] = value.detach().clone()
        copied.append({'name':key, 'shape':list(value.shape), 'numel':int(value.numel())})
    target.load_state_dict(target_state, strict=True)
    after = target.state_dict()
    for key, value in stage_state.items():
        if not torch.equal(after[key], value): raise RuntimeError(f'Copied tensor mismatch: {key}')
    target.to(device).eval()  # Keep copied BatchNorm running statistics unchanged during smoke test.
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(a.batch_size, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (a.batch_size,), device=device)
    target.zero_grad(set_to_none=True); logits = target(x)
    if logits.shape != (a.batch_size,10) or not torch.isfinite(logits).all(): raise RuntimeError('Invalid forward output')
    loss = nn.CrossEntropyLoss()(logits, labels); loss.backward()
    params = dict(target.named_parameters()); bad = []; grad_count = 0
    for name, param in params.items():
        if param.grad is not None:
            grad_count += 1
            if not torch.isfinite(param.grad).all(): bad.append(name)
    if bad: raise RuntimeError('Non-finite gradients: ' + ', '.join(bad))
    stage_grad = all(param.grad is not None for name,param in params.items() if name.startswith(prefix))
    bn = []
    for name, module in target.named_modules():
        if name.startswith(a.stage) and isinstance(module, nn.BatchNorm2d):
            bn.append({'module':name,'channels':module.num_features,
                'running_mean_finite':bool(torch.isfinite(module.running_mean).all()),
                'running_var_finite':bool(torch.isfinite(module.running_var).all()),
                'num_batches_tracked':int(module.num_batches_tracked.item())})
    copied_numel = sum(z['numel'] for z in copied)
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None
    # Create the output directory only after compatibility and forward/backward checks pass.
    out.mkdir(parents=True, exist_ok=False)
    init_path, json_path, md_path = out/'target_initialization.pt', out/'stage_a_report.json', out/'stage_a_report.md'
    torch.save({'model_state_dict':{k:v.detach().cpu().clone() for k,v in target.state_dict().items()},
        'architecture':expected_arch,'source_checkpoint':str(src),'source_checkpoint_sha256':source_hash,
        'transferred_stage':a.stage,'seed':a.seed,
        'note':'No training performed; non-stage tensors remain randomly initialized.'}, init_path)
    report = {'experiment':'Brahma MetaCore Construction Stage A v0.1','status':'passed',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_checkpoint':str(src),
        'source_checkpoint_sha256':source_hash,'source_seed':ckpt.get('seed'),'source_epoch':ckpt.get('epoch'),
        'source_validation_accuracy':ckpt.get('validation_accuracy'),'architecture':expected_arch,
        'transferred_stage':a.stage,'target_seed':a.seed,'device':str(device),'batch_size':a.batch_size,
        'forward_output_shape':list(logits.shape),'forward_logits_finite':bool(torch.isfinite(logits).all()),
        'smoke_test_loss':float(loss.detach().cpu()),'backward_pass_finite':not bad,
        'gradient_tensor_count':grad_count,'transferred_stage_gradients_present':stage_grad,
        'batchnorm_buffers':bn,'copied_tensor_count':len(copied),'copied_tensor_numel':copied_numel,
        'total_model_parameter_numel':sum(z.numel() for z in target.parameters()),
        'peak_cuda_memory_allocated_bytes':peak,'elapsed_seconds':time.perf_counter()-started,
        'copied_tensors':copied,'initialization_checkpoint':str(init_path),
        'limitations':['Compatibility smoke test only; no training or efficacy claim.',
          'Stem, other stages and classifier remain randomly initialized.',
          'Transferred stage includes BatchNorm affine parameters and running buffers.',
          'Stage-boundary activation mismatch remains possible.',
          'No evidence yet of faster convergence, better accuracy or reduced pretraining.']}
    json_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines = ['# MetaCore Construction Stage A v0.1','','**Status:** Passed','','## Configuration',
      f'- Source checkpoint: `{src}`',f'- Source SHA-256: `{source_hash}`',f'- Stage: `{a.stage}`',
      f'- Target seed: `{a.seed}`',f'- Device: `{device}`','','## Checks',f'- Copied stage tensors: {len(copied)}',
      f'- Copied tensor elements: {copied_numel:,}',f'- Output shape: `{list(logits.shape)}`',
      f'- Finite logits: {bool(torch.isfinite(logits).all())}',f'- Smoke-test loss: {float(loss.detach().cpu()):.6f}',
      f'- Backward pass finite: {not bad}',f'- Transferred-stage gradients present: {stage_grad}',
      f'- Peak CUDA memory bytes: {peak if peak is not None else "not measured (CPU)"}','','## BatchNorm state','',
      '| Module | Channels | Mean finite | Variance finite | num_batches_tracked |','|---|---:|---|---|---:|']
    for z in bn: lines.append(f"| `{z['module']}` | {z['channels']} | {z['running_mean_finite']} | {z['running_var_finite']} | {z['num_batches_tracked']} |")
    lines += ['','## Interpretation','', 'This confirms the mechanics of copying a complete residual stage and running a finite forward/backward pass. It does not test transfer efficacy. BatchNorm state is copied with the stage; non-stage tensors remain random. Stage-boundary activation mismatch remains possible.','','## Limitations']
    lines += ['- '+s for s in report['limitations']]
    md_path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Stage A checks passed.')
    print(f'Stage: {a.stage} | Device: {device} | copied tensors: {len(copied)}')
    print(f'Copied tensor elements: {copied_numel:,} | output: {list(logits.shape)}')
    print(f'Smoke-test loss: {float(loss.detach().cpu()):.6f}')
    print(f'JSON report: {json_path}\nMarkdown report: {md_path}')
    print('Existing checkpoints and reports were not modified.')

if __name__ == '__main__': main()
