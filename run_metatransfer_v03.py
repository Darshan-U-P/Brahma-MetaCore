"""MetaTransfer v0.3: new-target-seed replication on CIFAR-10.1 v6.
Run from repository root: python -m src.experiments.run_metatransfer_v03 --audit-only
then: python -m src.experiments.run_metatransfer_v03
Preserves existing reports, checkpoints, source data, and v0.1/v0.2 caches.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from src.experiments import run_metaanalyzer_v04 as v04
from src.experiments import run_metaanalyzer_v06 as v06
from src.experiments import run_metatransfer_v01 as v01
from src.experiments import run_metatransfer_v02 as v02

ROOT = Path(__file__).resolve().parents[2]
V01_REPORT = ROOT / 'reports/metatransfer_v01/transfer_report.json'
V02_REPORT = ROOT / 'reports/metatransfer_v02_external_run2'
SOURCE_CACHE_DEFAULT = ROOT / 'reports/metatransfer_v01/cache/source_score_effects_seed42_n1250_5741985755da.npz'
SPLIT_FILE = ROOT / 'checkpoints/milestone2_v02/split_indices_v02.npz'
DATA_ROOT = ROOT / 'data/cifar10'
OUT_DIR = ROOT / 'reports/metatransfer_v03_new_seeds'
CACHE_DIR = OUT_DIR / 'cache'
TARGETS = {
    'seed2027': ROOT / 'checkpoints/metatransfer_v03_seed2027/best_validation.pt',
    'seed31415': ROOT / 'checkpoints/metatransfer_v03_seed31415/best_validation.pt',
}
PRIMARY = .10
RANDOM_MAPPINGS = 1000
CONTROL_SEED = 20261018
BATCH_SIZE = 128
WIDTHS = {'layer2.1.bn1': 128, 'layer3.1.bn1': 256, 'layer4.1.bn1': 512}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def sha_indices(a):
    return hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()


def safe(x: Any):
    if isinstance(x, dict): return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [safe(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, Path): return str(x)
    return x


def read_json(path):
    if not path.is_file(): raise FileNotFoundError(f'Missing required file: {path}')
    obj = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(obj, dict): raise ValueError(f'Expected JSON object: {path}')
    return obj


def source_inputs():
    report = read_json(V01_REPORT)
    if report.get('experiment') != 'Brahma MetaCore MetaTransfer v0.1' or report.get('source_model') != 'seed42':
        raise ValueError('Unexpected frozen v0.1 report/source model.')
    info = report['source_score_cache']
    path = v06.resolve_from_root(info['path'])
    digest = sha(path)
    if info.get('sha256') and info['sha256'] != digest: raise ValueError('Frozen source-cache hash mismatch.')
    payload = v01.read_payload(path)
    effects = payload.get('effects')
    if not isinstance(effects, dict): raise ValueError('Frozen source cache has no effects.')
    for layer, width in WIDTHS.items():
        for c in range(width):
            item = effects.get(f'{layer}|{c}')
            if not isinstance(item, dict) or 'abs_delta_loss' not in item or 'signed_delta_loss' not in item:
                raise ValueError(f'Missing frozen source effect {layer}|{c}')
    return report, payload, path, digest


def feature_indices():
    report = read_json(v06.DEFAULT_OUTPUT)
    if report.get('pilot') is not False or report.get('feature_examples') != 2500:
        raise ValueError('Expected full v0.6 run with 2,500 feature images.')
    with np.load(SPLIT_FILE, allow_pickle=False) as z: val = z['val_indices'].copy()
    if len(val) != 5000: raise ValueError(f'Expected 5,000 validation indices, got {len(val)}')
    idx = np.random.default_rng(v06.SEED).permutation(val)[:2500]
    if report.get('feature_indices_sha256') and sha_indices(idx) != report['feature_indices_sha256']:
        raise ValueError('Feature indices do not match the frozen v0.6 report.')
    return idx


def source_signatures():
    path = v06.CACHE_DIR / 'full_f2500_i2500_seed42_signatures.npz'
    payload = v01.read_payload(path)
    out = {}
    for layer, width in WIDTHS.items():
        arr = np.asarray(payload[layer], dtype=np.float64)
        if arr.shape != (2500, width) or not np.isfinite(arr).all(): raise ValueError(f'Bad source signatures: {layer} {arr.shape}')
        out[layer] = arr
    return out


def signatures_for_target(key, checkpoint, idx, base_dataset, device):
    ckhash = sha(checkpoint); ihash = sha_indices(idx)
    path = CACHE_DIR / f'signatures_{key}_n2500_{ckhash[:12]}_{ihash[:10]}.npz'
    meta = {'experiment':'MetaTransfer v0.3 target signatures','target_model':key,'checkpoint':str(checkpoint.relative_to(ROOT)),
            'checkpoint_sha256':ckhash,'examples':2500,'feature_indices_sha256':ihash,'layers':list(v06.LAYERS)}
    if path.exists():
        payload = v01.read_payload(path)
        if payload.get('metadata') != meta: raise ValueError(f'Signature-cache metadata mismatch; preserving {path}')
        sig = {l:np.asarray(payload['signatures'][l], dtype=np.float64) for l in v06.LAYERS}
        print('Reusing signature cache:', path.relative_to(ROOT))
    else:
        loader = v06.make_loader(base_dataset, idx, device)
        model, ckpt = v06.load_model(checkpoint, device)
        print(f'Extracting signatures for {key}; checkpoint epoch={ckpt.get("epoch", -1)}')
        sig = v06.extract_signatures(model, loader, device)
        del model
        if device.type == 'cuda': torch.cuda.empty_cache()
        payload = {'metadata':meta,'signatures':{l:sig[l].tolist() for l in v06.LAYERS}}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if path.exists(): raise FileExistsError(f'Refusing to overwrite {path}')
        v06.save_cache(path, payload)
    for l,w in WIDTHS.items():
        if sig[l].shape != (2500,w) or not np.isfinite(sig[l]).all(): raise ValueError(f'Invalid {key} signatures for {l}')
    return sig, {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'metadata':meta}


def target_effects(key, checkpoint, images, labels, hashes, device):
    ckhash = sha(checkpoint)
    path = CACHE_DIR / f'external_effects_{key}_n2000_{ckhash[:12]}_{hashes["images_sha256"][:10]}_{hashes["labels_sha256"][:10]}.npz'
    meta = {'experiment':'MetaTransfer v0.3 external effects','dataset':'CIFAR-10.1 v6','target_model':key,
            'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':ckhash,'examples':2000,
            'images_sha256':hashes['images_sha256'],'labels_sha256':hashes['labels_sha256'],
            'normalization_mean':list(v04.MEAN),'normalization_std':list(v04.STD)}
    if path.exists():
        payload = v01.read_payload(path)
        if payload.get('metadata') != meta: raise ValueError(f'Effect-cache metadata mismatch; preserving {path}')
        print('Reusing external-effect cache:', path.relative_to(ROOT))
    else:
        ds = v02.ArrayCIFAR10Dataset(images, labels)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type=='cuda'), drop_last=False)
        model, ckpt = v06.load_model(checkpoint, device)
        print(f'Measuring external ablations for {key}; epoch={ckpt.get("epoch", -1)}')
        effects, baseline = v06.measure_channel_effects(model, loader, device, f'{key}/CIFAR-10.1-v6', channels_per_layer=None)
        del model
        if device.type == 'cuda': torch.cuda.empty_cache()
        payload = {'metadata':meta,'effects':effects,'baseline':baseline}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if path.exists(): raise FileExistsError(f'Refusing to overwrite {path}')
        v06.save_cache(path, payload)
    if int(payload.get('baseline',{}).get('examples',-1)) != 2000: raise ValueError(f'Bad external baseline for {key}')
    for l,w in WIDTHS.items():
        for c in range(w):
            item=payload.get('effects',{}).get(f'{l}|{c}')
            if not isinstance(item,dict) or int(item.get('examples',-1)) != 2000: raise ValueError(f'Missing external effect {key}/{l}/{c}')
    return payload, {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'baseline':payload['baseline'],'metadata':meta}


def random_control(actual, source, aligned, same, n, rng):
    k=max(1,int(np.ceil(len(actual)*PRIMARY))); key='top_10pct'
    actual_top=set(np.argsort(actual)[-k:].tolist()); ap=aligned['topk'][key]['precision']; sp=same['topk'][key]['precision']
    vals=np.empty(n)
    for i in range(n):
        mapping=rng.permutation(len(actual)); pred=np.empty(len(actual)); pred[mapping]=source
        vals[i]=len(actual_top & set(np.argsort(pred)[-k:].tolist()))/k
    return {'n_random_mappings':n,'precision_mean':float(vals.mean()),'precision_std':float(vals.std(ddof=1)),
            'precision_2_5_percentile':float(np.quantile(vals,.025)),'precision_97_5_percentile':float(np.quantile(vals,.975)),
            'empirical_p_random_precision_ge_aligned':float((1+np.sum(vals>=ap))/(n+1)),
            'empirical_p_random_precision_ge_same_index':float((1+np.sum(vals>=sp))/(n+1)),
            'note':'Permutation reference only; not a population-level significance test.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--random-mappings',type=int,default=RANDOM_MAPPINGS)
    p.add_argument('--seed',type=int,default=CONTROL_SEED)
    p.add_argument('--output',default=None)
    a=p.parse_args()
    if a.random_mappings<100: p.error('--random-mappings must be at least 100')
    out=v06.resolve_from_root(a.output) if a.output else OUT_DIR/'transfer_report.json'; md=out.with_suffix('.md')
    if not a.audit_only and (out.exists() or md.exists()): raise FileExistsError(f'Refusing to overwrite reports: {out} / {md}')
    old_v01, src_payload, src_path, src_hash=source_inputs()
    old_v02=read_json(V02_REPORT)
    idx=feature_indices(); src_sig=source_signatures()
    images,labels,data_hashes=v02.validate_dataset_files()
    if old_v02.get('dataset',{}).get('images_sha256')!=data_hashes['images_sha256'] or old_v02.get('dataset',{}).get('labels_sha256')!=data_hashes['labels_sha256']:
        raise ValueError('CIFAR-10.1 hashes differ from validated v0.2 report.')
    for key,cp in TARGETS.items():
        if not cp.is_file(): raise FileNotFoundError(f'Missing target checkpoint: {cp}')
        ckpt=torch.load(cp,map_location='cpu',weights_only=False)
        if int(ckpt.get('seed',-1))!=int(key.removeprefix('seed')) or int(ckpt.get('epoch',-1))!=20:
            raise ValueError(f'Checkpoint metadata mismatch for {key}: seed={ckpt.get("seed")}, epoch={ckpt.get("epoch")}')
    print('Frozen v0.1 source scores/cache: PASS')
    print('CIFAR-10.1 v6 hashes: PASS')
    print('New target checkpoints: seed2027, seed31415; epoch 20')
    print(f'Feature subset: 2500 images; SHA-256={sha_indices(idx)}')
    if a.audit_only:
        print('AUDIT PASS: no inference, ablations, cache writes, or report writes.'); return
    start=time.time(); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
    print('Frozen seed42 source scores; fresh target mappings; CIFAR-10.1 v6 effects. Official CIFAR-10 test set NOT loaded.')
    base=datasets.CIFAR10(root=str(DATA_ROOT),train=True,download=False,transform=None)
    CACHE_DIR.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(a.seed)
    result={}; sig_info={}; eff_info={}
    for key,cp in TARGETS.items():
        ts,si=signatures_for_target(key,cp,idx,base,device)
        tp,ei=target_effects(key,cp,images,labels,data_hashes,device); sig_info[key]=si; eff_info[key]=ei
        layers={}; sames=[]; aligns=[]; randoms=[]
        for layer,width in WIDTHS.items():
            mapping,astats=v06.align_channels(src_sig[layer],ts[layer])
            if len(mapping)!=width or not np.array_equal(np.sort(mapping),np.arange(width)): raise ValueError(f'Bad mapping {key}/{layer}')
            ss,sa=v01.transfer_score_arrays(src_payload['effects'],layer,width)
            tsigned,ta=v02.effect_arrays(tp['effects'],layer,width)
            aligned_s=np.empty(width); aligned_a=np.empty(width); aligned_s[mapping]=ss; aligned_a[mapping]=sa
            sm=v01.score_transfer(ta,sa); am=v01.score_transfer(ta,aligned_a)
            rm=random_control(ta,sa,am,sm,a.random_mappings,rng)
            layers[layer]={'channels':width,'alignment':{**safe(astats),'fraction_mapped_to_same_index':float(np.mean(mapping==np.arange(width))),
                'mapping_source_index_to_target_index':mapping.astype(int).tolist(),'mapping_sha256':hashlib.sha256(mapping.astype(np.int64).tobytes()).hexdigest()},
                'target_evaluation_baseline':tp['baseline'],'same_index_transfer':sm,'activation_aligned_transfer':am,'random_mapping_control':rm,
                'signed_effect_transfer_secondary':{'same_index':v01.signed_effect_metrics(tsigned,ss),'activation_aligned':v01.signed_effect_metrics(tsigned,aligned_s)}}
            sp=sm['topk']['top_10pct']['precision']; ap=am['topk']['top_10pct']['precision']; rp=rm['precision_mean']
            sames.append(sp); aligns.append(ap); randoms.append(rp)
            print(f'{key}/{layer}: same={sp:.3f}, aligned={ap:.3f}, random mean={rp:.3f}, delta={ap-sp:+.3f}')
        delta=float(np.mean(np.asarray(aligns)-np.asarray(sames)))
        result[key]={'source_model':'seed42','checkpoint':str(cp.relative_to(ROOT)),'checkpoint_sha256':sha(cp),'layers':layers,
            'mean_primary_precision_across_layers':{'same_index':float(np.mean(sames)),'activation_aligned':float(np.mean(aligns)),
            'random_mapping_mean':float(np.mean(randoms)),'aligned_minus_same_index':delta}}
    rows=[(t,l,r) for t,v in result.items() for l,r in v['layers'].items()]
    aligned=np.array([r['activation_aligned_transfer']['topk']['top_10pct']['precision'] for _,_,r in rows])
    same=np.array([r['same_index_transfer']['topk']['top_10pct']['precision'] for _,_,r in rows])
    rnd=np.array([r['random_mapping_control']['precision_mean'] for _,_,r in rows])
    deltas={t:result[t]['mean_primary_precision_across_layers']['aligned_minus_same_index'] for t in TARGETS}
    passed=bool(aligned.mean()>same.mean() and aligned.mean()>rnd.mean() and all(x>0 for x in deltas.values()))
    report={'experiment':'Brahma MetaCore MetaTransfer v0.3 Independent Target-Seed Replication','status':'completed_by_script_when_run',
        'purpose':'Test activation-based importance transfer on two new target initializations using external CIFAR-10.1 v6 evaluation.',
        'source_model':'seed42','target_models':list(TARGETS),'layers':list(v06.LAYERS),'seed_for_random_controls':a.seed,'random_mappings_per_target_layer':a.random_mappings,
        'dataset':{'name':'CIFAR-10.1 v6','source_url':v02.DATASET_SOURCE,'examples':2000,**data_hashes,'official_cifar10_test_set_loaded':False},
        'frozen_inputs':{'v01_report_path':str(V01_REPORT.relative_to(ROOT)),'v01_report_sha256':sha(V01_REPORT),
            'source_effect_cache_path':str(src_path.relative_to(ROOT)),'source_effect_cache_sha256':src_hash,'source_importance_scores_reused_without_refitting':True,
            'v02_external_dataset_hashes_verified':True,'old_target_specific_alignment_mappings_reused':False},
        'protocol':{'feature_examples':2500,'feature_indices_sha256':sha_indices(idx),'alignment_method':'Hungarian assignment maximizing absolute Pearson correlation between standardized per-image spatially pooled activation signatures.',
            'target_effect_evaluation_examples':2000,'target_importance_measured_fresh_on_external_images':True,'new_target_seeds':True,
            'new_target_seeds_independent_initializations':True,'same_architecture_training_data_and_frozen_split':True,'official_cifar10_test_set_loaded':False},
        'target_signature_caches':sig_info,'target_eval_caches':eff_info,'results':result,
        'aggregate_primary_metric':{'number_of_target_layer_comparisons':len(rows),'mean_aligned_top10_precision':float(aligned.mean()),
            'mean_same_index_top10_precision':float(same.mean()),'mean_random_mapping_top10_precision':float(rnd.mean()),
            'aligned_beats_same_index_comparisons':int(np.sum(aligned>same)),'ties_vs_same_index_comparisons':int(np.sum(aligned==same)),
            'aligned_beats_random_mapping_mean_comparisons':int(np.sum(aligned>rnd)),'target_seed_mean_aligned_minus_same_index_precision':deltas,
            'number_of_new_target_seeds_with_positive_mean_delta':int(sum(x>0 for x in deltas.values()))},
        'predefined_directional_screen':{'rule':'Screening support requires mean aligned Top-10% precision to exceed same-index and mean random-mapping controls, with positive layer-averaged deltas for both new target seeds.',
            'passed':passed,'interpretation':'Directional screening only; two new target seeds are insufficient for a strong population-level statistical claim.'},
        'limitations':['Only two new target initializations are evaluated.','Models share architecture, training data, and the same train/validation split.',
            'The alignment subset comes from a validation partition previously used for checkpoint selection.','Layer comparisons/channels are dependent and are not independent model replications.',
            'Random-mapping controls are permutation references, not stand-alone population-level significance tests.','Single-channel ablation does not capture channel interactions.',
            'This experiment does not establish semantic understanding, general intelligence, architecture generation, or reduced pretraining requirements.'],
        'elapsed_seconds':float(time.time()-start)}
    if out.exists() or md.exists(): raise FileExistsError(f'Refusing to overwrite reports: {out} / {md}')
    out.parent.mkdir(parents=True,exist_ok=True)
    lines=['# MetaTransfer v0.3 — Independent Target-Seed Replication','','- Frozen seed-42 source importance scores; newly computed target-specific mappings.','- External target evaluation: **2,000 CIFAR-10.1 v6 images**.','- New target seeds: **2027 and 31415**.','','| Target | Layer | Same-index P@10% | Aligned P@10% | Random-map mean | Delta |','|---|---|---:|---:|---:|---:|']
    for t,l,r in rows:
        sp=r['same_index_transfer']['topk']['top_10pct']['precision']; ap=r['activation_aligned_transfer']['topk']['top_10pct']['precision']; rp=r['random_mapping_control']['precision_mean']
        lines.append(f'| {t} | `{l}` | {sp:.3f} | {ap:.3f} | {rp:.3f} | {ap-sp:+.3f} |')
    ag=report['aggregate_primary_metric']; lines += ['', '## Aggregate','',f'- Mean aligned P@10%: **{ag["mean_aligned_top10_precision"]:.4f}**',f'- Mean same-index P@10%: **{ag["mean_same_index_top10_precision"]:.4f}**',f'- Mean random-map P@10%: **{ag["mean_random_mapping_top10_precision"]:.4f}**',f'- Aligned beats same-index: **{ag["aligned_beats_same_index_comparisons"]}/{len(rows)}** layer comparisons',f'- Directional screening passed: **{passed}**','','## Limitations','']
    lines += [f'- {x}' for x in report['limitations']]; lines.append('')
    md.write_text('\n'.join(lines),encoding='utf-8'); out.write_text(json.dumps(safe(report),indent=2,allow_nan=False),encoding='utf-8')
    print('\nJSON report saved:',out.relative_to(ROOT)); print('Markdown report saved:',md.relative_to(ROOT)); print(f'Elapsed: {report["elapsed_seconds"]:.1f}s')

if __name__=='__main__': main()
