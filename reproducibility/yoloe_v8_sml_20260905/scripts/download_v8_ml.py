"""Download only the two official README-linked YOLOE segmentation checkpoints."""
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request
import subprocess

root=Path('/workspace/YOLOE-RCAP')
logs=root/'baseline_logs'
meta=json.loads((logs/'v8_ml_official_download_metadata.json').read_text())
manifest={}
print('\n=== YOLOE-v8-M/L BASELINE PREPARATION ===',flush=True)
print('Start time:',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
print('v8-S: user-confirmed Fixed AP=27.90, APr=22.32, APc=27.81, APf=29.00; v8-S will not be rerun.',flush=True)
for scale in ('m','l'):
    name=f'yoloe-v8{scale}-seg.pt'
    metadata=meta['files'][name]
    url=f"https://huggingface.co/{meta['repository']}/resolve/{meta['revision']}/{name}"
    target=root/'pretrain'/name
    part=logs/(name+'.download')
    print('Official README source:',f'https://huggingface.co/jameslahm/yoloe/blob/main/{name}',flush=True)
    print('Immutable download URL:',url,flush=True)
    if not target.exists():
        assert not part.exists(), f'Existing partial download requires inspection: {part}'
        print('Downloading official checkpoint with bounded network retries only:', url, flush=True)
        subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','3',
                        '--retry-delay','10','--max-time','300','--output',str(part),url+'?download=true'],check=True)
        h=hashlib.sha256(part.read_bytes())
        size=part.stat().st_size
        assert size==metadata['size'] and h.hexdigest()==metadata['lfs']['sha256']
        assert not target.exists()
        part.rename(target)
    h=hashlib.sha256(target.read_bytes()).hexdigest()
    assert target.stat().st_size==metadata['size'] and h==metadata['lfs']['sha256']
    manifest[scale]=dict(filename=name,path=str(target),size_bytes=target.stat().st_size,sha256=h,
                         official_source=f'https://huggingface.co/jameslahm/yoloe/blob/main/{name}',
                         download_url=url,hf_revision=meta['revision'],official_metadata=metadata,
                         sha256_matches_official_lfs=True)
    print('Verified checkpoint:',json.dumps(manifest[scale],indent=2),flush=True)
(logs/'v8_ml_checkpoints.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Both official checkpoint downloads and SHA256 checks: PASS',flush=True)
