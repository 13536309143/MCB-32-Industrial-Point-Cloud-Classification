"""A password-free local handoff using the existing SSH control connection."""
import argparse
import datetime
import json
import shlex
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--expected', required=True)
    args = parser.parse_args()
    expected = json.loads(Path(args.expected).read_text())
    print('Waiting for existing resumable rsync upload...', flush=True)
    while True:
        result = subprocess.run(['ps', '-p', str(args.pid), '-o', 'command='], text=True, capture_output=True)
        if result.returncode or 'rsync' not in result.stdout or 'MCB_pointcloud' not in result.stdout:
            break
        time.sleep(10)
    remote = f"""
import json,os
from pathlib import Path
root=Path('/root/point/data/MCB_pointcloud')
paths=list(root.rglob('*.npy'))
total=sum(p.stat().st_size for p in paths)
assert len(paths)=={expected['count']}, ('Incomplete upload',len(paths))
assert total=={expected['bytes']}, ('Incomplete bytes',total)
value={{'files':len(paths),'bytes':total,'rsync_finished':True,'full_sha_check':'performed by the suite before the remaining queue'}}
out=Path('/root/point/remote_logs/upload_complete.json')
tmp=out.with_suffix('.tmp')
tmp.write_text(json.dumps(value,indent=2)+'\\n')
os.replace(tmp,out)
print(json.dumps(value))
"""
    command = ['ssh', '-S', args.control, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               '-p', str(args.port), args.host, '/root/enter/bin/python -c ' + shlex.quote(remote)]
    subprocess.run(command, check=True)
    print('Upload handoff complete. The server queue continues independently of this Mac.', flush=True)


if __name__ == '__main__':
    main()
