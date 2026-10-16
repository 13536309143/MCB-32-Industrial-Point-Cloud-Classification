#!/usr/bin/env python3
"""Portable MCB_POINTCLOUD_V1 converter, used by convert_mcb_9sets.bat.

Python 3.12 + NumPy 2.5.4; included Windows x64 native sampling DLL.
Source meshes are read-only. Output replacement is atomic and checkpointed.
"""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse,contextlib,ctypes,hashlib,io,json,logging,multiprocessing,signal,sqlite3,sys,time,shutil,concurrent.futures
import numpy as np

SIZES=(2048,4096,8192)
METHODS=('area_random','fps','poisson_disk')
POOL_SIZE=40960
BASE_SEED=42
SOURCE=OUTPUT=None
OVERWRITE=False
lib=None
PROTOCOL='MCB_POINTCLOUD_V1'

def sha256_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def initialize(source,output,kernel,seed,overwrite):
 global SOURCE,OUTPUT,lib,BASE_SEED,OVERWRITE
 SOURCE=Path(source);OUTPUT=Path(output);BASE_SEED=seed;OVERWRITE=overwrite
 lib=ctypes.CDLL(str(kernel))
 dp=np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS')
 ip=np.ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='C_CONTIGUOUS')
 lib.exact_fps.argtypes=[dp,ctypes.c_int,ctypes.c_int,ip]
 lib.poisson_elimination.argtypes=[dp,ctypes.c_int,ctypes.c_int,ctypes.c_double,ip]
 lib.exact_fps.restype=lib.poisson_elimination.restype=ctypes.c_int

def initialize_worker(*args):
 initialize(*args)
 signal.signal(signal.SIGINT,signal.SIG_IGN)

def fps(p,k):
 p=np.ascontiguousarray(p,dtype=np.float64);out=np.empty(k,np.int32)
 if lib.exact_fps(p,len(p),k,out):raise RuntimeError('Native FPS failed')
 return out

def poisson(p,k,area):
 p=np.ascontiguousarray(p,dtype=np.float64);out=np.empty(k,np.int32)
 if lib.poisson_elimination(p,len(p),k,area,out):raise RuntimeError('Native Poisson elimination failed')
 return out

def stable_seed(relative,stream):
 return int.from_bytes(hashlib.sha256(f'MCB_POINTCLOUD_V1|{BASE_SEED}|{relative}|{stream}'.encode('utf-8')).digest()[:8],'big')

def read_mesh(path):
 blob=path.read_bytes();mark=blob.index(b'\nf ')
 vertices=np.fromstring(blob[:mark+1].replace(b'v ',b'').decode('ascii'),sep=' ',dtype=np.float64).reshape(-1,3)
 faces=np.fromstring(blob[mark+1:].replace(b'f ',b'').decode('ascii'),sep=' ',dtype=np.int64).reshape(-1,3)-1
 if not np.isfinite(vertices).all() or not len(faces) or faces.min()<0 or faces.max()>=len(vertices):raise ValueError(f'Invalid mesh {path}')
 tri=vertices[faces]
 area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
 if not np.isfinite(area).all() or area.sum()<=0:raise ValueError(f'Invalid surface area {path}')
 return vertices,tri,np.cumsum(area),hashlib.sha256(blob).hexdigest()

def sample_pool(tri,cdf,rng):
 def draw(n):
  picked=np.searchsorted(cdf,rng.random(n)*cdf[-1],side='right');picked=np.minimum(picked,len(tri)-1)
  u=np.sqrt(rng.random((n,1)));v=rng.random((n,1))
  a,b,c=(tri[picked,k] for k in range(3))
  return (1-u)*a+u*(1-v)*b+u*v*c
 p=draw(POOL_SIZE)
 for attempt in range(20):
  _,first=np.unique(p.astype(np.float32),axis=0,return_index=True)
  first.sort();p=p[first]
  if len(p)>=POOL_SIZE:return np.ascontiguousarray(p[:POOL_SIZE])
  p=np.concatenate((p,draw(max(1024,POOL_SIZE-len(p)))))
 raise ValueError('Could not obtain enough unique float32 surface points')

def pointclouds(job):
 relative_string,repair=job
 relative=Path(relative_string)
 vertices,tri,cdf,source_hash=read_mesh(SOURCE/relative)
 seed=stable_seed(relative.as_posix(),'surface_candidates')
 pool=sample_pool(tri,cdf,np.random.default_rng(seed))
 # Both spacing algorithms use precisely the same area-distributed candidate pool.
 f=pool[fps(pool,8192)]
 p8=pool[poisson(pool,8192,float(cdf[-1]))]
 p4=p8[poisson(p8,4096,float(cdf[-1]))]
 p2=p4[poisson(p4,2048,float(cdf[-1]))]
 arrays={'area_random':{n:pool[:n] for n in SIZES},'fps':{n:f[:n] for n in SIZES},'poisson_disk':{2048:p2,4096:p4,8192:p8}}
 lo=vertices.min(0);hi=vertices.max(0)
 outputs=[]
 prepared=[]
 for method in METHODS:
  for n in SIZES:
   rng=np.random.default_rng(stable_seed(relative.as_posix(),f'order:{method}:{n}'))
   cloud=np.ascontiguousarray(arrays[method][n][rng.permutation(n)],dtype=np.float32)
   if cloud.shape!=(n,3) or cloud.dtype!=np.float32 or not np.isfinite(cloud).all():raise ValueError('Bad point cloud')
   if len(np.unique(cloud,axis=0))!=n:raise ValueError('Duplicate float32 points')
   if np.max(cloud.astype(np.float64)-hi)>2e-7 or np.max(lo-cloud.astype(np.float64))>2e-7:raise ValueError('Points outside source bounding box')
   if np.linalg.norm(cloud.astype(np.float64),axis=1).max()>1+2e-7:raise ValueError('Points outside unit sphere')
   buffer=io.BytesIO();np.save(buffer,cloud,allow_pickle=False);payload=buffer.getvalue()
   out_relative=Path(f'{method}_{n}')/relative.with_suffix('.npy')
   target=OUTPUT/out_relative
   digest=hashlib.sha256(payload).hexdigest()
   outputs.append((out_relative.as_posix(),digest,len(payload)))
   prepared.append((target,payload,digest))
 # Check the whole model before writing any of its nine arrays.
 for target,payload,digest in prepared:
  if target.exists() and sha256_file(target)!=digest and not (repair or OVERWRITE):
   raise ValueError(f'Untracked existing output differs: {target}. Use an empty --output or --overwrite.')
 for target,payload,digest in prepared:
  if target.exists() and sha256_file(target)==digest:
   continue
  temporary=target.with_name(target.name+'.partial')
  with temporary.open('wb') as stream:stream.write(payload)
  temporary.replace(target)
  if sha256_file(target)!=digest:raise IOError(f'Output checksum mismatch: {target}')
 return relative.as_posix(),source_hash,outputs


def expected_paths(relative):
 p=Path(relative).with_suffix('.npy')
 return {f'{method}_{n}/{p.as_posix()}' for method in METHODS for n in SIZES}

def record_valid(relative,source_hash,outputs_json):
 try:
  if sha256_file(SOURCE/relative)!=source_hash:return False
  outputs=json.loads(outputs_json)
  if len(outputs)!=9 or {p for p,digest,size in outputs}!=expected_paths(relative):return False
  for name,digest,size in outputs:
   path=OUTPUT/name
   if not path.is_file() or path.stat().st_size!=size or sha256_file(path)!=digest:return False
   cloud=np.load(path,allow_pickle=False)
   n=int(name.split('/')[0].rsplit('_',1)[1])
   if cloud.shape!=(n,3) or cloud.dtype!=np.float32 or not np.isfinite(cloud).all():return False
  return True
 except (OSError,ValueError,TypeError,KeyError):return False

def available_memory():
 if os.name=='nt':
  class MemoryStatus(ctypes.Structure):
   _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(name,ctypes.c_uint64) for name in ('total_phys','available_phys','total_page','available_page','total_virtual','available_virtual','extended_virtual')]
  data=MemoryStatus();data.length=ctypes.sizeof(data)
  if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(data)):return data.available_phys
 try:return os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
 except (ValueError,OSError,AttributeError):return None

def worker_count(requested):
 if requested>0:return requested
 count=min(8,max(1,(os.cpu_count() or 2)-1));free=available_memory()
 if free is not None:count=min(count,max(1,int(free//(1.5*1024**3))))
 return max(1,count)

@contextlib.contextmanager
def output_lock(cache,output):
 key=hashlib.sha256(os.path.normcase(str(output)).encode('utf-8')).hexdigest()
 path=cache/'locks'/f'{key}.lock';path.parent.mkdir(parents=True,exist_ok=True)
 stream=path.open('a+b');stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
 try:
  try:
   if os.name=='nt':
    import msvcrt
    msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
   else:
    import fcntl
    fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except OSError as error:raise RuntimeError('Another converter is already using this output directory.') from error
  yield
 finally:stream.close()

def main():
 base=Path(__file__).resolve().parent.parent
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--source',type=Path,default=base/'MCB_merged')
 parser.add_argument('--output',type=Path,default=base/'MCB_pointcloud')
 parser.add_argument('--state',type=Path,help='Optional local checkpoint/cache directory')
 parser.add_argument('--workers',type=int,default=0,help='0: auto based on CPU/RAM')
 parser.add_argument('--seed',type=int,default=42)
 parser.add_argument('--library',type=Path,default=Path(__file__).with_name('sampling_windows_x64.dll'))
 parser.add_argument('--limit',type=int,help='Optional number of models for a quick sample run')
 parser.add_argument('--overwrite',action='store_true',help='Replace differing untracked outputs')
 parser.add_argument('--verify-only',action='store_true')
 args=parser.parse_args()
 if np.__version__!='2.5.4':raise RuntimeError('Use NumPy 2.5.4; the BAT prepares it automatically.')
 if args.workers<0 or (args.limit is not None and args.limit<1):raise ValueError('workers must be >=0 and limit must be >=1')
 source=args.source.resolve();output=args.output.resolve();kernel=args.library.resolve()
 if output==source or output.is_relative_to(source):raise ValueError('Output must be outside the source mesh directory.')
 if not kernel.is_file():raise FileNotFoundError(f'Sampling library not found: {kernel}')
 initialize(str(source),str(output),str(kernel),args.seed,args.overwrite)
 for split in ('train','test'):
  if not (source/split).is_dir():raise FileNotFoundError(f'Missing source split: {source/split}')
 labels=sorted({p.parent.name for split in ('train','test') for p in (source/split).glob('*/*.obj')})
 files=sorted(p.relative_to(source).as_posix() for split in ('train','test') for p in (source/split).glob('*/*.obj'))
 if not files:raise ValueError('No source OBJ files found.')
 if args.limit is not None:files=files[:args.limit]
 if len(files)!=len(set(files)):raise ValueError('Repeated source paths')
 source_stems={}
 for relative in files:
  parts=Path(relative).parts;source_stems.setdefault((parts[0],parts[1]),set()).add(Path(relative).stem)
 cache=(args.state or Path(os.environ.get('LOCALAPPDATA',str(Path.home()/'.cache')))/'MCBPointcloudConverter').resolve()
 cache.mkdir(parents=True,exist_ok=True)
 config={'protocol':PROTOCOL,'source':str(source),'output':str(output),'kernel_sha256':sha256_file(kernel),'seed':args.seed,'pool_size':POOL_SIZE,'numpy':np.__version__,'methods':METHODS,'sizes':SIZES}
 namespace=hashlib.sha256(json.dumps(config,sort_keys=True).encode('utf-8')).hexdigest()
 state=cache/'jobs'/namespace;state.mkdir(parents=True,exist_ok=True)
 logging.basicConfig(level=logging.INFO,format='%(message)s',handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(state/'conversion.log',encoding='utf-8')])
 with output_lock(cache,output):
  (state/'configuration.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
  db=sqlite3.connect(state/'checkpoint.sqlite3')
  try:
   db.execute('CREATE TABLE IF NOT EXISTS completed(source TEXT PRIMARY KEY,source_hash TEXT,outputs TEXT)')
   records={r[0]:r[1:] for r in db.execute('SELECT source,source_hash,outputs FROM completed')}
   done=set();todo=[];check_start=time.monotonic()
   for i,relative in enumerate(files,1):
    saved=records.get(relative)
    if saved is not None and record_valid(relative,*saved):done.add(relative)
    else:todo.append((relative,saved is not None))
    if i%1000==0 and time.monotonic()-check_start>3:
     logging.info('Resume check: %s/%s source models; %s complete',f'{i:,}',f'{len(files):,}',f'{len(done):,}');check_start=time.monotonic()
   if args.limit is None:
    for method in METHODS:
     for n in SIZES:
      for split in ('train','test'):
       split_dir=output/f'{method}_{n}'/split
       if not split_dir.exists():continue
       for class_dir in split_dir.iterdir():
        if not class_dir.is_dir():continue
        if class_dir.name not in labels:raise ValueError(f'Unexpected output class directory: {class_dir}')
        expected=source_stems.get((split,class_dir.name),set())
        extra={p.stem for p in class_dir.glob('*.npy')}-expected
        if extra:raise ValueError(f'Output contains models absent from the source: {class_dir}. Choose a new --output.')
   logging.info('Source: %s',source);logging.info('Output: %s',output);logging.info('Models: %s | Classes: %s | Cached and verified: %s',f'{len(files):,}',len(labels),f'{len(done):,}')
   logging.info('Checkpoint and log: %s',state)
   if args.verify_only:
    if todo:raise RuntimeError(f'{len(todo):,} models are missing, modified, or unverified. Run normal conversion to repair them.')
   elif todo:
    output.mkdir(parents=True,exist_ok=True)
    per_model=sum(((n*12+128+4095)//4096)*4096 for n in SIZES)*len(METHODS)
    required=len(todo)*per_model+1024**3
    free=shutil.disk_usage(output).free
    if free<required:raise RuntimeError(f'Insufficient disk space: approximately {required/1024**3:.2f} GiB needed, {free/1024**3:.2f} GiB available.')
    for method in METHODS:
     for n in SIZES:
      for split in ('train','test'):
       for label in labels:(output/f'{method}_{n}'/split/label).mkdir(parents=True,exist_ok=True)
    workers=worker_count(args.workers);logging.info('Workers: %s | Pending models: %s | Base seed: %s',workers,f'{len(todo):,}',args.seed)
    start=time.monotonic();last_report=start;initial_done=len(done)
    executor=concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,initargs=(str(source),str(output),str(kernel),args.seed,args.overwrite),mp_context=multiprocessing.get_context('spawn'))
    pending={};iterator=iter(todo)
    try:
     for _ in range(min(workers*2,len(todo))):
      job=next(iterator);pending[executor.submit(pointclouds,job)]=job[0]
     while pending:
      finished,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
      for future in finished:
       relative=pending.pop(future)
       result,sha,outputs=future.result()
       db.execute('INSERT OR REPLACE INTO completed VALUES (?,?,?)',(result,sha,json.dumps(outputs)));db.commit();done.add(relative)
       now=time.monotonic()
       if now-last_report>=5 or len(done)==len(files):
        rate=(len(done)-initial_done)/max(now-start,1e-9);eta=(len(files)-len(done))/max(rate,1e-9)/60
        logging.info('Converted %s/%s models | %s NPY files | %.2f models/s | ETA %.1f min',f'{len(done):,}',f'{len(files):,}',f'{len(done)*9:,}',rate,eta);last_report=now
       try:job=next(iterator)
       except StopIteration:continue
       pending[executor.submit(pointclouds,job)]=job[0]
    finally:
     for future in pending:future.cancel()
     executor.shutdown(wait=True,cancel_futures=True)
   logging.info('Final verification of all nine files for every selected model...')
   all_rows={r[0]:r[1:] for r in db.execute('SELECT source,source_hash,outputs FROM completed')}
   for i,relative in enumerate(files,1):
    if relative not in all_rows or not record_valid(relative,*all_rows[relative]):raise RuntimeError(f'Final verification failed: {relative}')
    if i%2000==0:logging.info('Verified %s/%s models',f'{i:,}',f'{len(files):,}')
   summary={'passed':True,'source_models':len(files),'npy_files':len(files)*9,'classes':labels,'versions':[f'{m}_{n}' for m in METHODS for n in SIZES],'dtype':'float32','columns':['x','y','z'],'source':str(source),'output':str(output),'seed':args.seed}
   (state/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
   logging.info('SUCCESS: %s source models -> %s verified NPY files in nine datasets.',f'{len(files):,}',f'{len(files)*9:,}')
  finally:db.close()
 return 0

if __name__=='__main__':
 multiprocessing.freeze_support()
 try:raise SystemExit(main())
 except KeyboardInterrupt:
  logging.error('Interrupted. Finished models are checkpointed. Run the BAT again to resume.');raise SystemExit(130)
 except Exception as error:
  logging.error('ERROR: %s',error,exc_info=True);raise SystemExit(1)
