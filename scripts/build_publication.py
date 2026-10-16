"""Verify the published metric archive and rebuild all report figures without CUDA."""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NAMES={'pointnext_s':'PointNeXt-S','pointnext_b':'PointNeXt-B','pointnext_l':'PointNeXt-L',
       'pointnext_xl':'PointNeXt-XL','pointnet':'PointNet','pointnet2':'PointNet++'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'docs/assets')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    base=ROOT/'reports/key6';plan=json.loads((base/'protocol/plan.json').read_text())
    with (base/'tables/model_comparison.csv').open(encoding='utf-8-sig',newline='') as f:
        rows={r['model']:r for r in csv.DictReader(f)}
    runs=[]
    with zipfile.ZipFile(base/'metrics-and-protocol.zip') as z:
        assert z.testzip() is None
        for run in plan['runs']:
            folder=run['id'];name=NAMES[run['config']['model']]
            read=lambda rel:json.loads(z.read(folder+'/'+rel))
            tests={case['name']:read('tests/'+case['name']+'.json') for case in run['evaluations']}
            history=list(csv.DictReader(io.StringIO(z.read(folder+'/history.csv').decode('utf-8'))))
            assert len(history)==300 and [int(r['epoch']) for r in history]==list(range(300))
            for key,m in tests.items():
                cm=np.array(m['confusion_matrix']);tp=cm.diagonal();s=cm.sum(1);pr=cm.sum(0)
                f1=np.divide(2*tp,s+pr,out=np.zeros(len(s)),where=s+pr>0)
                assert cm.sum()==10107
                for k,v in [('oa',tp.sum()/cm.sum()),('macc',(tp/s).mean()),('macro_f1',f1.mean())]:
                    assert abs(m[k]-v)<1e-12,(name,key,k)
            for k in ['oa','macc','macro_f1','loss']:
                assert abs(float(rows[name][k])-tests['clean'][k])<1e-12
            runs.append(dict(name=name,tests=tests,history=history,benchmark=read('benchmark.json'),row=rows[name]))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=[r['name'] for r in runs];colors=plt.get_cmap('tab10')(np.arange(6))
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(a.output/(name+'.'+ext),dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(13,4.7),layout='constrained');x=np.arange(6)
    for i,k in enumerate(['oa','macc','macro_f1']):
        ax[0].bar(x+(i-1)*.25,[r['tests']['clean'][k]*100 for r in runs],.25,label={'oa':'OA','macc':'mAcc','macro_f1':'Macro F1'}[k])
    ax[0].set(xticks=x,xticklabels=names,ylabel='Clean test score (%)',ylim=(0,105));ax[0].tick_params(axis='x',rotation=25);ax[0].legend(loc='lower left')
    for r,col in zip(runs,colors):ax[1].scatter(r['benchmark']['batch_1']['p50_ms'],r['tests']['clean']['oa']*100,color=col,s=75,label=r['name'])
    ax[1].set(xlabel='Batch-1 median forward latency (ms)',ylabel='Clean test OA (%)');ax[1].legend(fontsize=8,loc='lower right');ax[1].grid(alpha=.2)
    save(fig,'classification_and_latency')
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for r,col in zip(runs,colors):
        for ax,key in zip(axes.flat,['train_loss','val_loss','val_oa','val_macro_f1']):
            values=[float(d[key])*(100 if 'loss' not in key else 1) for d in r['history']]
            ax.plot(range(1,301),values,color=col,lw=1,alpha=.85,label=r['name'])
    for ax,title in zip(axes.flat,['Training objective','Validation objective','Validation OA (%)','Validation macro F1 (%)']):
        ax.set(xlabel='Epoch',ylabel=title);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,ncol=2);save(fig,'training_curves')
    cases=['clean','jitter_1','jitter_2','jitter_3','rotation_1','rotation_2','rotation_3','so3',
           'outliers_1','outliers_2','outliers_3','occlusion_1','occlusion_2','occlusion_3','dropout_3','scale_3','translation_3']
    values=np.array([[r['tests'][k]['oa']*100 for k in cases] for r in runs])
    fig,ax=plt.subplots(figsize=(15,4.6),layout='constrained');im=ax.imshow(values,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    for i in range(6):
        for j in range(len(cases)):ax.text(j,i,f'{values[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if values[i,j]>60 else 'black')
    ax.set(xticks=np.arange(len(cases)),xticklabels=cases,yticks=np.arange(6),yticklabels=names);ax.tick_params(axis='x',rotation=50)
    fig.colorbar(im,ax=ax,label='Test OA (%)',shrink=.8);save(fig,'robustness_oa')
    classes=[c['name'] for c in runs[0]['tests']['clean']['per_class']]
    values=np.array([[c['f1']*100 for c in r['tests']['clean']['per_class']] for r in runs])
    fig,ax=plt.subplots(figsize=(15,4.8),layout='constrained');im=ax.imshow(values,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
    ax.set(xticks=np.arange(32),xticklabels=classes,yticks=np.arange(6),yticklabels=names);ax.tick_params(axis='x',rotation=70,labelsize=8)
    fig.colorbar(im,ax=ax,label='Per-class F1 (%)',shrink=.8);save(fig,'per_class_f1')
    variants=plan['runs'][0]['evaluations'][:9]
    values=np.array([[next(m['oa']*100 for m in r['tests'].values() if m['case']['dataset']==v['dataset'] and m['case']['corruption']=='clean') for v in variants] for r in runs])
    fig,ax=plt.subplots(figsize=(12,4.4),layout='constrained');im=ax.imshow(values,vmin=94,vmax=98,cmap='YlGnBu',aspect='auto')
    for i in range(6):
        for j in range(9):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if values[i,j]>96.5 else 'black')
    ax.set(xticks=np.arange(9),xticklabels=[v['dataset'] for v in variants],yticks=np.arange(6),yticklabels=names)
    ax.tick_params(axis='x',rotation=40);fig.colorbar(im,ax=ax,label='Test OA (%)',shrink=.8);save(fig,'cross_representation_oa')
    c=runs[0]['tests']['clean'];support=np.array([d['support'] for d in c['per_class']]);order=np.argsort(support)
    fig,axes=plt.subplots(1,2,figsize=(14,8),layout='constrained')
    axes[0].barh(np.arange(32),support[order],color='#406b9b');axes[0].set(xscale='log',xlabel='Held-out objects (log scale)',yticks=np.arange(32),yticklabels=np.array(classes)[order])
    for r,col in zip(runs,colors):
        recalls=np.array([d['recall']*100 for d in r['tests']['clean']['per_class']]);axes[1].scatter(support,recalls,s=25,color=col,alpha=.65,label=r['name'])
    axes[1].set(xscale='log',xlabel='Held-out objects per class (log scale)',ylabel='Class recall (%)',ylim=(-3,103));axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    save(fig,'class_support_and_recall')
    cm=np.array(c['confusion_matrix']);errors=[]
    for i,j in np.ndindex(cm.shape):
        if i!=j and cm[i,j]:errors.append(dict(true_class=classes[i],predicted_class=classes[j],count=int(cm[i,j]),true_support=int(support[i]),recall_loss_pp=100*cm[i,j]/support[i]))
    errors.sort(key=lambda x:x['count'],reverse=True)
    with (base/'tables/s_confusions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(errors[0]));w.writeheader();w.writerows(errors)
    print('Verified 6 models, 1800 epoch rows and 192 confusion-matrix metric records; rebuilt six PNG/SVG figures.')


if __name__=='__main__':main()
