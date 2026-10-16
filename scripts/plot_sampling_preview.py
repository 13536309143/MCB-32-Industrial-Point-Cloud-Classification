"""Draw nine existing representations of one held-out object; does not resample data."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/MCB_pointcloud')
    p.add_argument('--output',type=Path,default=ROOT/'docs/assets/sampling_preview.png')
    a=p.parse_args()
    # A fixed, auditable illustration, with no accuracy-driven example selection.
    example=next(iter(sorted((a.data/'area_random_2048/test/gear').glob('*.npy'))))
    rel=example.relative_to(a.data/'area_random_2048')
    fig=plt.figure(figsize=(11,9),facecolor='white',layout='constrained')
    for row,(method,label) in enumerate([('area_random','Area random'),('fps','Farthest point'),('poisson_disk','Poisson disk')]):
        for col,n in enumerate([2048,4096,8192]):
            x=np.load(a.data/f'{method}_{n}'/rel,allow_pickle=False)
            assert x.shape==(n,3)
            ax=fig.add_subplot(3,3,row*3+col+1,projection='3d')
            ax.scatter(x[:,0],x[:,1],x[:,2],c=x[:,2],cmap='viridis',vmin=-1,vmax=1,
                       s=1.6 if n==2048 else .9,alpha=.85,linewidths=0,rasterized=True,depthshade=False)
            ax.set(xlim=(-1.05,1.05),ylim=(-1.05,1.05),zlim=(-1.05,1.05))
            ax.set_box_aspect((1,1,1),zoom=1.35);ax.view_init(elev=26,azim=38);ax.set_axis_off()
            ax.set_title(f'{label} | {n:,} points',fontsize=12,pad=0,color='#17334d')
    fig.suptitle('One industrial object, nine paired point clouds',fontsize=18,color='#17334d')
    fig.supxlabel(f'Class: gear | Held-out object: {example.stem[:16]}... | Same coordinates and view; color = Z',fontsize=10)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=180,bbox_inches='tight');plt.close(fig)
    print('Saved',a.output,'source',rel)


if __name__=='__main__':main()
