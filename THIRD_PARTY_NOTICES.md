# Attribution and scope of licensing

The root MIT license covers this repository's original experiment harness,
conversion tools, documentation and analysis code. It does **not** grant new
rights over the source CAD models or derived point clouds.

## PointNeXt and OpenPoints

- PointNeXt: <https://github.com/guochengqian/PointNeXt>, MIT, copyright Guocheng Qian.
- Pinned commit: `b7ec10fd8b9acdb1c3b18e557f0d3465eb919164`.
- OpenPoints: <https://github.com/guochengqian/openpoints>, its own upstream notices apply.
- Pinned commit: `db31e0d94ede35bc672ef06d1b6e4073a9c269d5`.

The complete upstream projects are retained as nested Git submodules, with their
licenses and history. The build adapter copies CUDA/C++ sources to a build
directory and applies modern PyTorch tensor API compatibility changes there;
it does not replace the upstream CUDA sampling/grouping kernels.

## Mechanical Components Benchmark

Source project: <https://github.com/stnoah1/mcb>.

Kim, Sangpil; Chi, Hyung-gun; Hu, Xiao; Huang, Qixing; Ramani, Karthik.
*A Large-scale Annotated Mechanical Components Benchmark for Classification
and Retrieval Tasks with Deep Neural Networks*. ECCV 2020.

The MCB collection tooling has an MIT license (copyright 2019 HG). That software
license is not, by itself, evidence that every CAD asset has an MIT license.
MCB collects models from third-party CAD repositories; original asset rights,
attribution obligations and source distribution terms continue to apply.
The reorganized 32-class dataset is a derivative research preparation, not an
official MCB release or a grant of unrestricted commercial rights. No new
dataset-wide license is asserted by this repository.

## Scientific interpretation

Results in this repository come from a single-seed, common-recipe experiment.
PointNeXt-B/L/XL adapt upstream S3DIS encoders for object classification; they
are not official pretrained classification checkpoints. Cite both the original
methods and this experiment when reusing its protocol or results.
