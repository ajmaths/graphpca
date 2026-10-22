# GraphPCA

This repository contains the implementation of GraphPCA and related experiment notebooks from the paper "Covariance and Principal Component Analysis on Riemannian Manifolds and Graphs" by Luiz Hartmann, Wenwen Li, and Washington Mio. 

The paper is available on arXiv at: https://arxiv.org/abs/2609.15753

Given a graph whose edge weights represent distances between two adjacent vertices, together with a probability distribution on its vertex set. GraphPCA builds heat/diffusion potentials on the vertices, lifts them to edge flows through a length-scaled incidence operator, and then performs PCA in the edge space with respect to the inner product $<u,v>_C:=u^\top C v$ where $C$ is the weight matrix of the graph. The result, for each diffusion time t, is a set of $C$-orthonormal edge-space principal components, along with centered, per-vertex projection coefficients that can serve as multiscale coordinates for visualization, clustering, or downstream learning.

Both `GraphPCA_Cycle_UniformMu.ipynb` and `ModelNet10_(chair).ipynb` use the shared implementation in `graphpca_core_annotated.py`. Download the notebook and this module from the same repository version. The Python dependencies are listed in `requirements.txt`.

To run either notebook in Google Colab, open the notebook and upload `graphpca_core_annotated.py`. Place the module directly in `/content`, keeping its original filename, before running the notebook cells from top to bottom. Ensure that the required Python packages are installed. Uploaded files are temporary and must be uploaded again if the runtime is deleted or recreated.

For local Jupyter execution, download or clone the repository, install the dependencies with `pip install -r requirements.txt`, and start the notebook from the repository directory. The ModelNet notebook requires internet access for its initial dataset download, which is approximately 450 MB.
