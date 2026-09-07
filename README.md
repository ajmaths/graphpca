# GraphPCA

This repository contains the implementation of GraphPCA and related experiment notebooks from the paper "Covariance and Principal Component Analysis on Riemannian Manifolds and Graphs" by Luiz Hartmann, Wenwen Li, and Washington Mio. 

The paper is available on arXiv at: https://arxiv.org/abs/xxxx.xxxxx

Given a graph whose edge weights represent distances between two adjacent vertices, together with a probability distribution on its vertex set. GraphPCA builds heat/diffusion potentials on the vertices, lifts them to edge flows through a length-scaled incidence operator, and then performs PCA in the edge space under the inner product $<u,v>_C:=u^\top C v$ where $C$ is the weight matrix of the graph. The result, for each diffusion time t, is a set of $C$-orthonormal edge-space principal components, along with centered, per-vertex projection coefficients that can serve as multiscale coordinates for visualization, clustering, or downstream learning.
