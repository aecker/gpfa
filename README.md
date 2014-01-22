GPFA - Gaussian Process Factor Analysis
=======================================

This is an implementation of Gaussian Process Factor Analysis,
an analysis tool for simultaneous dimensionality reduction and
smoothing.  The model was developed by Byron Yu and John
Cunningham.  My implementation builds heavily on their original
code, which can be found here: 

http://users.ece.cmu.edu/~byronyu/software.shtml

Details about the model can be found in the following paper: 

Yu, B.M. et al., 2009. Gaussian-Process Factor Analysis for
Low-Dimensional Single-Trial Analysis of Neural Population
Activity. Journal of Neurophysiology, 102(1), pp.614-635.

I used this model in the following paper:

A.S. Ecker, P. Berens, R.J. Cotton, M. Subramaniyan, G.H. Denfield,
C.R. Cadwell, S.M. Smirnakis, M. Bethge, A.S. Tolias (2014): State
dependence of noise correlations in macaque primary visual cortex.
Neuron.

For more information go to: 
http://toliaslab.org/publications/ecker-et-al-2014/


What's the difference?
======================

I used the model in a slightly different context, whichrequired
some additions but also allowed me to simplify some aspects:

* Fixed trial length. In contrast to the original implementation, 
  my code does not allow for variable trial length (fixed T).

* Known regressors. I added the option to include a number of
  known regressors. This can be useful if you want to include an
  external signal as a predictor into the model. Examples of such
  signals could be the local field potential or a stimulus with
  known temporal dynamics.

* Estimation of residual covariance. I added the option to
  estimate the residual covariance that remains after a certain
  number of latent factors have been accounted for. This can be
  useful if you want to estimate the covariance in the presence
  of some confounder, which is modeled by the latent factor.
