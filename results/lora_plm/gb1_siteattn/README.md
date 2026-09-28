# GB1 model

This checkpoint accompanies the GB1 candidate pool in `../gb1_beam/`.
The exported head preserves the four-head, one-layer configuration actually used
for those predictions. It retains the first attention layer from the original
two-layer checkpoint without changing parameter values. Source hashes and the
sequence-indexing convention are stored in `meta.txt`.

Run prediction through the shared loader described in the repository README.
