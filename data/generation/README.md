
# Reference simualted datasets

Characteristics of the reference datasets.

To run the script:
```python
python data_generation_script.py -c data_generation_params_v.0.1.0.yml
```


**Dataset 0.x.x:**

- Original dataset for the WaveDiff paper
- Spatial variations (simulated variations order d=2)
- Spectral variations (stellar SEDs from 13 templates)
- Varying noise level (uniform SNRs from 10 to 110)
- v0.1.0 without masks
- v0.2.0 with dummy (unitary) masks
- v0.3.0 with realistic masks

**Dataset 1.x.x:**

- Spatial variations (simulated variations order d=2)
- Spectral variations (stellar SEDs from 13 templates)
- Varying noise level (uniform SNRs from 10 to 110)
- Intra-pixel shifts of max 1/2 pixel per direction: requires centroid correction
- CCD-misalignments
- v1.1.0 without masks
- v1.2.0 with dummy (unitary) masks
- v1.3.0 with realistic masks

**Dataset 2.x.x:**

- Spatial variations (simulated variations order d=2)
- Spectral variations (stellar SEDs from 13 templates)
- Varying noise level (uniform SNRs from 10 to 110)
- Intra-pixel shifts of max 1/2 pixel per direction: requires centroid correction
- CCD-misalignments
- v2.x.1/2/3/4 Adding realistic spatial variations (NoSFE) and priors with errors (simulated variations order d=4)
    - Different levels of prior errors will create different datasets
- v2.1.1/2/3/4 without masks
- v2.2.1/2/3/4 with dummy (unitary) masks
- v2.3.1/2/3/4 with realistic masks

**Dataset 3.x.x:**

- Spatial variations (simulated variations order d=2)
- Spectral variations (stellar SEDs from 13 templates)
- Varying noise level (uniform SNRs from 10 to 110)
- Intra-pixel shifts of max 1/2 pixel per direction: requires centroid correction
- CCD-misalignments
- v3.x.1/2/3/4 Adding realistic spatial variations (SFE) and priors with errors (simulated variations order d=4)
    - Different levels of prior errors will create different datasets
- v3.1.1/2/3/4 without masks
- v3.2.1/2/3/4 with dummy (unitary) masks
- v3.3.1/2/3/4 with realistic masks

