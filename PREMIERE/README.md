# Predictive Process Mining Meets Computer Vision

**The repository contains code referred to the work:**

*Vincenzo Pasquadibisceglie, Annalisa Appice, Giovanna Castellano, Donato Malerba*

[*Predictive Process Mining Meets Computer Vision*](https://link.springer.com/chapter/10.1007/978-3-030-58638-6_11)

Please cite our work if you find it useful for your research and work.

```
@InProceedings{10.1007/978-3-030-58638-6_11,
author="Pasquadibisceglie, Vincenzo
and Appice, Annalisa
and Castellano, Giovanna
and Malerba, Donato",
editor="Fahland, Dirk
and Ghidini, Chiara
and Becker, Jörg
and Dumas, Marlon",
title="Predictive Process Mining Meets Computer Vision",
booktitle="Business Process Management Forum",
year="2020",
publisher="Springer International Publishing",
address="Cham",
pages="176--192",
isbn="978-3-030-58638-6"
}
```
# How to use
Generate Feature set:
- event_log: event log name

```
python generate_feature.py -event_log receipt
```

Generate Image set:
- event_log: event log name

```
python generate_image.py -event_log receipt
```

Train neural network:
- event_log: event log name

```
python deep_ppm.py -event_log receipt
```
