# TMLR submission draft

This directory contains the current TMLR-format draft for:

**Deliberately Diverse Aggregation for Distributed Optimization**

Anonymous artifact:
https://anonymous.4open.science/r/local_gp-9F83/

Build from this directory with:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The draft is intentionally conservative: it includes the fixed-compute wins,
the adaptive HPO breadth results, and the boundary/null cases that keep the
claim honest.
