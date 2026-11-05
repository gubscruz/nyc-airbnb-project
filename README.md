# nyc-airbnb-project

Machine Learning course project: predicting the nightly **price** of Airbnb listings in New York City
([NYC Airbnb Open Data, Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)).

Report site: https://gubscruz.github.io/nyc-airbnb-project/

**Authors:** Gustavo Cruz, Henrique Mayor and Pedro Fardin.

## Layout

```
code/
  preprocess.py   # importable Pipeline + ColumnTransformer (fit on train only)
  eda.py          # regenerates every figure and number of the EDA report
data/
  AB_NYC_2019.csv # the dataset, committed so a clean clone just works
docs/
  index.md
  projects/eda/   # the EDA report (index.md, figures/, results.json)
mkdocs.yml
```

## Reproduce

```bash
pip install -r requirements.txt
python code/eda.py          # ~2 min, t-SNE and UMAP are the slow part
python code/preprocess.py   # prints shapes, NaN check and feature names
mkdocs serve                # local preview of the site
mkdocs gh-deploy            # builds and pushes the site to the gh-pages branch
```

Fixed seed everywhere (`random_state=42`).
