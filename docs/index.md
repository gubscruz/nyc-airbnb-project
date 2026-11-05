# NYC Airbnb - how much should a listing charge per night?

Semester project for the Machine Learning course, by **Gustavo Cruz, Henrique Mayor and Pedro Fardin**. We use the [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) (48,895 listings, 2019 snapshot) and the task is **regression on `price`**.

The project has a few deliverables and each one gets its own page:

| | deliverable | status |
|---|---|---|
| 1 | [Exploratory Data Analysis](projects/eda/index.md) - data dictionary, quality, distributions, correlations and the preprocessing pipeline | done |
| 2 | Regression model | soon |

## Running the code

```bash
git clone https://github.com/gubscruz/nyc-airbnb-project.git
cd nyc-airbnb-project
pip install -r requirements.txt
python code/eda.py          # regenerates every figure and docs/projects/eda/results.json
python code/preprocess.py   # prints the pipeline sanity report
```

The dataset is committed under `data/` so a clean clone runs without a Kaggle account.
