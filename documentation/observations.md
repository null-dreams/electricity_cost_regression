# Observations
Here I will record all the observations I'll be making while working on this project.

---

### Initial Observations from the dataset
After importing the dataset, I can notice that:
- The dataset is clean and has no missing value.
- The `structure type` has only 4 unique values, so one-hot encoding will result in only 2 more columns.
- The ranges of data varies a lot between `site area`, `water consumption` and other data columns, so normalization will be necessary.
---

### Observation from correlation heatmap
After plotting a correlation heatmap for all the numeric columns, few very interesting observations are made:
- The correlations values for all the data columns with electricity cost are
    - `site area` - $ 0.87 $ - High positive correlation
    - `water consumption` - $ 0.70 $ - High positive correlation
    - `recycling rate` - $ -0.01 $ - No linear correlation
    - `utilisation rate` - $ 0.21 $ - Significant positive correlation
    - `air quality index` - $ 0.02 $ - No linear correlation
    - `issue resolution time` - $ 0.04 $ - No linear correlation
    - `resident count` - $ 0.36 $ - Significant positive correlation
- This observation makes it clear that only `site area`, `water consumption`, `utilisation rate` and `resident count` has significance in the linear model. Thus, I'll not be using other columns as they would act like noise for the model.
---


