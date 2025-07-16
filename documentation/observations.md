# Observations
Here I will record all the observations I'll be making while working on this project.

### Initial Observations from the dataset
After importing the dataset, I can notice that:
- The dataset is clean and has no missing value.
- The `structure type` has only 4 unique values, so one-hot encoding will result in only 2 more columns.
- The ranges of data varies a lot between `site area`, `water consumption` and other data columns, so normalization will be necessary.
---
### Observation from correlation heatmap
After plotting a correlation heatmap for all the numeric columns, few very interesting observations are made:
- The correlations values for all the data columns with electricity cost are
    | Data Column | Correlation | Remarks |
    | -----       | -------     | ----    |
    | `site area` | $0.87$        | 🟢 Strong positive correlation |
    | `water consumption` | $0.70$        | 🟢 Strong positive correlation |
    | `recycling rate` |$-0.01$        | 🔴 No linear correlation |
    | `utilisation rate` | $0.21$        | 🟡 Mild positive correlation |
    | `air quality index` | $0.02$        | 🔴 No linear correlation |
    | `issue resolution time` | $0.04$        | 🔴 No linear correlation |
    | `resident count` | $0.36$        | 🟡 Mild positive correlation |

- This observation makes it clear that only `site area`, `water consumption`, `utilisation rate` and `resident count` has significance in the linear model. The other columns has near-zero linear correlation in the data, and will likely act as noise in the regression context. Hence I'll exclude these columns from the linear regression model, but may visit them later for tree-based models.

I'll be following this with scatterplots with the identified high sigficance data columns.

---
### Observations from plotting the scatterplots
Several interesting things to notice after the plotting the scatter-plots
- After plotting the `resident count` vs `electricity cost`, there is a very high density of points at near 0 residents with very high variance of electricity cost.
- This made me suspect that several structure types have very low residents but high electricity consumption. This points to `structure type` playing a role in this case.
- After some more specific analysis, I found that `industrial` and `commercial` has very high average electricity consumption for very low residents ($resident\ count < 5$).
- This means that for `industrial` and `commercial`, `resident count` holds no correlation with the `electricity cost`. This beckons me to split the data based on various `structure type` and check for correlation matrices for each dataset.

I'll be following with new heatmaps and scatterplots to explain the data.

---
### Observations from the new correlation heatmaps
After plotting all the correlation maps of various `structure type`, it is noticable that the `resident count` has :cros

