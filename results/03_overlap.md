# Why the ensemble is a percentile and not a fixed threshold

The study plan said to call a cell active when its activity passes a threshold.
That rule does not work on this dataset, and the numbers show why. Minian only
keeps cells it could detect in the first place, so within a few minutes almost
every kept cell fires at least a few times:

| session                   |   cells | at least 0.05 events per second   | at least 0.1   |
|:--------------------------|--------:|:----------------------------------|:---------------|
| Ca-EEG2-1 NeutralExposure |     623 | 84%                               | 65%            |
| Ca-EEG2-1 FC              |     597 | 95%                               | 87%            |
| Ca-EEG2-1 Recall1         |     336 | 89%                               | 60%            |
| Ca-EEG3-4 NeutralExposure |     851 | 97%                               | 91%            |
| Ca-EEG3-4 FC              |     783 | 98%                               | 94%            |
| Ca-EEG3-4 Recall1         |     758 | 100%                              | 98%            |
| Ca-EEG3-4 Recall2         |     831 | 100%                              | 98%            |
| Ca-EEG3-4 Recall3         |     809 | 100%                              | 98%            |

Written by scripts/03_overlap.py, using the session windows in config.yaml.
If a rule calls this many cells active, every pair of sessions overlaps almost
completely and the measurement carries no information.

So an ensemble here is the most active 25 percent of the cells in that
session. Chance overlap is then well defined, and the analysis is repeated at
[10, 25, 50] percent so the reader can see whether the cut-off drives the result.
