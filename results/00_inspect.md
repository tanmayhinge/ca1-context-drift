# Phase 0: what is inside DANDI 000718

Written by `scripts/00_inspect.py`. Nothing was downloaded; every read was an HTTP
byte range request against the archive.

## Files

| subject   | session             |   size_gb | path                                                            | asset_id                             |
|:----------|:--------------------|----------:|:----------------------------------------------------------------|:-------------------------------------|
| Ca-EEG2-1 | FC                  |      2.16 | sub-Ca-EEG2-1/sub-Ca-EEG2-1_ses-FC_image+ophys.nwb              | c2ca0ae4-f5d1-4fed-948f-a727f5e8b5a2 |
| Ca-EEG2-1 | NeutralExposure     |      4.66 | sub-Ca-EEG2-1/sub-Ca-EEG2-1_ses-NeutralExposure_image+ophys.nwb | 6ba4d4e4-287f-450b-82fe-06f4f1376ea2 |
| Ca-EEG2-1 | OfflineDay2Session1 |      4.67 | sub-Ca-EEG2-1/sub-Ca-EEG2-1_ses-OfflineDay2Session1_ophys.nwb   | 30c993ba-cda2-4d04-84a2-c22372afc865 |
| Ca-EEG2-1 | Recall1             |      2.31 | sub-Ca-EEG2-1/sub-Ca-EEG2-1_ses-Recall1_image+ophys.nwb         | d19355f8-e84f-4d57-a670-98faf57d13e6 |
| Ca-EEG2-1 | Week                |      3.67 | sub-Ca-EEG2-1/sub-Ca-EEG2-1_ses-Week.nwb                        | 0ce578db-2786-47d7-bf9a-fee3abf2ebd9 |
| Ca-EEG3-4 | FC                  |      2.13 | sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-FC_image+ophys.nwb              | 5e50a3b7-a11a-424a-b50c-e3eb5f4884be |
| Ca-EEG3-4 | NeutralExposure     |      4.72 | sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-NeutralExposure_image+ophys.nwb | 0e9666e8-8269-4da3-ab80-d215f91b5a5d |
| Ca-EEG3-4 | OfflineDay1Session1 |      4.63 | sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-OfflineDay1Session1_ophys.nwb   | 40c9201e-46f8-404d-a6a1-c951b6baa510 |
| Ca-EEG3-4 | OfflineDay2Session1 |      4.62 | sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-OfflineDay2Session1_ophys.nwb   | 07a024b1-13f1-48aa-9248-c2cd110e846b |
| Ca-EEG3-4 | Recall1             |      2.3  | sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-Recall1_image+ophys.nwb         | 13aac459-9bf6-4f9e-a02b-aa87581cce4d |
| Ca-EEG3-4 | Recall2             |      2.32 | sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-Recall2_image+ophys.nwb         | e791b7e3-cf61-48cf-8d8e-84cd2ee9f925 |
| Ca-EEG3-4 | Recall3             |      2.34 | sub-Ca-EEG3-4/sub-Ca-EEG3-4_ses-Recall3_image+ophys.nwb         | d00cc5c1-9e81-4bb9-ba06-8871af0e70da |

## What each session contains

| subject   | session             |   size_gb | start_time                       |   cells |   frames |   rate_hz |   minutes | denoised   | masks   | centroids   | max_projection   |   freezing_bouts | motion   | shock_times_s         |   shock_amplitude_mA | sleep_states   |
|:----------|:--------------------|----------:|:---------------------------------|--------:|---------:|----------:|----------:|:-----------|:--------|:------------|:-----------------|-----------------:|:---------|:----------------------|---------------------:|:---------------|
| Ca-EEG2-1 | FC                  |      2.16 | 2021-10-14T10:11:24.777000-04:00 |     597 |     4076 |     14.93 |       4.6 | True       | True    | True        | True             |               22 | True     | [120.0, 180.0, 240.0] |                 1.5  | False          |
| Ca-EEG2-1 | NeutralExposure     |      4.66 | 2021-10-12T10:44:18.744000-04:00 |     623 |     8880 |     14.93 |      10   | True       | True    | True        | True             |               43 | True     |                       |               nan    | False          |
| Ca-EEG2-1 | OfflineDay2Session1 |      4.67 | 2021-10-14T09:24:27.677000-04:00 |     625 |     8929 |     14.93 |      10   | True       | True    | True        | True             |              nan | False    |                       |               nan    | True           |
| Ca-EEG2-1 | Recall1             |      2.31 | 2021-10-15T10:33:19.999000-04:00 |     336 |     4463 |     14.93 |       5   | True       | True    | True        | True             |               58 | True     |                       |               nan    | False          |
| Ca-EEG3-4 | FC                  |      2.13 | 2022-09-19T09:18:41.001000-04:00 |     783 |     4076 |     14.93 |       4.6 | True       | True    | True        | True             |               21 | True     | [120.0, 180.0, 240.0] |                 0.25 | False          |
| Ca-EEG3-4 | NeutralExposure     |      4.72 | 2022-09-17T09:23:03.095000-04:00 |     851 |     8931 |     14.93 |      10   | True       | True    | True        | True             |               56 | True     |                       |               nan    | False          |
| Ca-EEG3-4 | OfflineDay1Session1 |      4.63 | 2022-09-17T09:40:40.554000-04:00 |     711 |     8933 |     14.93 |      10   | True       | True    | True        | True             |              nan | False    |                       |               nan    | True           |
| Ca-EEG3-4 | OfflineDay2Session1 |      4.62 | 2022-09-19T09:30:07.344000-04:00 |     858 |     8933 |     14.93 |      10   | True       | True    | True        | True             |              nan | False    |                       |               nan    | True           |
| Ca-EEG3-4 | Recall1             |      2.3  | 2022-09-20T09:19:43.303000-04:00 |     758 |     4463 |     14.93 |       5   | True       | True    | True        | True             |               33 | True     |                       |               nan    | False          |
| Ca-EEG3-4 | Recall2             |      2.32 | 2022-09-21T08:41:45.720000-04:00 |     831 |     4462 |     14.93 |       5   | True       | True    | True        | True             |               41 | True     |                       |               nan    | False          |
| Ca-EEG3-4 | Recall3             |      2.34 | 2022-09-22T08:40:12.644000-04:00 |     809 |     4463 |     14.93 |       5   | True       | True    | True        | True             |               45 | True     |                       |               nan    | False          |

## Cross day cell registration

The `cell_registration` module exists only in `sub-Ca-EEG2-1_ses-Week.nwb`, so the
archive provides matched cells for one of the two mice.

- 19 tables, one per offline recording, for example `OfflineDay2Session1vsConditioningSessions`
- columns of that table: ['Ca_EEG2-1_FC', 'Ca_EEG2-1_NeutralExposure', 'Ca_EEG2-1_OfflineDay2Session1', 'Ca_EEG2-1_Recall1']
- global cells listed: 1395
- cells found per session: {'Ca_EEG2-1_FC': 597, 'Ca_EEG2-1_NeutralExposure': 623, 'Ca_EEG2-1_OfflineDay2Session1': 625, 'Ca_EEG2-1_Recall1': 336}
- cells present in every column of that table: 82

Each table registers the same three behavioural sessions independently, so the tables
can be compared against each other:

- cells matched across all three behavioural sessions, per table: 65 to 118 (median 89)
- distinct matched cells pooled over all 19 tables: 140
- of those, found by at least 10 of 19 tables: 85
- of those, found by all 19 tables: 65

## Fallback dandiset 001710 (Plitt)

- 139 NWB files, 23 subjects, 78.6 GB total, 0.07 to 1.20 GB per file
- sessions for Cre-1: ['ymaze-day0-scan0-novel-arm-1', 'ymaze-day1-scan0-novel-arm-1', 'ymaze-day2-scan0-novel-arm-1', 'ymaze-day3-scan0-novel-arm-1', 'ymaze-day4-scan0-novel-arm-1', 'ymaze-day5-scan0-novel-arm-1']
- behaviour aligned to imaging: ['block', 'left or right', 'licks', 'position', 'reward', 'speed', 'trial end', 'trial number', 'trial start', 'x position', 'y position']
- ophys contents: ['processing/ophys/Backgrounds/meanImg_ch0', 'processing/ophys/Backgrounds/meanImg_ch1', 'processing/ophys/ImageSegmentation/PlaneSegmentation', 'processing/ophys/dF/dF', 'processing/ophys/fluorescence/fluorescence', 'processing/ophys/neuropil/neuropil fluorescence']
- fields naming cross day cell identity: none
