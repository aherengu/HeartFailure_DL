# Dataset Provenance Status

## Included File

This repository currently includes the following dataset file:

- `input/heart.csv`

## Source And License Status

The exact upstream source and redistribution license for `input/heart.csv` could not be proven from the repository contents, commit history, or nearby metadata available in this repository.

Current status:

- Upstream source: unknown from repo evidence alone
- Redistribution license: unknown from repo evidence alone
- Public redistribution safety: unverified

Because of that, this repository must not claim that the bundled dataset is legally safe to publish or redistribute.

## Data-Quality Warning

The bundled dataset contains physiologically implausible zero values, including:

- many `Cholesterol=0` rows
- at least one `RestingBP=0` row

This repository treats those values as possible anomalies or missing values for educational preprocessing purposes only. That handling is a documented assumption, not a medically validated correction.

## What Must Be Verified Before Public Release

Before making this repository public, the owner should verify all of the following:

1. The exact upstream dataset source.
2. The dataset license and whether redistribution in a public Git repository is allowed.
3. Any required attribution, citation, or notice text.
4. The intended meaning of the `HeartDisease` label and whether the current repository wording matches that definition.
5. Whether the bundled file contains any additional restrictions from a course, platform, or publication.

## Recommended Safe Options

Recommended options before public release:

1. Replace `input/heart.csv` with a documented public dataset whose source and license are explicitly recorded in this repository.
2. Remove the bundled dataset from future public commits until provenance is verified.

If the owner chooses option 2, the safe command sequence is:

```bash
git rm --cached input/heart.csv
echo input/heart.csv>>.gitignore
git commit -m "Stop tracking bundled dataset pending provenance review"
```

Important consequences:

- This removes the dataset from future commits only.
- It does not erase the dataset from existing git history.
- If the repository has already been pushed publicly, history cleanup would be a separate decision.

## Current Recommendation

Do not treat the bundled dataset as public-release ready until its source and license are verified.
