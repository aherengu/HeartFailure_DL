# Public Release Status Notes

This file tracks the public-release status items that still matter after the owner-approved history privacy cleanup.

## Resolved: Git History Privacy Blocker

The repository history was rewritten to replace the old personal Gmail metadata with GitHub noreply metadata.

Current verification status:

- `git log --all --format="%h %an <%ae> | %cn <%ce> | %s"` no longer shows the old Gmail address
- `git shortlog -sne --all` no longer shows the old Gmail address
- tracked files no longer contain the old Gmail address

Operational note:

- this cleanup required a history rewrite
- a `git push --force-with-lease` is intentionally required after owner approval
- old clones, forks, cached mirrors, and downloaded archives may still retain the old commit metadata until they are replaced

## Known Limitation: Dataset Provenance

The repository still includes `input/heart.csv`, and its exact upstream source and redistribution license remain unverified from repository evidence alone.

By owner choice, this is currently documented as an educational limitation rather than a release blocker.

What remains true:

- the repository must not claim dataset provenance or redistribution rights that have not been verified
- the repository must not claim clinical validity
- the dataset warning in [DATASET_PROVENANCE.md](DATASET_PROVENANCE.md) should remain in place

## Safe Commands To Inspect Current Git Author Metadata

```bash
git log --all --format="%h %an <%ae> %s"
git shortlog -sne --all
```

## Ongoing Recommendation

This repository is technically ready for educational portfolio use after the history cleanup, with dataset provenance documented as an unresolved limitation.
