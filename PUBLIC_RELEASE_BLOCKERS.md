# Public Release Blockers

This file tracks the known issues that still require owner review before the repository is safely public.

## Blocker 1: Dataset Provenance And Redistribution

- The repository includes `input/heart.csv`.
- The exact upstream source and redistribution license could not be proven from repository evidence alone.
- Until that is verified, the repository should not be treated as safe for public redistribution.

See [DATASET_PROVENANCE.md](DATASET_PROVENANCE.md) for the detailed dataset warning and recommended next steps.

## Blocker 2: Personal Email In Git History

An older commit in git history contains a personal Gmail address in the commit metadata:

- commit `911d8df`
- author email: personal Gmail address redacted in the working tree

This remains visible in git history even after current working-tree fixes.

## Why This Cannot Be Fixed Automatically Here

- Removing that email from git history requires a history rewrite.
- If the branch has already been pushed anywhere, that rewrite requires coordination.
- A rewritten branch would later require a force-style push such as `--force-with-lease`.
- That workflow is intentionally not executed in this repository pass.

## Safe Commands To Inspect Current Git Author Emails

```bash
git log --all --format="%h %an <%ae> %s"
git shortlog -sne --all
```

## Optional History-Rewrite Plan

This plan is not executed by this repository pass. It requires explicit owner approval.

1. Create a backup branch before rewriting history.
2. Confirm whether the repository has already been pushed and whether collaborators are affected.
3. Rewrite the old author email to a GitHub noreply address with a history-rewrite tool such as `git filter-repo`.
4. Re-run the git author inspection commands above.
5. Only after explicit approval and coordination, update the remote branch with a force-safe workflow such as `--force-with-lease`.

## Local Future-Commit Mitigation

Future commits for this repository should use a GitHub noreply address in local repo config so the old issue does not continue in new commits.
