# Generated artifacts

This directory stores local predictions, bootstrap draws, diagnostics, and
fitted-policy records. The root `Makefile` generates these files. They are
ignored by Git because a full research run can produce hundreds of megabytes.

Publication figures, compact result tables, and verification manifests are in
`evidence/`. Method notes are in `docs/`.

The current local artifact tree is preserved when running tests or rebuilding
the notebook. The `clean-evidence` target also refuses to delete it
automatically.
