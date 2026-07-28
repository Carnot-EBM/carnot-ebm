# These 12 cells are DISCARDED, not deleted (never-prune)

Written 16:19-16:36 UTC on 2026-07-27. `fourarm.py` gained its per-arm engine-store
isolation (`os.environ["CARNOT_ARC_E3_DIR"] = str(_arm_store)`, ~line 580) at 18:38 --
over two hours LATER. So these cells were produced by the version that let every arm
write the SHARED `results/arc_e3/<game>/world_model.py`, which is exactly the
contamination that caused the run to be stopped (see the workflow task note "STOPPED
contaminating four-arm run; origin engines frozen").

They cannot be VALIDATED, only dated: the cell schema records no engine-store
provenance, so there is no field that would reveal which store an induction wrote to.
Reusing them would have been silent -- `fourarm.py` resumes by cell filename, so a
stale cell is indistinguishable from a fresh one at read time.

Kept because they are real measurements of *something* and the never-prune rule applies
to the research record, not only to conclusions. Do NOT move them back into `cells/`.
