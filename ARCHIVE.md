# Backup / restore status

The whole working stack otherwise lives in exactly one place (the 4 nodes +
`~/glm53-mia` on the head). This repo + the artifacts below are the off-fleet
copy. Keep them on a machine that is NOT one of the Sparks.

## Done (off-fleet, on the bito box)
- **This repo** — curated patch + glue + docs (the reconstruction).
- **`glm53-fleet-backup/glm53-mia_<date>.tar.gz`** (~2.0 GB) — the complete
  `~/glm53-mia` tree (serving repo, overlays, patch sources, build-quant,
  Dockerfiles, logs). `gzip -t` verified. This is the full restore source if the
  curated repo is missing anything.

## TODO (do when GLM is idle — these load the serving head node)
> Do **not** run these while the model is serving (unified-memory swap risk).

1. `docker save` the two custom images off-fleet (they are the hardest to rebuild):
   ```bash
   # run on the head only when idle; nice/ionice to spare the node
   ssh <head> "ionice -c3 nice -n19 docker save glm53-exl3-quant:1.4.5" \
     > glm53-fleet-backup/img_quant-1.4.5.tar      # ~22 GB
   ssh <head> "ionice -c3 nice -n19 docker save glm53-exl3-tp4:keys3" \
     > glm53-fleet-backup/img_keys3.tar            # ~21 GB
   # verify: tar tf <file> | head ; record sha256
   ```
   Restore: `docker load < img_quant-1.4.5.tar` (and keys3) on a fresh node.
2. Optional: push this repo to a private GitHub remote and/or publish the
   aarch64 patch as a gist / PR to `turboderp-org/exllamav3` (true off-site copy
   of the novel part). Needs the operator's GitHub auth — not done automatically.

## Restore order after a node loss
1. `docker load` the two image tars onto the rebuilt node(s).
2. Restore `~/glm53-mia` from the tarball (or clone this repo's `serving/`).
3. Re-NFS-export the weights; fill `.env.tp4`; `start-tp4.sh` (see docs/BUILD.md).
If the image tars are missing: rebuild from `exllamav3-aarch64-patch/` +
`serving/Dockerfile.*` per docs/BUILD.md (the slow path — hours).
