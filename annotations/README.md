# Manual interior annotation

The zero-shot CLIP filter selected 578 AADB photographs as candidate interior spaces ("CLIP strict" set). Each candidate
was inspected manually to define the verified interior subset used in Study 1.

| File | Content |
|---|---|
| `interior_annotation.xlsx` | All 578 candidates labeled by the first annotator (the second author of the article). |
| `second_rater/second_rater_annotation.xlsx` | Random subsample of 100 candidates labeled independently by another author. |

Sheet `labels` holds one row per image: `key` (AADB image key), `label` (1 = interior space, 0 = not an interior space) and
`note`. In `interior_annotation.xlsx`, `ann_no` and `page` refer to the numbered annotation pages, `source` marks the 100
images of the pilot audit (`audit_100`) and the remaining candidates (`full_annotation`), and `member` gives the path of the
image in the AADB archive. Sheet `criteria` repeats the rules below.

The pilot audit of 100 randomly selected candidates was used to formulate the criteria, which were then applied to all
candidates. The annotation pages showed numbered thumbnails only, so human ratings, CLIP probabilities and harmony scores
were not visible during labeling. Agreement between the two annotators: 94 of 100 images (Cohen's kappa = 0.86, 95%
bootstrap CI [0.74, 0.95]). The first annotator labeled 234 of the 578 candidates as interior spaces.

## Criteria (Table S2 of the article)

| Rule | Criterion |
|---|---|
| Label 1 (interior space) | The main subject of the photograph is an enclosed interior space (room, hall, corridor, library, church nave, shop or restaurant interior, etc.). Enough of the space's boundaries (floor, walls and/or ceiling) is visible to read it as a space. People may be present but must not dominate the frame. |
| Label 0 (not an interior space) | Any of: close-up of an object, furniture piece, artwork or architectural detail; a view mainly of a ceiling, dome, window or opening; vehicle or transport interiors (train, car, escalator shaft); caves and tunnels; exteriors or views to the outside; people-dominated scenes; images too dark or abstract to read the space. |
| Ambiguous cases | Ask: would an interior architect use this photograph to show the space itself? If still unsure, choose the more conservative label (0) and write a short note. |
| Blinding | Label only from the images on the pages. Do not look up ratings, scores or analysis results while labeling. |

License: CC BY 4.0.
