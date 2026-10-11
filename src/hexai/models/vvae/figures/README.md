# Figures

Regenerate the whole set from one checkpoint:

```sh
uv run python -m hexai.models.vvae.figures                       # defaults below
uv run python -m hexai.models.vvae.figures --checkpoint .runs/<name>.pt --index 200
```

`figures.py` rebuilds the model from the checkpoint's own config, so a run at a different grid or latent width regenerates without editing anything. Every figure is drawn from the same window of the **held-out** building, so nothing here is a scene the model trained on.

The default window is a wall running away to the left with boxes on the floor to the right. It was picked on two things: the boxes *occlude*, so there is a wedge of unknown behind them and the third state has something to show; and the occupied mass sits mostly **below** the lens, so the top-down plan is furniture rather than ceiling tiles. Aiming up is the deceptive case — the camera flies at 1.75 m in sim, so a window tilted at the ceiling fills the floor plan with the one surface a walking robot never has to care about. 56.3% answerable, which also keeps the ungraded region from swallowing the error panel.

`--index` picks another window. Head jitter is drawn per load, so `--seed` pins which way the head was aimed — without it the same index gives a different figure every run.

The default is the last epoch of the 200-epoch onecycle run — 11 buildings, Office held out, `+0.526` answerable skill. Deliberately not that run's `-best.pt`: it sits at epoch 11, and a single validation swings about as far as the whole run improves, so the best raw epoch is the luckiest draw rather than the best model.

## The two that need no checkpoint

Exact arithmetic on a window. When a trained model looks wrong these are what say whether the geometry underneath it is sound, so they are worth regenerating first.

| | |
|---|---|
| `three-state-carving.png` | Two rows at four heights. The top row is the anchor camera's view with the pixels that land in that height slab lit and the rest dimmed, so you can see which part of the picture made the slice below it. The bottom row is the carving: dark unknown, teal free, yellow occupied, camera at bottom centre. The frustum opens from the camera and stops at surfaces, and the notch cut into the free cone is the pillar's shadow — a lift alone would call all of that empty. Viridis here is three categories, not depth; every actual depth panel uses `DEPTH_CMAP` |
| `slice-correspondence.png` | The anchor camera's view and its depth with one row marked, then those same pixels unprojected onto the floor plan they produced. The only cheap way to tell a real wall from a transform bug: if the green dots do not land on the carved surface, the pose convention is wrong. It reads the **middle** frame, whose pose is identity in the box, so no transform is needed to place its rays |

## The four that show the model

| | |
|---|---|
| `novel-view.png` | Seven panels: truth from a pose the encoder never saw, what the decoder renders from it, the graded error, then the lifted volume against the decoded one from above and from the side. The image pair alone cannot separate undertrained from broken — a saturated density field also renders flat — so the volumes sit beside it. The cyan bar in the side views is the hexapod standing, drawn at the box's lowest row — a scale reference rather than a clearance read, since sim flies the camera at 1.75 m and the box only reaches 1.5 m below it, so the real floor is outside the box |
| `pose-sweep.png` | Truth over render across a yaw sweep. A row that stays identical across the sweep is a latent that has stopped depending on viewpoint, which no single-pose panel can show. Ours changes with the pose and holds little structure — that gap is the open problem |
| `latent-top-down.png` | The latent from above, one panel per channel, beside the volume it decodes to. The architecture's whole claim is that the latent's axes are space; a latent reading the same in every cell has stopped being a map however well the render scores |

## In three dimensions

Every figure above throws an axis away — a slice fixes height, a floor plan sums it, a side view sums across. [`voxels.ipynb`](voxels.ipynb) is the same window as a volume you can spin: the frames that went in, the carving as free and occupied solids, the lifted volume beside the decoded one, and the latent itself as one square per latent voxel with a channel selector.

It is the only view that shows the free space is a *cone* opening away from the camera with the furniture's shadow cut out of it, and the only one where a latent channel reading the same everywhere is obvious at a glance.

It needs `plotly` and `nbformat`, both in the `dev` dependency group — plotly renders a figure *through* nbformat, so without it the first figure raises `Mime type rendering requires nbformat>=4.2.0`. `uv sync` installs the group.

### The grey in the error panel

Grey is not "no error", it is **not graded at all**. Aiming the head puts the held-out view somewhere the context frames never covered, and a pixel no context frame saw has no answer the model could have got right — so the loss masks it and the panel greys it. Only the coloured region is scored, and the title says what fraction that is.

This is why `skill_answerable` is the number to read and plain `skill` is not: a masked run is never taught the grey, and scoring it there charges the model for error no gradient ever reached. Getting that wrong once invalidated every conclusion drawn from a whole set of runs.

## What the current set does and does not show

The geometry works and the figures show it: the carving separates seen from unseen, and the lifted volume is a crisp floor plan. The decoded volume beside it is a smeared version of the same room, and the render tracks the query pose without holding structure. Depth error averages about 1.0 m over a 4 m box across the held-out building.

That blur is the objective rather than the model failing — squared error on an ambiguous ray is minimised by the mean of the possibilities, so it renders a surface that was never there. `.agents/learned.md` has the measurements, `.agents/todo.md` has what is being tried about it.

## Adding one

Put it in `figures.py` so the set stays regenerable from a single checkpoint. Figures produced by hand drift out of date silently and nobody can tell which model they came from — the two `side-view-panels*.png` files that used to sit here were exactly that, and `novel-view.png` now carries those panels.
