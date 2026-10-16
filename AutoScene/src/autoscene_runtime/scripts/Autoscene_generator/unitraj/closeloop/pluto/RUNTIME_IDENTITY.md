# Runtime identity audit — reported nuPlan snap after simulation starts

The reported snap has **not** been reproduced. The fixes below address specific
identity defects reproduced against the original Python helper implementations;
they are not a claim that VBD cadence caused the reported behavior. Verification
used CPU fixtures and read-only recorded-data inspection, not a ROS experiment,
hardware run, or GPU inference.

## Substantive fixes with original-versus-patched evidence

1. **Pluto padded-batch token offset.** Both dataset builders prepend the primary
   actor to `agent_tokens`. `pluto_utils.py:478` pads each sample's actor arrays
   to the batch maximum, and `pluto_model.py:147,185` predicts actors `1:A`, not
   the primary. Original `pluto_inference.py:360` dropped the primary token only
   when token count exactly equaled prediction count plus one. That equality
   fails for a sample padded to another sample's larger neighbor count.

   Reproducer: feature tokens `[cbv, neighbor, ego]`, four prediction rows,
   neighbor `x=100`, ego `x=200`, padded rows `x=-999`. Original mapping is
   `[cbv, neighbor, ego]`, and its ego-target helper selects padding `x=-999`.
   Patched mapping is `[neighbor, ego]`, ego selects `x=200`, and postprocessing
   trims padded rows before scoring/world conversion/export. Tests include two
   samples with unequal neighbor counts plus ordinary single ego/CBV outputs.
   Whether the reported run actually produced unequal counts is unknown; normal
   same-type actors can have equal counts and never trigger this defect.

2. **A suffix match can beat an exact target ID.** This is the Pluto inference
   helper `_extract_target_future` (original line 387), not the policy's separate
   exact-first ID resolver. It previously iterated once and returned the first
   exact **or suffix** match. With background tokens `neighbor_12345` before
   `target_12345`, requesting full `target_12345` returned neighbor `x=100` rather
   than target `x=200`. The patch checks full IDs first, accepts only a unique
   five-character legacy suffix, and does not suffix-match short names like ego.

3. **Missing ego silently becomes the nearest/first background actor.** Original
   `_extract_cbv_ego_future` (line 374) returned `pred[0]` if the SDC was absent
   from prediction tokens. The fixture returns neighbor `x=100` before the fix,
   and `None` afterward. Existing scoring already handles unavailable targets.

These helpers are used for prediction/scoring identities. Fixing them does not
prove that a MetaDrive object's pose was rebound or teleported in the reported
run, especially if that run did not use the affected Pluto scoring paths.

## Preventative validation, not an observed-cause claim

`Pluto_test_dataset.py` now verifies that the requested ID resolves to one track,
that the builder resolves that same primary, and that its output token zero is
the requested primary with unique tokens. Both original builders return literal
`ego` for a missing requested BV ID; such fallback must not produce an ego
trajectory later labeled as that BV.

The checks use the requested ID / `metadata.sdc_id`, not a numeric-zero
assumption. The bundled 478 nuPlan summaries were verified to use string
`ego`. Tests execute **both actual baseline builder resolver methods**, covering
implicit string ego, arbitrary explicit BV IDs and integer track keys. For a
noncanonical arbitrary SDC, those legacy builders still select literal `ego`
when no controlled ID is supplied: the adapter now rejects that mismatch. Full
noncanonical-SDC model/category/reference support is not implemented or claimed.

## Related runtime and GUI protections

- Runtime publication, initial-history placement, physical BV feedback, and
  initial-once mirrors now require a traffic-manager track-to-object mapping.
  A missing mapping cannot fall back to a same-named unrelated live object.
  Feedback resolves all eligible actors before moving any actor in the batch.
  This defect has CPU collision fixtures; it remains unattributed to the user's
  observed simulation snap.
- With `history_source=dataset`, edited GUI initial poses are not consumed by
  the backend. The GUI now blocks mismatched preview poses before parameter
  submission/start, explaining that the user must select `initial` or export
  and load a new PKL dataset. It does not switch the history mode automatically.

## Other runtime and VBD audit findings

- `scenario_runtime_node.py:auto_assign_agents_for_current_scenario` defaults to
  enabled and overwrites BV track bindings with proximity-ranked candidates at
  reset. The GUI already sends `auto_assign_agents=false` to retain explicit
  selections. Direct backend launches should check that parameter as well;
  this is not evidence that the reported GUI run enabled auto-assignment.
- The policy resolver already checks exact IDs before a unique suffix. No
  misrouting of that policy method was reproduced. It remains unchanged.
- VBD data preparation copies each selected token's center/world history into
  the matching array slot (`VBDdata_utils.py:334`). The encoder preserves the
  original `agents` tensor even though embeddings are locally normalized
  (`models/vbd/model/modules.py:28,58`). `vbd.py:182` rolls out those current
  states with `global_frame=True`; the adapter preserves world xy and maps
  returned slots by token. No erroneous local-to-world reattachment was found.
- The earlier VBD row-zero repatching fix concerns time indexing, not evidence
  of neighbor identity substitution.

## Existing recorded data

Read-only decoding of embedded ROS1 message schemas required no ROS libraries.
For scenario `64119f3102bc5085`:

| Original bag under `AutoScene/outputs` | World frames | Largest ego step | Largest agent_101 step |
| --- | ---: | ---: | ---: |
| `exp_hybrid_001_scenario_4_20260914_215318.bag` | 80 | 0.835821 m | 0.582675 m |
| `exp_hybrid_001_scenario_4_20260914_162453.bag` | 85 | 0.802668 m | 0.812797 m |

All listed step differences have 0.1 s simulation intervals. The first bag's
closest recorded agent_101/background center distance was 2.142270 m; it does
not show an exact center collapse onto that neighbor. Another bag,
`..._215216.bag`, contains no world states. These old recordings are not known
to be the reported failing run and do not carry sufficient planner/config
evidence to identify its cause. They cannot establish absence of a startup
reset jump before their first captured frame.

## Verification and remaining scope

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_runtime_identity.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_vbd_cadence.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_runtime_object_mapping.py' -v
```

19 prediction identity tests, 31 cadence/runtime-bridge tests, and 26 runtime
object-mapping tests pass (CPU dependency fakes and extracted actual methods,
no neural inference). Original-versus-patched helper comparison also produced
the numerical fixture results above.

Prediction-identity changes are in `pluto/pluto_inference.py` and
`datasets/Pluto_dataset/Pluto_test_dataset.py`. Both large dataset builder files
remain unchanged; tests exercise their actual resolver contracts.

Remaining diagnosis needs the failing run's ego/adv algorithms, selected track
IDs, which actor snaps, and approximate frame (reset, warmup, or first model
plan). Inspect its world poses against those IDs and model primary token/first
world trajectory point. GPU scene reproduction and attribution remain pending.
