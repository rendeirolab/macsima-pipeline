"""Stage 4: cell phenotyping (normalize + Astir/FlowSOM + spatial QC).

`workers.run_inproc` is imported lazily (it pulls scanpy/torch/squidpy); keep this
package import light so config/dry-run paths do not load heavy deps.
"""

from __future__ import annotations
