debug_mode = False  # uses the (probably) deterministic partition if True
if not debug_mode:
    from .partition import partition
else:
    from .partition import deterministic_debug_partition as partition

# renaming to general names
from .find_best_axis import find_best_axis as find_split_direction
from .collect_axis import collect_axis as project