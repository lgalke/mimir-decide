from conftest import make_decision as mk
from mimir_decide.train import LengthBucketBatches


def test_skipping_batches_gives_exactly_the_remainder_of_the_same_order():
    items = [mk(id=f"d{i}", state="word " * (i % 17 + 1)) for i in range(500)]
    full = LengthBucketBatches(items, 8, 0, 0).batches
    part = LengthBucketBatches(items, 8, 0, 0, skip=23).batches
    assert part == full[23:] and len(part) == len(full) - 23
    assert LengthBucketBatches(items, 8, 0, 1).batches != full           # another epoch has another order
