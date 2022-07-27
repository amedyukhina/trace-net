from tracenet.utils import one_al_to_coord, one_coord_to_al


def test_one_point(random_points):
    y1, x1, y2, x2 = random_points[0, :4]
    a, l = one_coord_to_al(x1, y1, x2, y2)
    x2n, y2n = one_al_to_coord(x1, y1, a, l)
    assert round(x2.item()) == round(x2n.item())
    assert round(y2.item()) == round(y2n.item())
