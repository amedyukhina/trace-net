from tracenet.utils.points import al_to_coord, coord_to_al


def test_one_point(random_coord):
    y1, x1, y2, x2 = random_coord
    a, l = coord_to_al(x1, y1, x2, y2)
    x2n, y2n = al_to_coord(x1, y1, a, l)
    assert round(x2.item()) == round(x2n.item())
    assert round(y2.item()) == round(y2n.item())
