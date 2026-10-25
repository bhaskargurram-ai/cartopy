# Copyright Crown and Cartopy Contributors
#
# This file is part of Cartopy and is released under the BSD 3-clause license.
# See LICENSE in the root of the repository for full licensing details.

import io
from unittest import mock

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pytest
import shapely

import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import InterProjectionTransform, _cache_projected_cells
import cartopy.mpl.path as cpath


def test_pcolormesh_partially_masked():
    data = np.ma.masked_all((39, 29))
    data[0:100] = 10

    # Check that a partially masked data array does trigger a pcolor call.
    with mock.patch('cartopy.mpl.geoaxes.GeoAxes.pcolor') as pcolor:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.pcolormesh(np.linspace(0, 360, 30), np.linspace(-90, 90, 40), data)
        assert pcolor.call_count == 1, ("pcolor should have been called "
                                        "exactly once.")


def test_pcolormesh_invisible():
    data = np.zeros((2, 2))

    # Check that a fully invisible mesh doesn't fail.
    with mock.patch('cartopy.mpl.geoaxes.GeoAxes.pcolor') as pcolor:
        ax = plt.axes(projection=ccrs.Orthographic())
        ax.pcolormesh(np.linspace(-75, 75, 3), np.linspace(105, 255, 3), data,
                      transform=ccrs.PlateCarree())
        assert pcolor.call_count == 0, ("pcolor shouldn't have been called, "
                                        "but was.")


def test_savefig_tight():
    nx, ny = 36, 18
    xbnds = np.linspace(0, 360, nx, endpoint=True)
    ybnds = np.linspace(-90, 90, ny, endpoint=True)

    x, y = np.meshgrid(xbnds, ybnds)
    data = np.exp(np.sin(np.deg2rad(x)) + np.cos(np.deg2rad(y)))
    data = data[:-1, :-1]

    ax = plt.subplot(2, 1, 1, projection=ccrs.Robinson())
    ax.pcolormesh(xbnds, ybnds, data, transform=ccrs.PlateCarree())
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')


def test_pcolormesh_arg_interpolation():
    # Test that making the edges from center point inputs
    # properly accounts for the wrapped coordinates appearing
    # in the middle of an array
    x = [359, 1, 3]
    y = [-10, 10]

    xs, ys = np.meshgrid(x, y)
    # Z with the same shape as X/Y to force the interpolation
    z = np.zeros(xs.shape)

    ax = plt.subplot(2, 1, 1, projection=ccrs.PlateCarree())
    coll = ax.pcolormesh(xs, ys, z, shading='auto',
                         transform=ccrs.PlateCarree())

    # Compare the output coordinates of the generated mesh
    expected = np.array([[[358, -20],
                          [360, -20],
                          [2, -20],
                          [4, -20]],
                         [[358, 0],
                          [360, 0],
                          [2, 0],
                          [4, 0]],
                         [[358, 20],
                          [360, 20],
                          [2, 20],
                          [4, 20]]])
    np.testing.assert_array_almost_equal(expected, coll._coordinates)


def test_pcolormesh_datalim():
    # Test that wrapping the coordinates still produces proper data limits
    x = [359, 1, 3]
    y = [-10, 10]

    xs, ys = np.meshgrid(x, y)
    # Z with the same shape as X/Y to force the interpolation
    z = np.zeros(xs.shape)

    ax = plt.subplot(3, 1, 1, projection=ccrs.PlateCarree())
    coll = ax.pcolormesh(xs, ys, z, shading='auto',
                         transform=ccrs.PlateCarree())

    coll_bbox = coll.get_datalim(ax.transData)
    np.testing.assert_array_equal(coll_bbox, [[-2, -20], [4, 20]])

    # Non-wrapped coordinates
    x = [-80, 0, 80]
    y = [-10, 10]

    xs, ys = np.meshgrid(x, y)
    ax = plt.subplot(3, 1, 2, projection=ccrs.PlateCarree())
    coll = ax.pcolormesh(xs, ys, z, shading='auto',
                         transform=ccrs.PlateCarree())

    coll_bbox = coll.get_datalim(ax.transData)
    np.testing.assert_array_equal(coll_bbox, [[-120, -20], [120, 20]])

    # A projection that doesn't support wrapping
    x = [-10, 0, 10]
    y = [-10, 10]

    xs, ys = np.meshgrid(x, y)
    ax = plt.subplot(3, 1, 3, projection=ccrs.Orthographic())
    coll = ax.pcolormesh(xs, ys, z, shading='auto',
                         transform=ccrs.PlateCarree())

    coll_bbox = coll.get_datalim(ax.transData)
    expected = [[-1650783.327873, -2181451.330891],
                [1650783.327873, 2181451.330891]]
    np.testing.assert_array_almost_equal(coll_bbox, expected)


@pytest.mark.parametrize('lon0', [0, -180])
@pytest.mark.parametrize('proj', [
    ccrs.PlateCarree(central_longitude=180),
    ccrs.Robinson(),
    ccrs.Robinson(central_longitude=-160),
    ccrs.Mollweide(),
    ccrs.EqualEarth(central_longitude=100),
], ids=['PlateCarree180', 'Robinson', 'Robinson-160', 'Mollweide',
        'EqualEarth100'])
def test_pcolormesh_wrapped_cells_projected_together(proj, lon0):
    # Cells which cross the edge of the map should be projected all together,
    # rather than one by one with project_geometry, and give the same shapes.
    src = ccrs.PlateCarree()
    lon = np.arange(lon0, lon0 + 360, 5)
    lat = np.linspace(-90, 90, 37)
    data = np.cos(np.deg2rad(lat))[:, np.newaxis] + np.sin(np.deg2rad(lon))

    ax = plt.axes(projection=proj)
    with mock.patch.object(ccrs.Projection, 'project_geometry', autospec=True,
                           side_effect=ccrs.Projection.project_geometry
                           ) as project_geometry:
        mesh = ax.pcolormesh(lon, lat, data, transform=src, shading='auto')
        ax.figure.canvas.draw()
    paths = mesh._wrapped_collection_fix.get_paths()
    # There are two rows of cells over the poles, and a column of cells
    # across the antimeridian.
    assert len(paths) >= lat.size
    # Only a few cells, with edges which are curved in the target projection,
    # are left to project_geometry.
    assert project_geometry.call_count <= 2

    transform = InterProjectionTransform(src, proj)
    for path in paths:
        if np.abs(path.vertices[:, 1]).max() > 90:
            # project_geometry is not a good reference for the cells which
            # overhang the poles.
            continue
        result = cpath.path_to_shapely(transform.transform_path_non_affine(path))
        expected = proj.project_geometry(shapely.Polygon(path.vertices[:-1]),
                                         src)
        # project_geometry can also give zero-area slivers on the far side of
        # the map from a vertex exactly on the cut, which aren't drawn.
        expected = shapely.MultiPolygon(
            [geom for geom in shapely.get_parts(expected)
             if geom.area > 1e-9 * result.area])
        assert shapely.hausdorff_distance(result, expected) <= proj.threshold


def test_pcolormesh_wrapped_polar_cells():
    # project_geometry turns some of the cells which overhang the poles into
    # polygons covering the whole map, which then show through masked cells.
    src = ccrs.PlateCarree()
    proj = ccrs.Mollweide()
    lon = np.arange(0, 360, 1)
    lat = np.linspace(-90, 90, 181)
    data = np.ma.masked_inside(np.ones((lat.size, lon.size)) * lat[:, None],
                               -5, 5)
    ax = plt.axes(projection=proj)
    mesh = ax.pcolormesh(lon, lat, data, transform=src, shading='auto')

    transform = InterProjectionTransform(src, proj)
    map_area = shapely.Polygon(proj.boundary).area
    for path in mesh._wrapped_collection_fix.get_paths():
        cell = cpath.path_to_shapely(transform.transform_path_non_affine(path))
        assert cell.area < 1e-4 * map_area


def _cell(x0, x1, y0, y1):
    return mpath.Path([[x0, y0], [x0, y1], [x1, y1], [x1, y0], [x0, y0]],
                      closed=True)


def test_cache_projected_cells():
    src = ccrs.PlateCarree()
    proj = ccrs.Robinson()
    transform = InterProjectionTransform(src, proj)
    # A cell across the antimeridian, and a cell over the pole.
    paths = [_cell(179, 181, 10, 11), _cell(-10, 10, 89, 91)]
    _cache_projected_cells(paths, src, proj)
    with mock.patch.object(ccrs.Projection, 'project_geometry', autospec=True,
                           side_effect=ccrs.Projection.project_geometry
                           ) as project_geometry:
        across, pole = [transform.transform_path_non_affine(path)
                        for path in paths]
    assert project_geometry.call_count == 0

    # The cell across the antimeridian is split in two, one piece on either
    # side of the map, each a closed ring.
    assert np.count_nonzero(across.codes == mpath.Path.MOVETO) == 2
    assert np.count_nonzero(across.codes == mpath.Path.CLOSEPOLY) == 2
    west, east = cpath.path_to_shapely(across).geoms
    x_edge = proj.transform_point(180, 10, src)[0]
    np.testing.assert_allclose(west.bounds[2], x_edge)
    np.testing.assert_allclose(east.bounds[0], -x_edge)
    # Meridians bend inwards towards the pole.
    corners = proj.transform_points(src, np.array([179, 181]),
                                    np.array([11, 11]))[:, 0]
    np.testing.assert_allclose([west.bounds[0], east.bounds[2]], corners)

    # The cell over the pole is clipped at the pole.
    np.testing.assert_allclose(cpath.path_to_shapely(pole).bounds[3],
                               proj.y_limits[1])


def test_cache_projected_cells_outside_map():
    # Cells which go outside the map are left to project_geometry.
    src = ccrs.PlateCarree()
    proj = ccrs.Mercator()
    paths = [_cell(179, 181, 10, 11), _cell(179, 181, 80, 89)]
    _cache_projected_cells(paths, src, proj)
    transform = InterProjectionTransform(src, proj)
    with mock.patch.object(ccrs.Projection, 'project_geometry', autospec=True,
                           side_effect=ccrs.Projection.project_geometry
                           ) as project_geometry:
        transform.transform_path_non_affine(paths[0])
        assert project_geometry.call_count == 0
        transform.transform_path_non_affine(paths[1])
        assert project_geometry.call_count == 1


@pytest.mark.parametrize('src, proj', [
    (ccrs.PlateCarree(), ccrs.RotatedPole(pole_longitude=120, pole_latitude=40)),
    (ccrs.PlateCarree(), ccrs.InterruptedGoodeHomolosine()),
    (ccrs.PlateCarree(), ccrs.Orthographic()),
    (ccrs.RotatedPole(pole_longitude=120, pole_latitude=40), ccrs.Robinson()),
])
def test_cache_projected_cells_unsupported(src, proj):
    # Projections which aren't cut along a single meridian, and sources which
    # aren't PlateCarree, are left to project_geometry.
    paths = [_cell(179, 181, 10, 11)]
    _cache_projected_cells(paths, src, proj)
    transform = InterProjectionTransform(src, proj)
    with mock.patch.object(ccrs.Projection, 'project_geometry', autospec=True,
                           side_effect=ccrs.Projection.project_geometry
                           ) as project_geometry:
        transform.transform_path_non_affine(paths[0])
    assert project_geometry.call_count == 1
