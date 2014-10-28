"""
___________________________________________________________________
Description

This file essentially list all the members that will be imported
upon loading this module with import pyFC.

"""
import clouds
import numpy as np
import mathtools as mt

# TODO make matplotlib dependence in utils optional
import matplotlib.pyplot as pl
import matplotlib.cm as cm

# Map all classes from clouds.py
# Not sure if this is best done here or 
# in __init__.py of the pyFC module
FCSlicer = clouds.FCSlicer
FCAffine = clouds.FCAffine
FCExtractor = clouds.FCExtractor
FCRayTracer = clouds.FCRayTracer
FCDataEditor = clouds.FCDataEditor
FCStats = clouds.FCStats
FractalCube = clouds.FractalCube
GaussianFractalCube = clouds.GaussianFractalCube
LogNormalFractalCube = clouds.LogNormalFractalCube
LogNormalPDF = mt.LogNormalPDF


# Map functions for working with cubes from
# the classes in clouds.py to names here. 
# The fundamental manipulations are
# listed in FractalCube.__init__
slice = FCSlicer().slice
tri_slice = FCSlicer().tri_slice

translate = FCAffine().translate
permute = FCAffine().permute
mirror = FCAffine().mirror

extract_feature = FCExtractor().extract_feature
lthreshold = FCExtractor().lthreshold

pp_raytrace = FCRayTracer().pp_raytrace

mult = FCDataEditor().pow
pow = FCDataEditor().mult

mean = FCStats().mean
std = FCStats().std
var = FCStats().var
rms = FCStats().rms
median = FCStats().median
skew = FCStats().skew
kurt = FCStats().kurt
flat = FCStats().flatness

write_cube = FractalCube().write_cube


# Utility functions

def _build_single_figure(labels=True, colorbar=False):
    """
    Construct a simple figure with a square panel. 
    Used in various routines here so functionalized.
    """

    # Axes widths and heights {whc}: width height colorbar
    # Arbitrary units
    pwa, pha, cwa = 1., 1., 0.025

    # Margins and gaps {lmrtg}: left bottom right top gap
    # Arbitrary units
    if labels and colorbar:
        lma, bma, rma, tma = 0.200, 0.150, 0.200, 0.030
    elif labels and not colorbar:
        lma, bma, rma, tma = 0.200, 0.150, 0.030, 0.030
    elif not labels and colorbar:
        lma, bma, rma, tma = 0.015, 0.015, 0.200, 0.015
    else:
        lma, bma, rma, tma = 0.015, 0.015, 0.015, 0.015
    gwa = 0.01

    # Figure widths and heights in arbitrary units
    fwa = lma + pwa + rma
    if colorbar: fwa += gwa + cwa
    fha = bma + pha + tma

    # Above quantities in normalized units
    pw, ph, cw = pwa / fwa, pha / fha, cwa / fwa
    lm, rm, tm, bm, gw = lma / fwa, rma / fwa, tma / fha, bma / fha, gwa / fwa

    # Figure height width, inches
    fh = 5.
    fw = fh * fwa / fha

    # Create figure and axes
    if colorbar:
        fig = pl.figure(figsize=(fw, fh))
        pax = pl.axes([lm, bm, pw, ph])
        cax = pl.axes([lm + pw + gw, bm, cw, ph])
    else:
        fig = pl.figure(figsize=(fh, fh))
        pax = pl.axes([lm, bm, pw, ph])
        cax = None

    # Ticks or no ticks. The labels are cut off anyway.
    pl.sca(pax)
    if labels:
        pl.tick_params(left='on', top='off', right='off', bottom='on',
                       labelleft='on', labeltop='off', labelright='off', labelbottom='on')
    else:
        pl.tick_params(left='off', top='off', right='off', bottom='off',
                       labelleft='off', labeltop='off', labelright='off', labelbottom='off')

    # Return figure and axes
    return fig, pax, cax


def paint_midplane_slice(fc, pax=pl.gca(), ax=0,
                         scaling='lin', cmap=cm.copper,
                         plottype='imshow', kminlabel=False,
                         vmin=None, vmax=None, cax=None
):
    """
    Create and paint midplane slice into axis pax

    :arg obj fc:          :class:`pyFC.FractalCube` object
    :arg int pax:         Axis to paint midplane slice into
    :arg int ax:          Direction of plane normal. {0|1|2}, default is 0 ("y-z plane")
    :arg str scaling:     Linear "lin" or logarithmic "log" data map
    :arg obj cmap:        Colormap. Any colormap object (default is cm.copper)
    :arg int plottype:    Type of plot {'imshow'|'pcolormesh'}
    :arg int kminlabel:   Boolean, draw label with kmin value?
    :arg int vmin, vmax:  Min max dynamic range of plotted data
    :arg int cax:         Colorbar axis to paint colorbar into

    This plot is independent of cube type.
    """

    # Get data
    fcs = FCSlicer()
    slice = fcs.slice(fc, ax)
    if scaling == 'log': slice = np.log10(slice)

    # Plot slice
    pl.sca(pax)
    if fc.ndim > 1:
        if plottype == 'imshow':
            pl.imshow(slice, vmin=vmin, vmax=vmax, cmap=cmap)
        elif plottype == 'pcolormesh':
            pl.pcolormesh(slice, vmin=vmin, vmax=vmax, cmap=cmap)
            if ax == 2: pl.xlim((0, fc.ni)); pl.ylim((0, fc.nj));
            if ax == 1: pl.xlim((0, fc.ni)); pl.ylim((0, fc.nk));
            if ax == 0: pl.xlim((0, fc.nj)); pl.ylim((0, fc.nk));
        else:
            ValueError('plot_midplane_slice: Unknown plottype, ' + plottype)
    else:
        pl.semilogy(slice)

    # plot kminlabel if desired
    if kminlabel:
        pl.text(0.5, 0.95, r'$k_\mathrm{min}=' + str(int(fc.kmin)) + '$',
                horizontalalignment='center', color='k',
                backgroundcolor=(1., 1., 1., 0.8),
                fontsize=15, transform=pax.transAxes)

    # plot colorbar if desired
    if cax != None:
        cb = pl.colorbar(cax=cax)
        if scaling == 'log':
            cb.set_label(r'$\log_{10}\,\rho$', fontsize=15)
        elif scaling == 'lin':
            cb.set_label(r'$\rho$', fontsize=15)

    # Touch up
    pl.xlabel(r'$x\,(\mathrm{pixel})$', size=15)
    pl.ylabel(r'$y\,(\mathrm{pixel})$', size=15)


def plot_midplane_slice(fc, ax=0, scaling='lin', cmap=cm.copper,
                        plottype='imshow', labels=False,
                        colorbar=False, kminlabel=False,
                        vmin=None, vmax=None
):
    """
    Create and plot midplane slice

    :arg obj fc:          :class:`pyFC.FractalCube` object
    :arg int ax:          Direction of plane normal. {0|1|2}, default is 0 ("y-z plane")
    :arg str scaling:     Linear "lin" or logarithmic "log" data map
    :arg obj cmap:        Colormap. Any colormap object (default is cm.copper)
    :arg int plottype:    Type of plot {'imshow'|'pcolormesh'}
    :arg bool labels:     Boolean, draw tick labels?
    :arg bool colorbar:   Boolean, draw colorbar?
    :arg int kminlabel:   Boolean, draw label with kmin value?
    :arg int vmin, vmax:  Min max dynamic range of plotted data
    """

    # Build figure
    fig, pax, cax = _build_single_figure(labels=labels, colorbar=colorbar)

    # Create plot
    paint_midplane_slice(fc, pax, ax=ax, scaling=scaling, cmap=cmap,
                         plottype=plottype, kminlabel=kminlabel, cax=cax,
                         vmin=vmin, vmax=vmax)


def paint_raytrace(fc, pax=pl.gca(), ax=0,
                   scaling='lin', cmap=cm.copper,
                   plottype='imshow', kminlabel=False,
                   absorb_coeff=1., emiss_coeff=1.,
                   vmin=None, vmax=None, cax=None
):
    """
    Plot a raytraced image into axis. Emissivity has same units as intensity 
    (the ray integration quantity). Assume width of cube box is 1, so that 
    :math:`\Delta x = 1/n{xyz}`

    :arg obj fc:            :class:`pyFC.FractalCube` object
    :arg int pax:           Axis to paint midplane slice into
    :arg int ax:            Direction of plane normal. {0|1|2}, default is 0 ("y-z plane")
    :arg str scaling:       Linear "lin" or logarithmic "log" data map
    :arg obj cmap:          Colormap. Any colormap object (default is cm.copper)
    :arg int plottype:      Type of plot {'imshow'|'pcolormesh'}
    :arg int kminlabel:     Boolean, draw label with kmin value?
    :arg flt absorb_coeff:  Absorption coefficient   
    :arg flt emiss_coeff:   Emission coefficient   
    :arg int vmin, vmax:    Min and max dynamic range of plotted data. If scaling == 'lin', 
                            vmin = 0.005, vmax = 10 are good for vizualization
    :arg int cax:           Colorbar axis to paint colorbar into

    :return:                1 for reaching end of function
    :rtype:                 int


    """

    # Do raytrace and get data
    fcr = FCRayTracer()
    data = fcr.pp_raytrace(fc, absorb_coeff, emiss_coeff, ax)
    if scaling == 'log': data = np.log10(data)

    # Plot the images
    pl.sca(pax)
    if plottype == 'imshow':
        pl.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap)
    elif plottype == 'pcolormesh':
        if ax == 2: pl.xlim((0, fc.ni)); pl.ylim((0, fc.nj));
        if ax == 1: pl.xlim((0, fc.ni)); pl.ylim((0, fc.nk));
        if ax == 0: pl.xlim((0, fc.nj)); pl.ylim((0, fc.nk));
        pl.pcolormesh(data, vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        ValueError('plot_midplane_slice: Unknown plottype, ' + plottype)

    # Draw kmin label if desired
    if kminlabel:
        pl.text(0.5, 0.95, r'$k_\mathrm{min}=' + str(int(fc.kmin)) + '$',
                horizontalalignment='center', color='k',
                backgroundcolor=(1., 1., 1., 0.8),
                fontsize=15, transform=pax.transAxes)

    # Draw colorbar label if desired
    if cax != None:
        cb = pl.colorbar(cax=cax)
        if scaling == 'log':
            cb.set_label(r'$\log_{10}\,I$', fontsize=15)
        elif scaling == 'lin':
            cb.set_label(r'$I$', fontsize=15)

    # Touch up
    pl.xlabel(r'$x\,(\mathrm{pixel})$', size=15)
    pl.ylabel(r'$y\,(\mathrm{pixel})$', size=15)

    # Return 1 for reaching here
    return 1


def plot_raytrace(fc, ax=0, scaling='log', cmap=cm.copper,
                  plottype='imshow', colorbar=True,
                  labels=True, kminlabel=False,
                  absorb_coeff=1.e-7, emiss_coeff=1.,
                  vmin=None, vmax=None
):
    """
    Plot a raytraced image into axis. Emissivity has same units as intensity 
    (the ray integration quantity). Assume width of cube box is 1, so that 
    :math:`\Delta x = 1/n{xyz}`

    :arg obj fc:            :class:`pyFC.FractalCube` object
    :arg int ax:            Direction of plane normal. {0|1|2}, default is 0 ("y-z plane")
    :arg str scaling:       Linear "lin" or logarithmic "log" data map
    :arg obj cmap:          Colormap. Any colormap object (default is cm.copper)
    :arg int plottype:      Type of plot {'imshow'|'pcolormesh'}
    :arg bool colorbar:     Boolean, draw colorbar?
    :arg bool labels:       Boolean, draw tick labels?
    :arg int kminlabel:     Boolean, draw label with kmin value?
    :arg flt absorb_coeff:  Absorption coefficient   
    :arg flt emiss_coeff:   Emission coefficient   
    :arg int vmin, vmax:    Min and max dynamic range of plotted data. If scaling == 'lin', 
                            vmin = 0.005, vmax = 10 are good for vizualization

    :return:                1 for reaching end of function
    :rtype:                 int

    """

    # Build figure
    fig, pax, cax = _build_single_figure(labels=labels, colorbar=colorbar)

    # Create the plot
    paint_raytrace(fc, pax=pax, ax=ax, scaling=scaling, cmap=cmap,
                   plottype=plottype, kminlabel=kminlabel,
                   absorb_coeff=absorb_coeff, emiss_coeff=emiss_coeff,
                   vmin=vmin, vmax=vmax, cax=cax)

    # return 1 for reaching here
    return 1


def plot_field_stats(fc, scaling='lin', cmap=cm.copper, plottype='imshow',
                     vmin=None, vmax=None):
    """
    Create a three-panel plot with:

    1. Cube density slice
    2. Cube single-point pdf
    3. Cube power spectrum

    :arg obj fc:            :class:`pyFC.FractalCube` object
    :arg str scaling:       Linear "lin" or logarithmic "log" data map
    :arg obj cmap:          Colormap. Any colormap object (default is cm.copper)
    :arg int plottype:      Type of plot {'imshow'|'pcolormesh'}
    :arg int vmin, vmax:    Min and max dynamic range of plotted data. If scaling == 'lin', 
                            vmin = 0.005, vmax = 10 are good for vizualization

    :return:                1 for reaching end of function
    :rtype:                 int
    """

    # TODO Try this with AxesGrid

    # Width and heights (a: arbitrary units)
    # {lbrtp}: left bottom right top
    # {mpcg}: margin, panel, colorbar, gap 
    # {wh}: width, height
    lma, bma, tma, rma, = 2.3, 2.0, 0.4, 0.6
    pwa, pha, gwa, gw2a = 17, 17, 7., 4.
    cwa, cgwa = 0.5, 0.1

    # figure height (inches)
    fh = 5.

    # Derived quantities
    fwa = lma + pwa + cgwa + cwa + gwa + pwa + gw2a + pwa + rma
    fha = bma + pha + tma
    lm, bm, tm, rm = lma / fwa, bma / fha, tma / fha, rma / fwa
    pw, ph, gw, gw2 = pwa / fwa, pha / fha, gwa / fwa, gw2a / fwa
    cw, cgw = cwa / fwa, cgwa / fwa
    fw = fh * fwa / fha

    # Create figure and axes
    pl.figure(figsize=(fw, fh))
    pax0 = pl.axes([lm, bm, pw, ph])
    cax0 = pl.axes([lm + pw + cgw, bm, cw, ph])
    pax1 = pl.axes([lm + pw + cgw + cw + gw, bm, pw, ph])
    pax2 = pl.axes([lm + pw + cgw + cw + gw + pw + gw2, bm, pw, ph])

    # Plot midplane slice
    paint_midplane_slice(fc, pax0, scaling=scaling, kminlabel=True,
                         cmap=cmap, plottype=plottype,
                         vmin=vmin, vmax=vmax, cax=cax0)

    # Plot single point field distribution
    paint_pdf(fc, pax1)

    # Plot power spectrum of Gaussian if cube is Lognormal
    if fc.cubetype == 'lognormal':
        paint_power_spec(fc.gnfc(), pax2, 'Gaussian', 'r-')

    # Plot power spectrum of Lognormal
    if fc.cubetype == 'lognormal':
        speclabel = 'Lognormal'
    elif fc.cubetype == 'gaussian':
        speclabel = None
    paint_power_spec(fc, pax2, speclabel, 'b-',
                     k0line=True, target_line=True)

    # Return 1 for reaching here
    return 1


def paint_power_spec(fc, pax=pl.gca(), label=None, line='-',
                     k0line=False, target_line=False):
    """
    Plot power spectrum of fractal cube. Essentially plots output of :func:`pyFC.FractalCube.iso_power_spec`.
    (This function may require more flexibility in its argument list.)

    :arg obj fc:            :class:`pyFC.FractalCube` object
    :arg int pax:           Axis to paint midplane slice into
    :arg str scaling:       Linear "lin" or logarithmic "log" data map
    :arg bool k0line:       Plot k0line?
    :arg bool target_line:  Plot target line?

    :return:                1 for reaching end of function
    :rtype:                 int
    """


    # Create power spectrum
    means, binc = fc.iso_power_spec()

    # Plotting bins
    pmeans = fc.norm_spec(means)
    pmeans, k0val = pmeans[1:], pmeans[0]
    pbinc, k0bin = binc[1:], binc[0]

    # Plot data spectrum
    pl.sca(pax)
    lpbinc, lpmeans = mt.zero_log10(pbinc), mt.zero_log10(pmeans)
    pl.plot(lpbinc, lpmeans, line, label=label)

    # Target spetrum. Normalize roughly to data spectrum
    if target_line:
        lkmin = np.log10(fc.kmin)
        sf = np.s_[lpbinc >= lkmin]
        weights = np.r_[np.diff(pbinc), np.diff(pbinc[-2:])]
        ts = fc.norm_spec(fc.func_target_spec(pbinc))
        fit0 = np.average(mt.zero_log10(pmeans[sf]) -
                          mt.zero_log10(ts[sf]),
                          weights=weights[sf])
        lts = mt.zero_log10(10 ** fit0 * ts)
        pl.plot(lpbinc, lts, 'k-', label='target' if fc.cubetype == 'lognormal' else '')

    # Plot mean (k=0 component) as a horizontal line
    if k0line:
        pl.hlines(np.log10(k0val), np.log10(k0bin),
                  np.log10(pbinc[-1]), line[0], 'dotted',
                  label=r'$k=0$' if fc.cubetype == 'lognormal' else '')

    # Plot adjustments
    pl.xlabel(r'$\log_{10}\, k$', size=15)
    pl.ylabel(r'$\log_{10}\, D(k)$', size=15)
    pl.ylim((-5.0, 5.0))
    if label != None: pl.legend(loc=3)

    # Return 1 for reaching here
    return 1


def plot_power_spec(fc, label=None, line='-', k0line=False, target_line=True):
    """
    Plot power spectrum of fractal cube. Essentially plots output of :func:`pyFC.FractalCube.iso_power_spec`.
    (This function may require more flexibility in its argument list.)

    :arg obj fc:            :class:`pyFC.FractalCube` object
    :arg bool label:        Label on plot?
    :arg str line:          Linestyle
    :arg bool k0line:       Plot k0line?
    :arg bool target_line:  Plot target line?

    :return:                1 for reaching end of function
    :rtype:                 int
    """

    # Build figure
    fig, pax, cax = _build_single_figure(labels=True)

    # Paint the power spectrum
    paint_power_spec(fc, pax=pax, label=label, line=line, k0line=False, target_line=True)

    # Return 1 for reaching here
    return 1


def paint_pdf(fc, pax=pl.gca(), min=None, max=None, step=None):
    """
    Plot a cube's pdf. In the case of a gaussian cube (:attr:`pyFC.GaussianFractalCube.cubetype` == 'gaussian') 
    it plots :math:`dN/dx` vs :math:`x`. In the case of a lognormal cube 
    (:attr:`pyFC.LogNormalFractalCube.cubetype` == 'lognormal') it plots :math:dN/d\log_{10}(x) vs :math:`\log_{10}x`

    :arg obj fc:       :class:`pyFC.FractalCube` object
    :arg int pax:      Axis to paint midplane slice into
    :arg flt min max:  Lower and upper density bin edges
    :arg flt step:     Size of bins

    :return:                1 for reaching end of function
    :rtype:                 int
    """

    # Some defaults
    wfac = 3.
    nsteps = 30.

    if fc.cubetype == 'lognormal':

        # Base e to base 10 conversions
        e2ten, ten2e = np.log10(np.e), np.log(10.)

        ln = mt.LogNormalPDF(fc.mean, fc.sigma)
        hw = wfac * ln.sigma_g * e2ten
        mu = ln.mu_g * e2ten

    elif fc.cubetype == 'gaussian':
        hw = wfac * fc.sigma
        mu = fc.mean

    if min == None: min = mu - hw
    if max == None: max = mu + hw
    if step == None: step = 2. * hw / nsteps

    # Rho space arrays
    x_hist = np.arange(min, max, step)
    x_pdf = x_hist[0:-1] + 0.5 * step

    # Gaussian data plot and theoretical pdfs
    pl.sca(pax)
    if fc.cubetype == 'lognormal':
        res, bins, ignored = pl.hist(np.ravel(np.log(fc.cube) * e2ten), x_hist,
                                     histtype='step', normed=True, align='mid',
                                     label='data')

        gpdf = ln.gpdf(x_pdf * ten2e)
        pl.plot(x_pdf, gpdf * ten2e, label='target')
        pl.ylabel(r'$d N/d \log_{10}\,\rho$', size=15)
        pl.xlabel(r'$\log_{10}\,\rho$', size=15)

    elif fc.cubetype == 'gaussian':
        res, bins, ignored = pl.hist(np.ravel(fc.cube), x_hist,
                                     histtype='step', normed=True, align='mid',
                                     label='data')

        gpdf = mt.gaussian(x_pdf, fc.mean, fc.sigma)
        pl.plot(x_pdf, gpdf, label='target')
        pl.ylabel(r'$d N/d \rho$', size=15)
        pl.xlabel(r'$\rho$', size=15)

    # Plot adjustments
    pl.ylim((0, wfac / (2 * hw)))
    pl.legend()

    # Return 1 for reaching here
    return 1


def plot_pdf(fc, min=None, max=None, step=None):
    """
    Plot a cube's pdf. In the case of a gaussian cube (:attr:`pyFC.GaussianFractalCube.cubetype` == 'gaussian') 
    it plots :math:`dN/dx` vs :math:`x`. In the case of a lognormal cube 
    (:attr:`pyFC.LogNormalFractalCube.cubetype` == 'lognormal') it plots :math:dN/d\log_{10}(x) vs :math:`\log_{10}x`

    :arg obj fc:       :class:`pyFC.FractalCube` object
    :arg flt min max:  Lower and upper density bin edges
    :arg flt step:     Size of bins

    :return:                1 for reaching end of function
    :rtype:                 int
    """

    # Build figure
    fig, pax, cax = _build_single_figure(labels=True)

    # Paint plot
    paint_pdf(fc, pax, min=min, max=max, step=step)

    # Return 1 for reaching here
    return 1

