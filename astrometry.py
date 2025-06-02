"""
Improved astrometry for INT frames
*** Could be more efficient, but works! ***
"""

import os
import sys
import glob
import warnings
import tempfile
import traceback
import subprocess
import numpy as np
from datetime import datetime
import sep
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.table import (
    Table,
    vstack,
    hstack
)
from astropy.coordinates import Angle
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.time import (
    Time,
    TimeDelta
)
from astropy.stats import (
    sigma_clipped_stats,
    sigma_clip
)
from astropy.modeling import (
    fitting,
    models
)
from astropy.utils.exceptions import AstropyWarning
from photutils.aperture import (
    RectangularAperture,
    aperture_photometry
)

# user inputs
DATA_DIR = './imgs/flux'
REF_DIR = './catalogue/gaia'
BP_MASK_PATH = '{}/bp_master.fits'.format(DATA_DIR)
KWD = {'OBSTYPE' : 'TARGET',
       'CCDXBIN' : 2,
       'CCDSPEED' : 'SLOW',
       'WFFBAND' : 'V'}

SIDEREAL_RATE = 15.041  # arcsec/s
PLATESCALE = 0.33
BINNING = 2
BORDER = 30

ASTROMETRY_LOC = '/usr/local/bin/'
TRACKING_RATES = {
    'RA': SIDEREAL_RATE,  # arcsec/s
    'DEC': 0.
}
SCALE_LOWER = 0.6  # arcsec/pixel
SCALE_UPPER = 0.7

GAIA_DR3_REFERENCE_EPOCH_ISOT = '2016-01-01T00:00:00'  # reference epoch for gaia dr3 is 2016.0
GAIA_MAG_LIM = 19  # [gaia g]
GAIA_FIELD_SIZE = 5.
GAIA_DEC_LIM = 22.5
GAIA_CAT_RA = np.arange(GAIA_FIELD_SIZE / 2, 360., GAIA_FIELD_SIZE)
GAIA_CAT_DEC = np.arange(-GAIA_DEC_LIM + GAIA_FIELD_SIZE / 2, GAIA_DEC_LIM, GAIA_FIELD_SIZE)
APERTURE = 10  # pixels

DIAGNOSTICS = 1
VERBOSE = 1

if DIAGNOSTICS:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

# disable astropy warnings - e.g. deprecated headers
warnings.simplefilter('ignore', category=AstropyWarning)

class Frame:
    """
    INT frame
    """
    def __init__(self, frame_header):
        """
        Initialise Frame
        
        Parameters
        ----------
        frame_header : astropy.io.fits.header.Header
            Primary header of INT frame
        """
        self.frame_hdr = frame_header
        self.unpacked = self._unpack()
    
    def _unpack(self):
        """
        Unpack Frame info
        
        Returns
        -------
        True | False : bool
            Success | Failure
        """
        # check frame relevance
        for k in KWD:
            if self.frame_hdr[k] != KWD[k]:
                return 0
        
        # check header contains necessary info
        if not {'EXPTIME', 'DATE-OBS', 'UT-MID', 'RA', 'DEC'} <= set(self.frame_hdr):
            return 0
        
        # store key params as attributes
        self.exptime = self.frame_hdr['EXPTIME']
        
        self.platescale = PLATESCALE * BINNING  # preliminary estimate
        self._star_trail_length()
        
        self.utc_mid = Time(datetime.strptime('{}T{}'.format(self.frame_hdr['DATE-OBS'], self.frame_hdr['UT-MID']), '%Y-%m-%dT%H:%M:%S.%f'))
        self.utc_start = self.utc_mid - TimeDelta((self.exptime / 2) * u.s)
        self.utc_end = self.utc_mid + TimeDelta((self.exptime / 2) * u.s)
        
        self.center = SkyCoord(ra='{} hours'.format(self.frame_hdr['RA']), dec='{} deg'.format(self.frame_hdr['DEC']))
        
        self._chip = 0  # user must load chip
        self._wcs = 0   # user must load wcs
        
        return 1
    
    def _platescale(self):
        """
        Compute platescale 
        """
        # right ascension of extremal corners
        ra_corners, _ = self.chip_to_wcs([1, self.chip_width], 
                                         [1, self.chip_height])
        # platescale along ra and dec axes
        self.platescale = 3600 * (max(ra_corners) - min(ra_corners)) / self.chip_height
        return None
    
    def _star_trail_length(self):
        """
        Compute expected star trail length
        """
        self.star_trail_length = trail_length(self.exptime, self.platescale, SIDEREAL_RATE)
        return None
    
    def load_chip(self, chip_header):
        """
        Load chip to be processed
        
        Parameters
        ----------
        chip_header : astropy.io.fits.header.Header
            Header containing chip info
        
        Returns
        -------
        True | False : bool
            Success | Failure
        """
        if not {'NAXIS1', 'NAXIS2'} <= set(chip_header):
            return 0
        
        self.chip_hdr = chip_header
        
        self.chip_width = self.chip_hdr['NAXIS1']
        self.chip_height = self.chip_hdr['NAXIS2']
        self._chip = 1
        return 1
    
    def load_wcs(self, wcs_header, prelim=False):
        """
        Load WCS information for chip
        
        Parameters
        ----------
        wcs_header : astropy.io.fits.header.Header
            Header containing WCS metadata
        
        Returns
        -------
        True | False : bool
            Success | Failure
        """
        if not self._chip:
            print('FrameWarning: chip not loaded.')
            sys.exit()
        
        # load wcs information
        self.wcs_hdr = wcs_header
        self._wcs = WCS(self.wcs_hdr)
        
        # update attributes
        ra_c, dec_c = self.chip_to_wcs(self.chip_width // 2, self.chip_height // 2)
        self.chip_center = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg)
        
        if prelim:
            self._ps = 0
            for comment in self.wcs_hdr['COMMENT']:
                if 'scale:' in comment:
                    self.platescale = float(comment.split(' ')[1])
                    self._ps = 1
            if not self._ps:
                return 0
        else:
            self._platescale()
        
        self._star_trail_length()
        return 1
    
    def load_corr(self, corr_table):
        """
        Load corr table from preliminary astrometric solution
        
        Parameters
        ----------
        corr_table : astropy.table.Table
            Table containing corr info
        """
        self.corr = corr_table
        return None
    
    def load_stars(self, star_table):
        """
        Load table of star trails
        
        Parameters
        ----------
        star_table : astropy.table.Table
            Table containing star trail info
        """
        self.stars = star_table
        return None
    
    def chip_to_wcs(self, x, y):
        """
        Convert a list of chip xy coords to RA/Dec coords
    
        Parameters
        ----------
        x, y : array-like
            Lists of xy positions [1-indexed] in WCS-solved chip
        
        Returns
        -------
        ra, dec : array-like
            RA/Dec coords corresponding to input chip coords
        """
        if not self._wcs:
            print('FrameWarning: WCS not loaded.')
            sys.exit()
        
        return self._wcs.all_pix2world(x, y, 1, ra_dec_order=True)  # FITS convention, so use 1-based origin
    
    def wcs_to_chip(self, ra, dec):
        """
        Convert a list of RA/Dec coords to chip xy coords
    
        Parameters
        ----------
        ra, dec : array-like
            Lists of RA/Dec coords to be converted
        
        Returns
        -------
        x, y : array-like
            Chip coords corresponding to input RA/Dec coords
        """
        if not self._wcs:
            print('FrameWarning: WCS not loaded.')
            sys.exit()
        
        return self._wcs.all_world2pix(ra, dec, 1)  # FITS convention, so use 1-based origin 
    
    def save_chip(self, out_path):
        """
        Save chip logs to output fits file
        
        Parameters
        ----------
        out_path : str
            Path to output file
        """
        hdu_list = fits.HDUList([])
        
        # log latest wcs info in chip header
        self.frame_hdr.update({'WCS_RA': self.chip_center.ra.deg})
        self.frame_hdr.update({'WCS_DEC': self.chip_center.dec.deg})
        self.frame_hdr.update({'WCS_PS': self.platescale})
        self.frame_hdr.update({'L_STAR': self.star_trail_length})
        
        # log frame header with wcs quality and zero point info
        hdu_list.append(fits.PrimaryHDU(header=self.frame_hdr))
        
        # log star trails and chip header
        if self.stars:
            hdu_list.append(fits.BinTableHDU(data=self.stars, header=self.chip_hdr))
        
        # log astrometry.net corr table and post-fit wcs header
        if self.corr and self._wcs:
            hdu_list.append(fits.BinTableHDU(data=self.corr, header=self.wcs_hdr))
        
        hdu_list.writeto(out_path, overwrite=True)
        return None  
    
    def reset(self):
        """
        Reset Frame for new chip
        """
        self._unpack()
        self.corr = None
        self.stars = None
        return None

def trail_length(exp_time, plate_scale, rate):
    """
    Compute expected trail length from rate and exposure time

    Parameters
    ----------
    exp_time : float
        Exposure time [sec]
    plate_scale : float
        Plate scale [arcsec/pixel]
    rate : float
        Rate, e.g. sidereal [arcsec/sec]

    Returns
    -------
    length : float
        Expected trail length [pixel]
    """
    return (exp_time * rate) / plate_scale     

def gaussian(x, A, x0, sigma):
    """
    Gaussian function
    
    Parameters
    ----------
    x : array-like
        Input x array
    A, x0, sigma : float
        Function parameters - amplitude, peak centre, standard deviation
    
    Returns
    -------
    y : array-like
        Output y array
    """
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) 

def runAstrometry(table_path, scale_low, scale_high, chip_width, chip_height,
                  ra_estimate, dec_estimate, search_radius, solve_timeout=25):
    """
    Run Astrometry.net to solve WCS for a table of stars

    Parameters
    ----------
    table_path : str
        Path to file containing table of stars
    scale_low : float
        Lower bound of platescale [arcsec/pixel]
    scale_high : float
        Upper bound of platescale [arcsec/pixel]
    chip_width : int
        Width of CCD chip [pixel]
    chip_height : int
        Height of CCD chip [pixel]
    ra_estimate : float
        Estimate of center right ascension [degrees]
    dec_estimate : float
        Estimate of center declination [degrees]
    search_radius : float
        Search radius for solver, relative to center estimate [degrees]
    solve_timeout : int, optional
        Timeout limit for solver [seconds]
        Default = 25

    Returns
    -------
    True | False : bool
        Success | Failure
    """
    try:
        astrometry_args = [
            '{}solve-field'.format(ASTROMETRY_LOC),                # location of solve-field
            '{}'.format(table_path),                               # path to file containing star table
            '--no-verify',                                         # ignore existing wcs info
            '--no-tweak',                                          # do not compute SIP polynomial
            '--no-plots',                                          # no output plots
            '--crpix-center',                                      # set wcs reference to center
            '--temp-axy',                                          # temporary axy file
            '--solved', 'none',                                    # no solved file
            '--match', 'none',                                     # no match file
            '--rdls', 'none',                                      # no rdls file
            '--index-xyls', 'none',                                # no index-xyls file
            '--scale-low', '{}'.format(str(scale_low)),            # lower bound of platescale [arcsec/pixel]
            '--scale-high', '{}'.format(str(scale_high)),          # upper bound of platescale [arcsec/pixel]
            '--scale-units', 'arcsecperpix',                       # units of scale bounds
            '--x-column', 'x',                                     # name of column containing x
            '--y-column', 'y',                                     # name of column containing y
            '--sort-column', 'flux',                               # name of column to sort by
            '--width', '{}'.format(str(chip_width)),               # width of chip [pixel]
            '--height', '{}'.format(str(chip_height)),             # height of chip [pixel]
            '--ra', '{}'.format(str(ra_estimate)),                 # estimate of center RA [degrees]
            '--dec', '{}'.format(str(dec_estimate)),               # estimate of center Dec [degrees]
            '--radius', '{}'.format(str(search_radius))            # specify search radius [degrees]
        ]
        subprocess.check_call(astrometry_args,
                              cwd=os.path.dirname(table_path),
                              timeout=solve_timeout,               # timeout limit [secs]
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    except:
        return 0
    else:
        # check if solved
        if os.path.exists(os.path.join(os.path.dirname(table_path),
                                       '{}.wcs'.format(os.path.basename(table_path).split('.')[0]))):
            return 1
        else:
            return 0

def gaia_g_to_johnson_v(catalog):
    """
    Convert Gaia G to Johnson V for a Gaia reference catalogue
    - uses equation from Carrasco & Bellazzini (Gaia DR3 Documentation)
    
    Parameters
    ----------
    catalog : astropy.table.Table
        Gaia reference catalogue containing relevant photometric information
    
    Returns
    -------
    catalog : astropy.table.Table
        Updated reference catalogue with Johnson V magnitudes
    """
    x = catalog['phot_bp_mean_mag'] - catalog['phot_rp_mean_mag']
    y= -0.02704 + 0.01424 * x - 0.2156 * x ** 2 + 0.01426 * x ** 3
    
    catalog['johnson_v'] = catalog['phot_g_mean_mag'] - y
    return catalog

def query_gaia(frame, epoch):
    """
    Query pre-compiled master files (generated via queryGaia.py) to obtain a table of comparison stars for frame

    Parameters
    ----------
    

    Returns
    -------
    comparisons : astropy.table.Table | None
        Table of comparison stars for cross-matching | None if master reference files unavailable/incomplete
    """
    # compute extremal RA and DEC for field
    ra_corners = np.zeros(8)
    dec_corners = np.zeros(8)
    k = 0
    for i in [0, frame.chip_width]:
        for j in [0, frame.chip_height]:
            for t in [frame.utc_start, frame.utc_end]:
                ra, dec = frame.chip_to_wcs(i, j)
                dt = t - frame.utc_mid
                ra_corners[k] = ra + (TRACKING_RATES['RA'] / 3600) * dt.sec
                dec_corners[k] = dec + (TRACKING_RATES['DEC'] / 3600) * dt.sec
                k += 1
    
    # low dec, so use simple excess to cover declination effect and other uncertainties
    excess = 0.2  # [deg]
    
    # check if wrapping necessary
    if np.min(ra_corners) < 90. and np.max(ra_corners) > 270.:
        ra_lower, ra_upper = np.min(ra_corners) + excess, np.max(ra_corners) - excess
    else:
        ra_lower, ra_upper = np.min(ra_corners) - excess, np.max(ra_corners) + excess
    
    dec_lower, dec_upper = np.min(dec_corners) - excess, np.max(dec_corners) + excess
    
    # identify relevant reference catalogues
    ra_upper_idx = np.argmin(np.abs(GAIA_CAT_RA - ra_upper))
    ra_lower_idx = np.argmin(np.abs(GAIA_CAT_RA - ra_lower))
    dec_upper_idx = np.argmin(np.abs(GAIA_CAT_DEC - dec_upper))
    dec_lower_idx = np.argmin(np.abs(GAIA_CAT_DEC - dec_lower))
    
    catalogs = []
    # wrap if necessary
    if ra_lower < 90. and ra_upper > 270.:
        for i in range(ra_lower_idx + 1):
            for j in range(dec_lower_idx, dec_upper_idx + 1):
                catalog_path = '{}/gaia_{}_{}_{}_{}.csv'.format(REF_DIR, GAIA_CAT_RA[i], GAIA_CAT_DEC[j], GAIA_FIELD_SIZE, GAIA_MAG_LIM)
                if os.path.exists(catalog_path):
                    catalog = Table.read(catalog_path, format='csv')
                else:
                    print('GenerateCatalogError: missing reference catalogue.')
                    sys.exit()
                
                catalogs.append(catalog)
        
        for i in range(ra_upper_idx, len(GAIA_CAT_RA)):
            for j in range(dec_lower_idx, dec_upper_idx + 1):
                catalog_path = '{}/gaia_{}_{}_{}_{}.csv'.format(REF_DIR, GAIA_CAT_RA[i], GAIA_CAT_DEC[j], GAIA_FIELD_SIZE, GAIA_MAG_LIM)
                if os.path.exists(catalog_path):
                    catalog = Table.read(catalog_path, format='csv')
                else:
                    print('GenerateCatalogError: missing reference catalogue.')
                    sys.exit()
                
                catalogs.append(catalog)
    # standard case
    else:
        for i in range(ra_lower_idx, ra_upper_idx + 1):
            for j in range(dec_lower_idx, dec_upper_idx + 1):
                catalog_path = '{}/gaia_{}_{}_{}_{}.csv'.format(REF_DIR, GAIA_CAT_RA[i], GAIA_CAT_DEC[j], GAIA_FIELD_SIZE, GAIA_MAG_LIM)
                if os.path.exists(catalog_path):
                    catalog = Table.read(catalog_path, format='csv')
                else:
                    print('GenerateCatalogError: missing reference catalogue.')
                    sys.exit()
            
                catalogs.append(catalog)
    
    catalog = vstack(catalogs)
    
    # trim reference catalogue to relevant search box and remove faint stars to avoid distractors
    if ra_lower < 90. and ra_upper > 270.:
        catalog = catalog[(catalog['ra'] < ra_lower + excess) |
                          (catalog['ra'] > ra_upper - excess)]
    else:
        catalog = catalog[(catalog['ra'] > ra_lower - excess) &
                          (catalog['ra'] < ra_upper + excess)]
    
    catalog = catalog[(catalog['dec'] > dec_lower - excess) &
                      (catalog['dec'] < dec_upper + excess)]
    
    catalog = catalog[catalog['phot_g_mean_mag'] <= GAIA_MAG_LIM - 2.]  #
    
    # apply proper motion offsets
    delta_years = (epoch - Time(GAIA_DR3_REFERENCE_EPOCH_ISOT)).to(u.year).value
    for s, star in enumerate(catalog):
        catalog['ra'][s] = star['ra'] + float(star['pmra'] / 3.6E6) / np.cos(np.radians(star['dec'])) * delta_years
        catalog['dec'][s] = star['dec'] + float(star['pmdec'] / 3.6E6) * delta_years
    
    # avoid blended comparisons
    ra_exclude_delta = (TRACKING_RATES['RA'] / 3600) * frame.exptime + 1. * APERTURE * (frame.platescale / 3600)  # [deg]
    dec_exclude_delta = (TRACKING_RATES['DEC'] / 3600) * frame.exptime + 1. * APERTURE * (frame.platescale / 3600)
    
    catalog['blended'] = np.zeros(len(catalog), dtype=int)
    for s, star in enumerate(catalog):
        dra = (catalog['ra'] - star['ra'])
        ddec = (catalog['dec'] - star['dec'])

        blends = np.logical_and(
            np.abs(dra) < ra_exclude_delta,
            np.abs(ddec) < dec_exclude_delta
        )
        
        if np.sum(blends) == 1:
            catalog['blended'][s] = 0
        else:
            catalog['blended'][s] = 1
    
    #catalog = catalog[catalog['blended'] != 1]
    
    # remove stars outside conversion limits
    bp_rp = catalog['phot_bp_mean_mag'] - catalog['phot_rp_mean_mag']
    catalog = catalog[(bp_rp > -0.5) & (bp_rp < 5.)]
    
    # convert photometry to johnson v
    catalog = gaia_g_to_johnson_v(catalog)
    
    if VERBOSE:
        print('Keeping {} unblended comparison stars'.format(len(catalog)))
    
    return catalog

def cross_match(catalog, star_table, frame):
    """
    Cross-match detected star trails against reference catalog

    Parameters
    ----------
    catalog : astropy.table.Table
        Gaia reference catalogue
    star_table : astropy.table.Table
        Table of stars from astrometric calibration [xy as frame coords]
    frame

    Returns
    -------
    matched_cat : astropy.table.Table
        Table of comparison stars (matched to stars from astrometric calibration)
    zp_delta_mag : array-like
        Individual (star-by-star) zero point measurements
    zp_mean, zp_stddev : float
        Zero point mean and standard deviation
    zp_filter : array-like
        False detection exclusion mask
    """
    # load in coords
    cat_coords = SkyCoord(ra=catalog['ra'] * u.deg,
                          dec=catalog['dec'] * u.deg)
    star_coords = SkyCoord(ra=star_table['ra'] * u.deg,
                           dec=star_table['dec'] * u.deg)
                           
    # cross-match against catalog
    match_idx, _, _ = star_coords.match_to_catalog_sky(cat_coords)
    matched_cat = catalog[match_idx]
    
    # flag blends and objects with inconsistent separations/brightness
    blended = matched_cat['blended'] == 1
    
    wcs_x, wcs_y = frame.wcs_to_chip(matched_cat['ra'], matched_cat['dec'])
    
    delta_x = np.abs(wcs_x - star_table['x'])
    delta_y = np.abs(wcs_y - star_table['y'])
    delta_xy = np.sqrt(delta_x ** 2 + delta_y ** 2)
    
    if DIAGNOSTICS:
        plt.hist(delta_xy, bins=500)
        plt.show()
    
    zp_mask = (delta_xy > 10) | blended
    
    zp_delta_mag = matched_cat['johnson_v'] + 2.5 * np.log10(star_table['flux'] / frame.exptime)
    zp_mean, _, zp_stddev = sigma_clipped_stats(np.array(zp_delta_mag), mask=zp_mask, sigma=3)
    
    zp_filter = np.logical_and.reduce([
        np.logical_not(zp_mask),
        zp_delta_mag > zp_mean - 3 * zp_stddev,
        zp_delta_mag < zp_mean + 3 * zp_stddev
    ])
    
    if DIAGNOSTICS:
        plt.plot(matched_cat['johnson_v'], zp_delta_mag, '.', color='#cccccc')
        plt.plot(matched_cat['johnson_v'][zp_filter], zp_delta_mag[zp_filter], 'k.')
        plt.axhline(zp_mean)
        plt.xlabel('Johnson V')
        plt.ylabel('Zero Point (Johnson V)')
        plt.show()
    
    return matched_cat, zp_delta_mag, zp_mean, zp_stddev, zp_filter

def check_wcs_quality(frame, star_table, catalog):
    """
    Check for WCS quality control, comparing xy positions of observed star trails to those of catalogue matches

    Parameters
    ----------
    chip : sm.skymapper.SMChip
        SMChip containing field for quality check
    star_table : astropy.table.Table
        Table of stars from astrometric calibration
    catalog : astropy.table.Table
        Table of comparison stars (matched)

    Returns
    -------
    header : dict
        Dictionary of catalog-observed x, y, and xy discrepancies
    """
    wcs_x, wcs_y = frame.wcs_to_chip(catalog['ra'], catalog['dec'])
    
    delta_x = np.abs(wcs_x - star_table['x'])
    delta_y = np.abs(wcs_y - star_table['y'])
    delta_xy = np.sqrt(delta_x**2 + delta_y**2)

    header = {}
    prefixes = ['X', 'Y', 'XY']
    for m, median in enumerate([np.median(delta_x),
                                np.median(delta_y),
                                np.median(delta_xy)]):
        if ~np.isnan(median):
            header[prefixes[m]] = median
        else:
            header[prefixes[m]] = -1
    return header

def polynomial_from_header(header, prefix, degree=3):
    """
    Obtain polynomial from header (used for distortion and zero point fitting)

    Parameters
    ----------
    header : astropy.header.Header
        Header containing relevant metadata
    prefix : str
        Keyword prefix
    degree : int, optional
        Degree of polynomial
        Default = 3

    Returns
    -------
    model : astropy.models.Polynomial2D
        Polynomial model
    """
    order_key = '{}_ORDER'.format(prefix)
    if order_key not in header:
        return models.Polynomial2D(degree=degree)

    coeffs = {}
    start = len(prefix) + 1
    for key in header:
        if key.startswith(prefix + '_') and len(key) == start + 3 and key[-2] == '_':
            coeffs['c' + key[start:]] = header[key]

    return models.Polynomial2D(degree=header[order_key], **coeffs)

def fit_distortion(frame, star_table, catalog):
    """
    Apply core (CD matrix) transformation to RA and DEC, ignoring distortion, relative to CRPIX (see SIP paper)

    Parameters
    ----------
    chip : sm.skymapper.SMChip
        SMChip requiring distortion fitting
    star_table : astropy.table.Table
        Table of stars from astrometric calibration
    catalog : astropy.table.Table
        Table of comparison stars (matched)
    """
    U, V = frame.wcs_to_chip(catalog['ra'], catalog['dec'])
    U -= frame.wcs_hdr['CRPIX1']
    V -= frame.wcs_hdr['CRPIX2']
    
    # SIP paper's u and v coords are image coords relative to CRPIX
    u = star_table['x'] - frame.wcs_hdr['CRPIX1']
    v = star_table['y'] - frame.wcs_hdr['CRPIX2']
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # solve for f(u, v) = U - u
        f_init = polynomial_from_header(frame.wcs_hdr, 'A')
        f_fit = fitting.FittingWithOutlierRemoval(fitting.LevMarLSQFitter(), sigma_clip, niter=3, sigma=3.)
        f_poly, _ = f_fit(f_init, u, v, U - u)

        # solve for g(u, v) = V - v
        g_init = polynomial_from_header(frame.wcs_hdr, 'B')
        g_fit = fitting.FittingWithOutlierRemoval(fitting.LevMarLSQFitter(), sigma_clip, niter=3, sigma=3.)
        g_poly, _ = g_fit(g_init, u, v, V - v)

    # update header with the distortion coefficients
    for c, a, b in zip(f_poly.param_names, f_poly.parameters, g_poly.parameters):
        frame.wcs_hdr['A_' + c[1:]] = a
        frame.wcs_hdr['B_' + c[1:]] = b

    # sometimes Astrometry.net doesn't fit distortion terms
    frame.wcs_hdr['A_ORDER'] = f_init.degree
    frame.wcs_hdr['B_ORDER'] = g_init.degree
    frame.wcs_hdr['CTYPE1'] = 'RA---TAN-SIP'
    frame.wcs_hdr['CTYPE2'] = 'DEC--TAN-SIP'

    # update WCS for frame
    frame.load_wcs(frame.wcs_hdr)
    return None

if __name__ == "__main__":
    
    # check data dir
    if not os.path.exists(DATA_DIR):
        print('Data directory not found.')
        sys.exit()
        
    # load bad pixel mask
    with fits.open(BP_MASK_PATH) as bp:
        frame_mask = {}
        for i in range(1,5):
            # ignore bad chip
            if i == 2:
                continue
            frame_mask[i] = bp[i].data.astype(bool)
    
    # iterate through available nights
    for night_dir in glob.glob('{}/UT*'.format(DATA_DIR)):
        
        if VERBOSE:
            print('Processing night {}'.format(night_dir.split('/')[-1].split('T')[-1]))
        
        # for output logs
        out_dir = '{}/calib/output/'.format(night_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        # iterate through available frames
        file_paths = glob.glob('{}/calib/*.fit'.format(night_dir))
        
        log_path = '{}/log.csv'.format(out_dir)
        if os.path.exists(log_path):
            log = Table.read(log_path, format='csv')
        else:
            log = Table(data=[['XXXXXXXX'] * len(file_paths)], names=['FILE'])
        
        for fp, file_path in enumerate(file_paths):
            
            file_prefix = file_path.split('/')[-1].split('.')[0]  # for output logs
            
            # check if already fully processed
            if file_prefix in log['FILE']:
                continue
            
            if VERBOSE:
                print('File {}/{} {}.fit'.format(fp + 1, len(file_paths), file_prefix))
            
            # load data
            with fits.open(file_path) as f:
                
                # check if file is relevant
                primary_hdr = f[0].header
                frame = Frame(primary_hdr)  # load frame info
            
                if not frame.unpacked:
                    continue
                
                # store chips
                frame_data, frame_hdr = {}, {}
                for i in range(1,5):
                    # ignore bad chip
                    if i == 2:
                        continue
                    frame_data[i] = f[i].data.astype(float)
                    frame_hdr[i] = f[i].header
            
            # iterate through chips
            for chip_id in frame_data.keys():
                
                print('Processing chip {}'.format(chip_id))
                
                #if chip_id != 4:
                    #continue
                
                data = frame_data[chip_id]
                mask = frame_mask[chip_id]
                
                # load current chip
                if not frame.load_chip(frame_hdr[chip_id]):
                    print('FrameWarning: Failed to load chip.')
                    continue
                
                ###############################
                ### subtract sky background ###
                ###############################
                try:
                    bkg = sep.Background(data, mask=mask, bw=256, bh=256, fw=10, fh=10)
                except:
                    data = data.byteswap(True).newbyteorder()  # FITS files can be backwards byte order - SEP needs this fixed
                    bkg = sep.Background(data, mask=mask, bw=256, bh=256, fw=10, fh=10)
                
                bkg_rms = bkg.globalrms
                data -= bkg
                
                if DIAGNOSTICS:
                    bkg_image = bkg.back()
                    plt.imshow(bkg_image, interpolation='nearest', cmap='gray', origin='lower')
                    plt.colorbar()
                    plt.show()
                
                ###########################
                ### extract star trails ###
                ###########################
                try:
                    star_table = sep.extract(data, 1.3 * bkg_rms, mask=mask, minarea=100, deblend_cont=1.)  # low threshold to avoid too few candidates
                except:
                    print('ExtractWarning: deblend overflow.')
                    continue
                
                # filter to get star trails
                l_trail = np.sqrt((star_table['xmax'] - star_table['xmin']) ** 2 +
                                  (star_table['ymax'] - star_table['ymin']) ** 2)
                
                star_table = Table(star_table[np.logical_and.reduce([
                    # morphological cuts
                    abs(Angle(star_table['theta'], u.rad).deg) > Angle(np.pi / 2, u.rad).deg - 5.,  # orientation
                    abs(Angle(star_table['theta'], u.rad).deg) < Angle(np.pi / 2, u.rad).deg + 5.,  # orientation
                    l_trail > 0.5 * frame.star_trail_length,          # sub-trails
                    l_trail < 1.5 * frame.star_trail_length,          # big oddities
                    # border cuts
                    star_table['xmin'] > BORDER,
                    star_table['xmax'] < frame.chip_width - BORDER,
                    star_table['ymin'] > 3 * frame.star_trail_length / 4,
                    star_table['ymax'] < frame.chip_height - 3 * frame.star_trail_length / 4
                ])])
            
                if VERBOSE:
                    print('Found {} stellar candidates.'.format(len(star_table)))
                
                # check for bad frames
                if len(star_table) == 0:
                    continue
                
                ###################################################################################
                ### refine trail centroids - place apertures along trail and identify drop-offs ###
                ###################################################################################
                
                flux_fraction = 0.2  # fraction of flux signalling end of trail, from which we track back to along trail to find the half maximum
                aper_r = 2.  # aperture radius
                overhang = 0.25  # fraction of star trail for overhang
                aper_spacing = 0.25  # pixels
                trail_chunk_factor = 8  # proportion factor of apertures taken as representative of trail for flux estimate
                n_aper = int((1 / aper_spacing) * (0.5 + overhang) * frame.star_trail_length)  # number of apertures for along-trail profile
                
                aper_cross_span = 8  # extent of aperture placement for cross-trail profile [pixels]
                n_aper_cross = int((1 / aper_spacing) * aper_cross_span)  # number of apertures for cross-trail profile
                
                fwhm = np.median(2 * np.sqrt(np.log(2) * (2 * star_table['b'] ** 2)))  # median fwhm for detected star trails
                
                for case in ['s', 'c', 'e']:
                    star_table['x{}'.format(case)] = np.zeros(len(star_table))
                    star_table['y{}'.format(case)] = np.zeros(len(star_table))
                    star_table['x{}err'.format(case)] = np.zeros(len(star_table))
                
                remove_idx = []  # for removing blended stars
                for i in range(len(star_table)):
        
                    ## upper trail - end point
                    aper_x = np.linspace(star_table['x'][i], star_table['x'][i] + (0.5 + overhang) * np.cos(star_table['theta'][i]) * frame.star_trail_length, n_aper)
                    aper_y = np.linspace(star_table['y'][i], star_table['y'][i] + (0.5 + overhang) * np.sin(star_table['theta'][i]) * frame.star_trail_length, n_aper)
                    
                    flux_up, _, _ = sep.sum_circle(data, aper_x, aper_y, [aper_r] * n_aper)
                    
                    # identify upper drop-off with low flux fraction
                    up_idx = None
                    trail_lvl = np.median(flux_up[:n_aper // trail_chunk_factor])
                    for j in range(len(flux_up)):
                        if flux_up[j] < flux_fraction * trail_lvl:
                            up_idx = j
                            break
                    
                    # remove likely blends
                    if up_idx is None:
                        remove_idx.append(i)
                        continue
                    
                    # trace back to half maximum point
                    up_idx_hm = None
                    for j in range(3 * int(fwhm / aper_spacing) + 1):  # allow for minor bleeds
                        if flux_up[up_idx - j] > 0.5 * trail_lvl:
                            up_idx_hm = up_idx - j
                            break
                    
                    # remove likely blends
                    if up_idx_hm is None:
                        remove_idx.append(i)
                        continue
                    
                    x_up = aper_x[up_idx_hm] - (0.5 * fwhm * np.cos(star_table['theta'][i]))
                    y_up = aper_y[up_idx_hm] - (0.5 * fwhm * np.sin(star_table['theta'][i]))
                    
                    # refine x_up using cross-trail profile
                    aper_cross_x = np.linspace(x_up - (aper_cross_span / 2) * np.sin(star_table['theta'][i]), x_up + (aper_cross_span / 2) * np.sin(star_table['theta'][i]), n_aper_cross)
                    aper_cross_y = np.linspace(y_up + (aper_cross_span / 2) * np.cos(star_table['theta'][i]), y_up - (aper_cross_span / 2) * np.cos(star_table['theta'][i]), n_aper_cross)
                    flux_up_cross, _, _ = sep.sum_circle(data, aper_cross_x, aper_cross_y, [aper_r] * n_aper_cross)
                    
                    try:
                        parameters, covariance = curve_fit(gaussian, aper_cross_x, flux_up_cross, p0=(flux_up_cross[n_aper_cross // 2], x_up, fwhm)) 
                    except:
                        remove_idx.append(i)
                        continue
                    else:
                        _, x_up, _ = parameters
                        
                        # compute fit error
                        x_up_err = np.sqrt(np.diag(covariance))[1]
                      
                    """
                    # plot image
                    plt.plot(aper_y, flux_up, 'k.')
                    plt.show()
                    
                    fig, ax = plt.subplots()
                    data_mask = np.ma.masked_where(mask, data)
                    m, s = np.mean(data_mask), np.std(data_mask)
                    im = ax.imshow(data_mask, interpolation='nearest', cmap='gray',
                                   vmin=m-s, vmax=m+s, origin='lower')
                    
                    plt.plot(aper_x, aper_y, 'r.')
                    plt.plot(x_up, y_up, 'b.')
                    plt.show()
                    quit()
                    """
        
                    ## lower trail - start point
                    aper_x = np.linspace(star_table['x'][i], star_table['x'][i] - (0.5 + overhang) * np.cos(star_table['theta'][i]) * frame.star_trail_length, n_aper)
                    aper_y = np.linspace(star_table['y'][i], star_table['y'][i] - (0.5 + overhang) * np.sin(star_table['theta'][i]) * frame.star_trail_length, n_aper)
        
                    flux_down, _, _ = sep.sum_circle(data, aper_x, aper_y, [aper_r] * n_aper)
                    
                    # identify lower drop-off with low flux fraction
                    down_idx = None
                    trail_lvl = np.median(flux_down[:n_aper // trail_chunk_factor])
                    for j in range(len(flux_down)):
                        if flux_down[j] < flux_fraction * trail_lvl:
                            down_idx = j
                            break
        
                    # remove likely blends
                    if down_idx is None:
                        remove_idx.append(i)
                        continue
                    
                    # trace back to half maximum point
                    down_idx_hm = None
                    for j in range(3 * int(fwhm / aper_spacing) + 1):  # allow for minor bleeds
                        if flux_down[down_idx - j] > 0.5 * trail_lvl:
                            down_idx_hm = down_idx - j
                            break
                    
                    # remove likely blends
                    if down_idx_hm is None:
                        remove_idx.append(i)
                        continue
                    
                    x_down = aper_x[down_idx_hm] + (0.5 * fwhm * np.cos(star_table['theta'][i]))
                    y_down = aper_y[down_idx_hm] + (0.5 * fwhm * np.sin(star_table['theta'][i]))
                    
                    # refine x_down using cross-trail profile
                    aper_cross_x = np.linspace(x_down - (aper_cross_span / 2) * np.sin(star_table['theta'][i]), x_down + (aper_cross_span / 2) * np.sin(star_table['theta'][i]), n_aper_cross)
                    aper_cross_y = np.linspace(y_down + (aper_cross_span / 2) * np.cos(star_table['theta'][i]), y_down - (aper_cross_span / 2) * np.cos(star_table['theta'][i]), n_aper_cross)
                    flux_down_cross, _, _ = sep.sum_circle(data, aper_cross_x, aper_cross_y, [aper_r] * n_aper_cross)
                    
                    try:
                        parameters, covariance = curve_fit(gaussian, aper_cross_x, flux_down_cross, p0=(flux_down_cross[n_aper_cross // 2], x_down, fwhm)) 
                    except:
                        remove_idx.append(i)
                        continue
                    else:
                        _, x_down, _ = parameters
                        
                        # compute fit error
                        x_down_err = np.sqrt(np.diag(covariance))[1]
                    
                    """
                    plt.plot(aper_cross_x, flux_down_cross, 'k.')
                    plt.plot(aper_cross_x, gaussian(aper_cross_x, amp, x0, sigma), 'b')
                    plt.show()
                    
                    fig, ax = plt.subplots()
                    data_mask = np.ma.masked_where(mask, data)
                    m, s = np.mean(data_mask), np.std(data_mask)
                    im = ax.imshow(data_mask, interpolation='nearest', cmap='gray',
                                   vmin=m-s, vmax=m+s, origin='lower')
                    plt.plot(aper_cross_x, aper_cross_y, 'r.')
                    plt.plot(x_down, y_down, 'b.')
                    plt.show()
                    
                    quit()
                    """
                    
                    ## centroid
                    x_c = (x_up + x_down) / 2
                    y_c = (y_up + y_down) / 2
                    
                    # refine x_c using cross-trail profile
                    aper_cross_x = np.linspace(x_c - (aper_cross_span / 2) * np.sin(star_table['theta'][i]), x_c + (aper_cross_span / 2) * np.sin(star_table['theta'][i]), n_aper_cross)
                    aper_cross_y = np.linspace(y_c + (aper_cross_span / 2) * np.cos(star_table['theta'][i]), y_c - (aper_cross_span / 2) * np.cos(star_table['theta'][i]), n_aper_cross)
                    flux_c_cross, _, _ = sep.sum_circle(data, aper_cross_x, aper_cross_y, [aper_r] * n_aper_cross)
                    
                    try:
                        parameters, covariance = curve_fit(gaussian, aper_cross_x, flux_c_cross, p0=(flux_c_cross[n_aper_cross // 2], x_c, fwhm)) 
                    except:
                        remove_idx.append(i)
                        continue
                    else:
                        _, x_c, _ = parameters
                        
                        # compute fit error
                        x_c_err = np.sqrt(np.diag(covariance))[1]
                   
                    # store refined centroid and cross-trail fit errors
                    star_table['xc'][i] = x_c
                    star_table['yc'][i] = y_c
                    star_table['xcerr'] = x_c_err
                    
                    # store refined start/end points and cross-trail fit errors
                    star_table['xs'][i] = x_down
                    star_table['ys'][i] = y_down
                    star_table['xserr'] = x_down_err
                    
                    star_table['xe'][i] = x_up
                    star_table['ye'][i] = y_up
                    star_table['xeerr'] = x_up_err
    
                # remove blended stars
                star_table.remove_rows(remove_idx)
                
                # check for bad frames
                if len(star_table) == 0:
                    continue
                
                # obtain better flux estimates
                apertures = RectangularAperture(zip(star_table['xc'], star_table['yc']),
                                                w=frame.star_trail_length + APERTURE,
                                                h=APERTURE,
                                                theta=np.median(star_table['theta']))
                
                star_table['flux'] = aperture_photometry(data, apertures, mask=mask, method='subpixel', subpixels=16)['aperture_sum']
                
                if DIAGNOSTICS:
                    # plot image
                    fig, ax = plt.subplots()
                    data_mask = np.ma.masked_where(mask, data)
                    m, s = np.mean(data_mask), np.std(data_mask)
                    im = ax.imshow(data_mask, interpolation='nearest', cmap='gray',
                                   vmin=m-s, vmax=m+s, origin='lower')

                    # plot indicator for each object
                    apertures.plot()
                    plt.plot(star_table['xs'], star_table['ys'], 'g.')
                    plt.plot(star_table['xe'], star_table['ye'], 'r.')
                    plt.show()
                
                ###########################################
                ### astrometry.net for initial solution ###
                ###########################################
                
                # astrometry.net expects 1-index positions
                for case in ['s', 'c', 'e']:
                    star_table['x{}'.format(case)] += 1
                    star_table['y{}'.format(case)] += 1
                
                star_table.sort('flux')  # astrometry.net defaults to descending order, by flux
                star_table.reverse() 
                
                # set xy cols to mid-exposure case
                print('Solving mid-exposure case...')
                
                star_table['x'] = star_table['xc']
                star_table['y'] = star_table['yc']
                
                # run astrometry.net
                with tempfile.TemporaryDirectory() as tempdir:
                    xyls_path = os.path.join(tempdir, 'scratch.xyls')
                    star_table.write(xyls_path, format='fits')

                    if runAstrometry(xyls_path, SCALE_LOWER, SCALE_UPPER, frame.chip_width, frame.chip_height, frame.center.ra.deg, frame.center.dec.deg, 10.):
                        # fetch and load relevant outputs
                        corr_table = Table(fits.getdata(os.path.join(tempdir, 'scratch.corr')))
                        wcs_hdr = fits.getheader(os.path.join(tempdir, 'scratch.wcs'))
                        
                        frame.load_corr(corr_table)
                        frame.load_wcs(wcs_hdr)
                    else:
                        print('AstrometryWarning: failed to solve chip.')
                        continue
                
                # update star table with preliminary wcs
                star_table['ra'], star_table['dec'] = frame.chip_to_wcs(star_table['x'], star_table['y'])
                
                ######################################
                ### fetch appropriate gaia catalog ###
                ######################################
                catalog = query_gaia(frame, frame.utc_mid)
                
                if DIAGNOSTICS:
                    catx, caty = frame.wcs_to_chip(catalog['ra'], catalog['dec'])
                    catx -= 1
                    caty -= 1
                    
                    fig, ax = plt.subplots()
                    data_mask = np.ma.masked_where(mask, data)
                    m, s = np.mean(data_mask), np.std(data_mask)
                    im = ax.imshow(data_mask, interpolation='nearest', cmap='gray',
                                   vmin=m-s, vmax=m+s, origin='lower')
                    
                    plt.plot(catx, caty, 'r.')
                    plt.show()
                
                ##########################################################################
                ### iteratively improve cross-match, WCS fit and zero point estimation ###
                ##########################################################################
                fit_iteration = 0
                revert = 0  # for cases where solution deteriorates
                fail = 0  # for cases where distortion fitting fails
                wcs_headers = []
                while True:
                    # cross-match against catalog to exclude false detections and improve distortion fit
                    try:
                        matched_cat, zp_delta_mag, zp_mean, zp_stddev, zp_filter = cross_match(catalog, star_table, frame)
                    except:
                        fail = 1
                        break
                    
                    # check pre-fit solution quality
                    before_match = check_wcs_quality(frame, star_table[zp_filter], matched_cat[zp_filter])
                    
                    if fit_iteration == 0:
                        for k, v in before_match.items():
                            frame.frame_hdr.update({'{}B'.format(k): v})
                    
                    # store current WCS header in case solution deteriorates
                    wcs_headers.append(frame.wcs_hdr)
                    
                    # fit SIP distortion for trustworthy matches
                    try:
                        fit_distortion(frame, star_table[zp_filter], matched_cat[zp_filter])
                    except:
                        fail = 1
                        break
                    else:
                        fit_iteration += 1
                    
                    # update object positions using new WCS solution
                    star_table['ra'], star_table['dec'] = frame.chip_to_wcs(star_table['x'], star_table['y'])
                    
                    # check post-fit solution quality
                    try:
                        after_match = check_wcs_quality(frame, star_table[zp_filter], matched_cat[zp_filter])
                    except:
                        fail = 1
                        break
                    
                    # check if solution has improved
                    match_improvement = [before_match[k] - after_match[k] for k in after_match]

                    # deteriorated?
                    if np.mean(match_improvement) < 0:
                        revert = 1
                        break
                    
                    # cut-off reached or converged?
                    if fit_iteration > 5 or np.max(match_improvement) < 0.1:
                        break
                
                # move on to next chip if distortion fitting fails
                if fail:
                    print('FitWarning: distortion fitting failed.')
                    continue
                
                # revert back to old solution if better
                if revert:
                    if fit_iteration == 1:
                        frame.load_wcs(wcs_headers[0])  # take original solution
                    else:
                        frame.load_wcs(wcs_headers[-2])  # take two steps back and repeat cross-match
                    
                    # update object positions using reverted WCS solution
                    star_table['ra'], star_table['dec'] = frame.chip_to_wcs(star_table['x'], star_table['y'])
                    
                    # cross-match with catalog
                    try:
                        matched_cat, zp_delta_mag, zp_mean, zp_stddev, zp_filter = cross_match(catalog, star_table, frame)
                    except:
                        print('FitWarning: diverging solution')
                        continue
                    
                    if fit_iteration > 1:
                        # fit SIP distortion for trustworthy matches
                        try:
                            fit_distortion(frame, star_table[zp_filter], matched_cat[zp_filter])
                        except:
                            print('FitWarning: distortion fitting failed.')
                            continue
                        
                        # update object positions using new WCS solution
                        star_table['ra'], star_table['dec'] = frame.chip_to_wcs(star_table['x'], star_table['y'])
                        
                        # cross-match with catalog
                        try:
                            matched_cat, zp_delta_mag, zp_mean, zp_stddev, zp_filter = cross_match(catalog, star_table, frame)
                        except:
                            print('FitWarning: diverging solution')
                            continue
                    
                    # check post-fit solution quality
                    try:
                        after_match = check_wcs_quality(frame, star_table[zp_filter], matched_cat[zp_filter])
                    except:
                        print('FitWarning: diverging solution')
                        continue
                    
                for k, v in after_match.items():
                    frame.frame_hdr.update({'{}A'.format(k): v})
                    
                    if VERBOSE:
                        if k == 'XY':
                            print('Astrometric quality: {} px'.format(v))
                
                # update chip header with zp info and log star table
                frame.frame_hdr.update({'ZP_MEAN': round(zp_mean, 3)})
                if VERBOSE:
                    print('Zero point: {}'.format(round(zp_mean, 3)))
                
                frame.frame_hdr.update({'ZP_STD': round(zp_stddev, 3)})
                frame.frame_hdr.update({'ZP_CNT': np.sum(zp_filter)})
                
                frame.load_stars(hstack([star_table, Table(data=[zp_filter], names=['ZP_FILTER'])]))
                
                # save mid-exposure output file
                out_path = '{}/{}_{}_c.fits'.format(out_dir, file_prefix, chip_id)
                frame.save_chip(out_path)
                
                ###################################################################
                ### repeat astrometric calibration for start/end exposure cases ###
                ###################################################################
                
                for case in ['s', 'e']:
                    
                    print('Solving {}-exposure case...'.format(case))
                    
                    # update star xy to relevant case
                    star_table['x'] = star_table['x{}'.format(case)]
                    star_table['y'] = star_table['y{}'.format(case)]
                
                    # run astrometry.net
                    with tempfile.TemporaryDirectory() as tempdir:
                        xyls_path = os.path.join(tempdir, 'scratch.xyls')
                        star_table.write(xyls_path, format='fits')

                        if runAstrometry(xyls_path, SCALE_LOWER, SCALE_UPPER, frame.chip_width, frame.chip_height, frame.center.ra.deg, frame.center.dec.deg, 10.):
                            # fetch and load relevant outputs
                            corr_table = Table(fits.getdata(os.path.join(tempdir, 'scratch.corr')))
                            wcs_hdr = fits.getheader(os.path.join(tempdir, 'scratch.wcs'))
                            frame.load_corr(corr_table)
                            frame.load_wcs(wcs_hdr)
                        else:
                            print('AstrometryWarning: failed to solve chip.')
                            continue
                    
                    # update star table with preliminary wcs
                    star_table['ra'], star_table['dec'] = frame.chip_to_wcs(star_table['x'], star_table['y'])
                    
                    # fetch appropriate gaia catalog
                    if case == 's':
                        epoch = frame.utc_start
                    else:
                        epoch = frame.utc_end
                    
                    catalog = query_gaia(frame, epoch)
                    
                    # iteratively improve cross-match, WCS fit and zero point estimation 
                    fit_iteration = 0
                    revert = 0  # for cases where solution deteriorates
                    fail = 0  # for cases where distortion fitting fails
                    wcs_headers = []
                    while True:
                        # cross-match against catalog to exclude false detections and improve distortion fit
                        try:
                            matched_cat, zp_delta_mag, zp_mean, zp_stddev, zp_filter = cross_match(catalog, star_table, frame)
                        except:
                            fail = 1
                            break
                        
                        # check pre-fit solution quality
                        before_match = check_wcs_quality(frame, star_table[zp_filter], matched_cat[zp_filter])
                    
                        if fit_iteration == 0:
                            for k, v in before_match.items():
                                frame.frame_hdr.update({'{}B'.format(k): v})
                        
                        # store current WCS header in case solution deteriorates
                        wcs_headers.append(frame.wcs_hdr)
                    
                        # fit SIP distortion for trustworthy matches
                        try:
                            fit_distortion(frame, star_table[zp_filter], matched_cat[zp_filter])
                        except:
                            fail = 1
                            break
                        else:
                            fit_iteration += 1
                    
                        # update object positions using new WCS solution
                        star_table['ra'], star_table['dec'] = frame.chip_to_wcs(star_table['x'], star_table['y'])
                    
                        # check post-fit solution quality
                        try:
                            after_match = check_wcs_quality(frame, star_table[zp_filter], matched_cat[zp_filter])
                        except:
                            fail = 1
                            break
                        
                        # check if solution has improved
                        match_improvement = [before_match[k] - after_match[k] for k in after_match]

                        # deteriorated?
                        if np.mean(match_improvement) < 0:
                            revert = 1
                            break
                    
                        # cut-off reached or converged?
                        if fit_iteration > 5 or np.max(match_improvement) < 0.1:
                            break
                    
                    # move on to next chip if distortion fitting fails
                    if fail:
                        print('FitWarning: distortion fitting failed.')
                        continue
                    
                    # revert back to old solution if better
                    if revert:
                        if fit_iteration == 1:
                            frame.load_wcs(wcs_headers[0])  # take original solution
                        else:
                            frame.load_wcs(wcs_headers[-2])  # take two steps back and repeat cross-match
                    
                        # update object positions using reverted WCS solution
                        star_table['ra'], star_table['dec'] = frame.chip_to_wcs(star_table['x'], star_table['y'])
                    
                        # cross-match with catalog
                        try:
                            matched_cat, zp_delta_mag, zp_mean, zp_stddev, zp_filter = cross_match(catalog, star_table, frame)
                        except:
                            print('FitWarning: diverging solution')
                            continue
                    
                        if fit_iteration > 1:
                            # fit SIP distortion for trustworthy matches
                            try:
                                fit_distortion(frame, star_table[zp_filter], matched_cat[zp_filter])
                            except:
                                print('FitWarning: distortion fitting failed.')
                                continue
                        
                            # update object positions using new WCS solution
                            star_table['ra'], star_table['dec'] = frame.chip_to_wcs(star_table['x'], star_table['y'])
                        
                            # cross-match with catalog
                            try:
                                matched_cat, zp_delta_mag, zp_mean, zp_stddev, zp_filter = cross_match(catalog, star_table, frame)
                            except:
                                print('FitWarning: diverging solution')
                                continue
                    
                        # check post-fit solution quality
                        try:
                            after_match = check_wcs_quality(frame, star_table[zp_filter], matched_cat[zp_filter])
                        except:
                            print('FitWarning: diverging solution')
                            continue
                    
                    for k, v in after_match.items():
                        frame.frame_hdr.update({'{}A'.format(k): v})
                        
                        if VERBOSE:
                            if k == 'XY':
                                print('Astrometric quality: {} px'.format(v))
                    
                    # update chip header with zp info and log star table
                    frame.frame_hdr.update({'ZP_MEAN': round(zp_mean, 3)})
                    if VERBOSE:
                        print('Zero point: {}'.format(round(zp_mean, 3)))
                    
                    frame.frame_hdr.update({'ZP_STD': round(zp_stddev, 3)})
                    frame.frame_hdr.update({'ZP_CNT': np.sum(zp_filter)})
                    
                    star_table['ZP_FILTER'] = zp_filter
                    frame.load_stars(star_table)
                    
                    # save case output file
                    out_path = '{}/{}_{}_{}.fits'.format(out_dir, file_prefix, chip_id, case)
                    frame.save_chip(out_path)
                
                # reset frame for next chip
                frame.reset()
                
                # update progress log
                log['FILE'][fp] = file_prefix
                log.write(log_path, format='csv', overwrite=True)
                
                print('Chip solved.')
