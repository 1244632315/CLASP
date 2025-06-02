"""
Query Gaia DR3 for bulk catalogue of comparison stars for GSO region
"""

import os
import numpy as np
from astroquery.gaia import Gaia
from astropy.table import Table

# unlimited rows
Gaia.ROW_LIMIT = -1

QUERY = """
    SELECT ra, dec, pmra, pmdec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag \
        FROM gaiadr3.gaia_source \
        WHERE CONTAINS(
            POINT('ICRS', gaiadr3.gaia_source.ra, gaiadr3.gaia_source.dec), \
            BOX('ICRS', {}, {}, {}, {}))=1 \
        AND phot_g_mean_mag < {}
        AND pmra IS NOT NULL
        AND pmdec IS NOT NULL;
    """

FIELD_SIZE = 5.  # [deg]
FIELD_CENTRES_RA = np.arange(FIELD_SIZE / 2, 360., FIELD_SIZE)  # [deg]
FIELD_CENTRES_DEC = np.arange(-22.5 + FIELD_SIZE / 2, 22.5, FIELD_SIZE)  # [deg]
MAG_LIMIT = 19  # gaia g

if __name__ == "__main__":
    
    # set up output directory
    out_dir = './catalogue/gaia'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    # iterate through fields and query gaia
    for ra in FIELD_CENTRES_RA:
        for dec in FIELD_CENTRES_DEC:
            print('Query RA: {} DEC: {}'.format(ra, dec))
            
            query = Gaia.launch_job_async(QUERY.format(ra, dec, FIELD_SIZE, FIELD_SIZE, MAG_LIMIT))
            cat = query.get_results()
            
            # save output
            out_path = '{}/gaia_{}_{}_{}_{}.csv'.format(out_dir, ra, dec, FIELD_SIZE, MAG_LIMIT)
            cat.write(out_path, format='csv', overwrite=True)
