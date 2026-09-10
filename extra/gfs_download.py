#!/bin/bash

CACHE_DIR="/e/project1/e-ben-2026b09-120/earth2_cache/gfs"
mkdir -p $CACHE_DIR

# Data for the 1st of January 2024
# wget https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20231231/18/atmos/gfs.t18z.pgrb2.0p25.f000 -O $CACHE_DIR/gfs.20231231.t18z.pgrb2.0p25.f000
# wget https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20240101/00/atmos/gfs.t00z.pgrb2.0p25.f000 -O $CACHE_DIR/gfs.20240101.t00z.pgrb2.0p25.f000

# Data for the 15th of January 2024
# wget https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20240114/18/atmos/gfs.t18z.pgrb2.0p25.f000 -O $CACHE_DIR/gfs.20240114.t18z.pgrb2.0p25.f000
# wget https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20240115/00/atmos/gfs.t00z.pgrb2.0p25.f000 -O $CACHE_DIR/gfs.20240115.t00z.pgrb2.0p25.f000

# Data for the 1st of August 2026
wget https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260731/18/atmos/gfs.t18z.pgrb2.0p25.f000 -O $CACHE_DIR/gfs.20260731.t18z.pgrb2.0p25.f000
wget https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260801/00/atmos/gfs.t00z.pgrb2.0p25.f000 -O $CACHE_DIR/gfs.20260801.t00z.pgrb2.0p25.f000

# Data for the 15th of August 2026
wget https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260814/18/atmos/gfs.t18z.pgrb2.0p25.f000 -O $CACHE_DIR/gfs.20260814.t18z.pgrb2.0p25.f000
wget https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260815/00/atmos/gfs.t00z.pgrb2.0p25.f000 -O $CACHE_DIR/gfs.20260815.t00z.pgrb2.0p25.f000

echo "Downloaded GFS GRIB files to $CACHE_DIR"