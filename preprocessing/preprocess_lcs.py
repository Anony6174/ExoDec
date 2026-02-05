import random
from astropy.io import fits
from astropy.table import Table 
import matplotlib.pyplot as plt
import os, sys
import pandas as pd
import numpy as np
import tensorflow as tf
import csv
from collections import defaultdict
from tqdm import tqdm
import multiprocessing
import math
import warnings
import pprint
import argparse
import wotan
from astropy.stats import sigma_clip
from wotan import t14
with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import lightkurve as lk


_BASE_PATH = ''
# tic2file = None
# tic_catalog = None
# file2tic = None



def _set_float_feature(ex, name, value):
        assert name not in ex.features.feature, "Duplicate feature: %s" % name
        ex.features.feature[name].float_list.value.extend([float(v) for v in value])

def _set_bytes_feature(ex, name, value):
        assert name not in ex.features.feature, "Duplicate feature: %s" % name
        ex.features.feature[name].bytes_list.value.extend([str(v).encode("utf-8") for v in value])


def _set_int64_feature(ex, name, value):
        assert name not in ex.features.feature, "Duplicate feature: %s" % name
        ex.features.feature[name].int64_list.value.extend([int(v) for v in value])

def load_tic2file(filename):
        global tic2file, file2tic
        if not os.path.isfile(filename):
                raise FileNotFoundError
        tic2file = pd.read_csv(filename, index_col='tic_id')
        # In case we have multiple curves for the same known TIC
        tic2file = tic2file[~tic2file.index.duplicated(keep='first')]
        file2tic = pd.read_csv(filename, index_col='Filename')

        return tic2file, file2tic



def normalize_view(lc, bins):
    binned = lc.bin(bins=bins, method='median') - 1
    binned = (binned / np.abs(binned.flux.min())) * 2.0 + 1
    return binned.remove_nans()



def load_catalog(filename, enableImputation=False):
    global tic_catalog

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Catalog file not found: {filename}")

    print(f"📄 Loading catalog from: {filename}")
    tic_catalog = pd.read_csv(filename, index_col='TIC_ID')

    # Relevant columns
    final_columns = [
        'T0', 'Depth', 'Period', 'Duration', 'TMag', 'Teff', 'Radius', 'NumTransits', 'snr', 'sde',
        'Mass', 'a', 'b', 'logg', 'distance', 'lum', 'rho', 'rp_rs',
        'DepthOdd', 'DepthEven', 'Disposition', 'tdur'
    ]
    raw_columns = [
        'T0', 'Depth', 'Period', 'Duration', 'TMag', 'Disposition', 'NumTransits',
        'snr', 'sde', 'rp_rs', 'DepthOdd', 'DepthEven', 'tdur'
    ]

    # Normalize column names if needed
    if 'Logg' in tic_catalog.columns:
        tic_catalog['logg'] = tic_catalog.pop('Logg')

    # Drop irrelevant columns
    tic_catalog = tic_catalog.loc[:, tic_catalog.columns.isin(final_columns)]

    # Compute tdur column using default values first
    print("⏱️  Computing transit durations (tdur)...")
    default_tdur = t14(R_s=1.0, M_s=1.0, P=13.5, small_planet=False)
    tic_catalog['tdur'] = default_tdur

    # Update tdur values using available stellar parameters
    for tic in tic_catalog.index:
        try:
            tic_catalog.at[tic, 'tdur'] = t14(
                R_s=tic_catalog.at[tic, 'Radius'],
                M_s=tic_catalog.at[tic, 'Mass'],
                P=tic_catalog.at[tic, 'Period'],
                small_planet=False
            )
        except Exception:
            # Use default tdur if parameters are missing or invalid
            continue

    # Remove duplicate TIC entries
    tic_catalog = tic_catalog[~tic_catalog.index.duplicated(keep='first')]
    print(f"📌 Unique TIC entries loaded: {len(tic_catalog)}")

    # Drop rows with missing values if imputation is disabled
    if not enableImputation:
        before_drop = len(tic_catalog)
        tic_catalog = tic_catalog.dropna()
        print(f"❌ Dropped {before_drop - len(tic_catalog)} rows due to missing data.")

    # Impute and normalize numeric columns (except raw ones)
    for col in tic_catalog.columns:
        if col in raw_columns:
            continue

        # Replace string 'None' values with NaN and convert to float
        tic_catalog[col] = pd.to_numeric(tic_catalog[col], errors='coerce')

        if enableImputation:
            missing_count = tic_catalog[col].isna().sum()
            if missing_count > 0:
                print(f"🔧 Imputing {missing_count} missing values in column: {col}")
                tic_catalog[col].fillna(tic_catalog[col].median(), inplace=True)

        # Normalize if not already in small range [-2, 2]
        col_min, col_max = tic_catalog[col].min(), tic_catalog[col].max()
        if pd.notnull(col_min) and pd.notnull(col_max) and not (-2 < col_min and col_max < 2):
            median = tic_catalog[col].median()
            std = tic_catalog[col].std()
            if std != 0 and not np.isnan(std):
                tic_catalog[col] = (tic_catalog[col] - median) / std
            else:
                print(f"⚠️ Skipping normalization for column: {col} (std = 0)")

    print(f"✅ Catalog ready with shape: {tic_catalog.shape}")
    return tic_catalog


#####################################################################################
def _load_lightcurve_list(file_list):
    """
    Internal helper to read and stitch a list of lightcurve FITS files.

    Parameters:
        file_list (List[str]): List of file paths.

    Returns:
        LightCurve or None: Stitched PDCSAP_FLUX lightcurve or None if loading fails.
    """
    lcfs = []
    for f in file_list:
        if not os.path.exists(f):
            print(f"[SKIP] File not found: {f}")
            continue
        try:
            lcf = lk.search.open(f)
            lcfs.append(lcf)
        except (OSError, TypeError) as e:
            print(f"[SKIP] Failed to read {f}: {e}")
            continue

    if not lcfs:
        return None

    try:
        collection = lk.LightCurveFileCollection(lcfs)
        return collection.PDCSAP_FLUX.stitch()
    except Exception as e:
        print(f"[FAIL] Stitching failed: {e}")
        return None

###################################################################################################
def load_lightcurve(filename):

    return _load_lightcurve_list([filename])


def load_lightcurve_from_files(filenames):

    return _load_lightcurve_list(filenames)


def load_lightcurves(tess_id):

    try:
        filenames = list(tic2file['Filename'][tess_id])
    except Exception as e:
        print(f"[ERROR] Couldn't get filenames for TIC {tess_id}: {e}")
        return None

    file_paths = [os.path.join(_BASE_PATH, f) for f in filenames]
    print(f"[INFO] Loading {len(file_paths)} files for TIC {tess_id}")
    return _load_lightcurve_list(file_paths)
###########################################################################################################

def get_folded_lightcurve(tess_id, lc_raw, period, t0, duration_hours):
    """
    Uses Lightkurve's flatten() to detrend the lightcurve and return a folded lightcurve.

    Parameters:
        tess_id (int): TIC ID of the object.
        lc_raw (LightCurve): Raw lightcurve.
        period (float): Transit period.
        t0 (float): Transit epoch.
        duration_hours (float): Transit duration in hours.

    Returns:
        LightCurve: Folded and flattened lightcurve or None on failure.
    """
    try:
        # Step 1: Clean the lightcurve
        lc_clean = lc_raw.remove_outliers(sigma=20, sigma_upper=5)

        # Step 2: Estimate fractional transit duration
        fractional_duration = (duration_hours / 24.0) / period

        # Step 3: Build mask around transit
        temp_fold = lc_clean.fold(period, t0=t0)
        phase_mask = np.abs(temp_fold.phase) < (1.5 * fractional_duration)
        in_transit_times = temp_fold.time_original[phase_mask]
        transit_mask = np.in1d(lc_clean.time, in_transit_times)


        # Step 4: Flatten with transit masked out
        lc_flat, _ = lc_clean.flatten(return_trend=True, mask=transit_mask)

        # Step 5: Fold again after detrending
        lc_fold = lc_flat.fold(period, t0=t0)

        return lc_fold

    except ValueError as e:
        print(f"⚠️  ValueError for TIC {tess_id} in get_folded_lightcurve: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error for TIC {tess_id} in get_folded_lightcurve: {e}")
        return None



def process_lightcurve(tess_id, lc_raw, period, t0, duration_hours):
    """
    Given a raw lightcurve, convert it into global, local, and phase-shifted views.
    """
    try:
        # Fold the lightcurve
        lc_fold = get_folded_lightcurve(tess_id, lc_raw, period, t0, duration_hours)
        if lc_fold is None:
            print(f"⚠️  Folded LC is None for TIC {tess_id}")
            return None, None, None

        # Generate global view (no normalization warning with pre-normalized inputs)
        lc_global = normalize_view(lc_fold, bins=201)

        # Generate local view (zoomed around transit)
        fractional_duration = (duration_hours / 24.0) / period
        phase_mask = (lc_fold.phase > -2.0 * fractional_duration) & (lc_fold.phase < 2.0 * fractional_duration)
        lc_zoom = lc_fold[phase_mask]
        lc_local = normalize_view(lc_zoom, bins=81)

        if len(lc_local.to_pandas()) == 0:
            print(f"⚠️  Empty local LC for TIC {tess_id}")
            return None, None, None

        # Shift T0 by 0.25 * period to catch secondary eclipses
        shifted_t0 = t0 + 0.25 * period
        lc_shifted = get_folded_lightcurve(tess_id, lc_raw, period, shifted_t0, duration_hours)
        lc_global_shifted = normalize_view(lc_shifted, bins=201)

        return lc_global, lc_local, lc_global_shifted

    except ValueError as e:
        print(f"❌ ValueError for TIC {tess_id}: {e}")
        return None, None, None
    except Exception as e:
        print(f"❌ Unexpected error in process_lightcurve for TIC {tess_id}: {e}")
        return None, None, None


def _process_lightcurve(tess_id, lc_raw, tic_catalog):
    if tess_id not in tic_catalog.index:
        print(f"⚠️  {tess_id} not found in catalog index.")
        return None, None, None

    try:
        period = tic_catalog.at[tess_id, 'Period']
        duration_hours = tic_catalog.at[tess_id, 'Duration']
        t0 = tic_catalog.at[tess_id, 'T0']

        if any(pd.isna([period, duration_hours, t0])):
            print(f"⚠️  Missing period/duration/T0 for TIC {tess_id}")
            return None, None, None

        return process_lightcurve(tess_id, lc_raw, period, t0, duration_hours)

    except KeyError as e:
        print(f"❌ KeyError while accessing TIC {tess_id}: {e}")
        return None, None, None
    except Exception as e:
        print(f"❌ Unexpected error in processing TIC {tess_id}: {e}")
        return None, None, None


def build_halfphase_views(tess_id, lc_raw):
        '''Generate zoom ins by folding around t0, t0-period, t0-period/2.  The objective here
        is to zoom on odd/even transits and the mid-point of the transit.'''
        try:
                period = tic_catalog['Period'][tess_id]
                duration_hours = tic_catalog['Duration'][tess_id]
                t0 = tic_catalog['T0'][tess_id]
                lc_fold = get_folded_lightcurve(tess_id, lc_raw, period, t0, duration_hours)
                bins = 81
                fractional_duration = (duration_hours / 24.0) / period
                phase_mask = (lc_fold.phase > -2.0*fractional_duration) & (lc_fold.phase < 2.0*fractional_duration)
                lc_zoom = lc_fold[phase_mask]
                phase_fold_t0 = lc_zoom.bin(bins=bins, method='median') - 1

                lc_fold = get_folded_lightcurve(tess_id, lc_raw, period, t0 - 0.5 * period, duration_hours)
                phase_mask = (lc_fold.phase > -2*fractional_duration) & (lc_fold.phase < 2.0*fractional_duration)
                lc_zoom = lc_fold[phase_mask]
                phase_fold_half = lc_zoom.bin(bins=bins, method='median') - 1
                lc_phase = combine_odd_even(phase_fold_half, phase_fold_t0)

                missing = np.sum(np.isnan(lc_phase.flux))
                # Fill in NaN's using neighbor values...
                if missing > 0:
                        mask = np.isnan(lc_phase.flux)
                        lc_phase.flux[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), 
                                                                lc_phase.flux[~mask])
                return lc_phase
        except:
                return None
####################################################################################################################3

def combine_odd_even(lc_odd_zoom, lc_even_zoom):
        concat_view_t = np.concatenate((lc_odd_zoom.time, lc_even_zoom.time + (2.5 * np.max(lc_odd_zoom.time))))
        concat_view_f = np.concatenate((lc_odd_zoom.flux, lc_even_zoom.flux))
        concat_lc = lk.LightCurve(concat_view_t, concat_view_f)
        concat_lc = (concat_lc / np.abs(concat_lc.flux.min()) ) * 2.0 + 1
        return concat_lc

def split_train_lcs(num_workers, lcs, output_dir, basename = 'train'):
        splits = []
        split_size = int(math.ceil(len(lcs) / num_workers))
        #
        # Split them into positive and negative examples
        #
        #print(tic_catalog.index)
        positives, negatives = [], []
        for lc in lcs:
                tess_id = file2tic['tic_id'][lc]
                if not tess_id in tic_catalog.index:
                        continue
                if tic_catalog['Disposition'][tess_id] in ['KP', 'PC', 'CP']:
                        positives.append(lc)
                else:
                        negatives.append(lc)

        num_positive_workers = int(math.ceil(len(positives) / len(lcs) * num_workers))
        split_size = int(math.ceil(len(positives) / num_positive_workers))
        print(f'Split size: {split_size}, positives = {len(positives)}, workers = {num_positive_workers}')
        for i in range(num_positive_workers):
                output_file = "{}-{}-part-{:05d}-of-{:05d}.tfRecords".format(basename, 'positive', i, num_workers)
                splits.append((positives[i * split_size: (i + 1) * split_size], output_dir, output_file))
        num_negative_workers = int(math.ceil(len(negatives) / len(lcs) * num_workers))
        split_size = int(math.ceil(len(negatives) / num_negative_workers))
        print(f'Split size: {split_size}, negatives = {len(negatives)}, workers = {num_negative_workers}')
        for i in range(num_negative_workers):
                output_file = "{}-{}-part-{:05d}-of-{:05d}.tfRecords".format(basename, 'negative', i, num_workers)
                splits.append((negatives[i * split_size: (i + 1) * split_size], output_dir, output_file))

        # print(splits)
        return splits

def create_train_test(lcs):
        random.seed(318)
        random.shuffle(lcs)
        split_1 = int(0.9 * len(lcs))
        train_lcs = lcs[:split_1]
        test_lcs = lcs[split_1:]    
        print(f'train: {len(train_lcs)}, test: {len(test_lcs)}')
        return train_lcs, test_lcs

def create_train_test_val(lcs):
        random.seed(318)
        random.shuffle(lcs)
        split_1 = int(0.8 * len(lcs))
        split_2 = int(0.9 * len(lcs))
        train_lcs = lcs[:split_1]
        test_lcs = lcs[split_1:split_2] 
        val_lcs = lcs[split_2:] 
        print(f'train: {len(train_lcs)}, test: {len(test_lcs)}, val: {len(val_lcs)}')
        return train_lcs, test_lcs, val_lcs
############################################################################################################

def write_records(lcs, output_dir, output_file, file2tic):
    output_path = os.path.join(output_dir, output_file)
    os.makedirs(output_dir, exist_ok=True)
    print(f'📤 Writing out to: {output_path}')

    count = 0  # number of PC-disposition entries

    with tf.io.TFRecordWriter(output_path) as writer:
        for lc in tqdm(lcs, desc="Writing TFRecords"):
            try:
                tess_id = file2tic['tic_id'][lc]
            except KeyError:
                print(f"⚠️  Could not find TIC ID for {lc}")
                continue

            lc_raw = load_lightcurve(os.path.join(_BASE_PATH, lc))
            if lc_raw is None:
                print(f'⚠️  Could not load lightcurve for {tess_id}')
                continue

            lc_global, lc_local, lc_global_shifted = _process_lightcurve(tess_id, lc_raw, tic_catalog)
            if lc_local is None:
                continue

            try:
                lc_phase = build_halfphase_views(tess_id, lc_raw)
            except ValueError:
                print(f'⚠️  Skipping {tess_id}: unable to build secondary view')
                continue

            if lc_phase is None:
                continue

            if tic_catalog['Disposition'][tess_id] == 'PC':
                count += 1

            metadata = tic_catalog.loc[tess_id]

            ex = tf.train.Example()
            _set_int64_feature(ex, 'TIC_ID', [tess_id])
            _set_float_feature(ex, 'global view', lc_global.flux.astype(float))
            _set_float_feature(ex, 'local view', lc_local.flux.astype(float))
            _set_float_feature(ex, 'shifted global view', lc_global_shifted.flux.astype(float))
            _set_float_feature(ex, 'odd_even view', lc_phase.flux.astype(float))
            _set_bytes_feature(ex, 'Disposition', [metadata['Disposition']])

            for k, v in metadata.items():
                if k == 'Disposition':
                    continue
                _set_float_feature(ex, k, [v])

            writer.write(ex.SerializeToString())

    print(f'✅ Finished: {len(lcs)} examples written, {count} PCs included.')


def do_work(basename, output_dir):
    print("🛠️ Starting TFRecord generation (single-process)...")

    # Only train and test splits
    train_lcs, test_lcs = create_train_test(list(file2tic.index))

    num_splits = 8
    splits = split_train_lcs(num_splits, train_lcs, os.path.join(output_dir, 'train'), basename)

    # Add test split
    splits.append((test_lcs, os.path.join(output_dir, 'test'), f'{basename}-test.tfRecords'))

    for lcs, out_dir, out_file in splits:
        try:
            write_records(lcs, out_dir, out_file, file2tic)
        except Exception as e:
            print(f"❌ Error processing split {out_file}: {e}")

    print("✅ TFRecord generation complete.")


##################################################################################################################33

if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Script for generating records for training purposes.")
        parser.add_argument("--catalog", type=str, required=True, help="TIC Catalog (such as, period_info-dl3.csv)")
        parser.add_argument("--tic2fileDB", type=str, required=True, help="File that maps TIC ID to associated .fits file")
        parser.add_argument("--input", type=str, required=True, help="Input folder where the .fits file are")
        parser.add_argument("--output", type=str, required=True, help="Output folder where the .tfRecords will be saved")
        parser.add_argument("--basename", type=str, required=True, help="Base name for .tfRecords (e.g., toi)")
        parser.add_argument("--exclude", type=str, default="", help="TIC2FileDB that lists TICs that should be excluded")
        args = parser.parse_args()

        _BASE_PATH = args.input
        tic2file, file2tic = load_tic2file(args.tic2fileDB)
        print(tic2file.head(),file2tic.head())
        # We have decided not to do imputation
        
        tic_catalog = load_catalog(args.catalog, enableImputation=False)
        print(tic_catalog.head())
        do_work(args.basename, args.output)