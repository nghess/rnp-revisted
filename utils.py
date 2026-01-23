import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from kilosort import run_kilosort
from kilosort.io import save_preprocessing, load_ops

# ============================================================================
# FILE I/O AND PATH UTILITIES
# ============================================================================

def natural_sort_key(path):
    """
    Create a key for sorting strings with numbers naturally.
    Converts "1", "2", "10" into proper numeric order instead of lexicographic order.
    """
    # Convert path to string if it's a Path object
    path_str = str(path)
    # Split string into chunks of numeric and non-numeric parts
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', path_str)]

def get_file_paths(directory: str = '', extension: str = '', keyword: str = '', session_type: str = '', keyword_exact=False, keyword_bool=False, not_keyword='', print_paths=False, print_n=np.inf) -> list:
    """
    Get all files matching extension and keyword in a directory and its subdirectories.

    Parameters:
    -----------
    directory : str
        Directory to search in
    extension : str
        File extension to match (e.g., 'npy', 'csv')
    keyword : str
        Keyword to match in filename
    session_type : str
        Session type prefix to filter by (e.g., 'm' for mouse sessions)
    keyword_exact : bool
        If True, keyword must match filename exactly
    keyword_bool : bool
        If True, apply not_keyword filter
    not_keyword : str
        Keyword to exclude from results
    print_paths : bool
        If True, print found paths
    print_n : int
        Number of paths to print (if print_paths=True)

    Returns:
    --------
    list
        List of Path objects matching the criteria, sorted naturally
    """
    if keyword_bool:
        if keyword_exact:
            paths = [f for f in Path(directory).glob(f'**/{session_type}*/{keyword}.{extension}') if not_keyword not in str(f)]
        else:
            paths = [f for f in Path(directory).glob(f'**/{session_type}*/*.{extension}') if keyword in f.name and not_keyword not in str(f)]
    else:
        if keyword_exact:
            paths = [f for f in Path(directory).glob(f'**/{session_type}*/{keyword}.{extension}')]
        else:
            paths = [f for f in Path(directory).glob(f'**/{session_type}*/*.{extension}') if keyword in f.name]
    # Sort paths using natural sorting
    paths = sorted(paths, key=natural_sort_key)
    print(f'Found {len(paths)} {keyword}.{extension} files')
    if print_paths:
            show_paths(paths, print_n)
    return paths

def show_paths(data_paths, print_n=np.inf):
    """
    Print collected paths and their indices.

    Parameters:
    -----------
    data_paths : list
        List of paths to print
    print_n : int
        Maximum number of paths to print
    """
    for ii, path in enumerate(data_paths[:min(print_n, len(data_paths))]):
        print(f"{ii} {path}")

def filter_paths(paths_to_filter, paths_to_reference):
    """
    Paths are valid only if their final two parent directories (mouse and session ids)
    match the reference paths.

    Parameters:
    -----------
    paths_to_filter : list
        Paths to filter
    paths_to_reference : list
        Reference paths to match against

    Returns:
    --------
    list
        Filtered paths
    """
    filtered_paths = []
    for path in paths_to_filter:
        for ref_path in paths_to_reference:
            if (path.parents[1].name == ref_path.parents[1].name and
                path.parents[0].name == ref_path.parents[0].name):
                filtered_paths.append(path)
                break
    return filtered_paths

def filter_paths_numeric(paths, session_type: str, min_value):
    """
    Filter paths by checking if session number meets minimum threshold.

    Parameters:
    -----------
    paths : list
        Paths to filter
    session_type : str
        Session type prefix (e.g., 'm' for mouse)
    min_value : int
        Minimum session number to include

    Returns:
    --------
    list
        Filtered paths
    """
    return [
        p for p in paths
        if not any(
            part.startswith(session_type) and part[1:].isdigit() and int(part[1:]) < min_value
            for part in p.parts
        )
    ]

def filter_paths_by_session_id(data_paths, session_id_threshold: int, part=-2, greater=True):
    """
    Filter paths by checking if session id is greater/less than a given threshold.

    Parameters:
    -----------
    data_paths : list
        Paths to filter
    session_id_threshold : int
        Threshold value for comparison
    part : int
        Index of path part containing session_id (default: -2)
    greater : bool
        If True, keep sessions > threshold; if False, keep sessions < threshold

    Returns:
    --------
    list
        Filtered paths
    """
    filtered_paths = []
    for path in data_paths:
        session_id = path.parts[part] # Extract session_id
        numeric_value = int(re.search(r'\d+', session_id).group()) # Extract numeric part of session_id
        if greater:
            if int(numeric_value) > session_id_threshold:
                filtered_paths.append(path)
        else:
            if int(numeric_value) < session_id_threshold:
                filtered_paths.append(path)
    return filtered_paths

def get_savedirs(path):
    """
    Get the name of the last two directories in data path.

    Parameters:
    -----------
    path : str or Path
        Path to extract directories from

    Returns:
    --------
    str
        Last two directories joined by os separator
    """
    path = str(path)
    parts = path.split(os.path.sep)
    return os.path.sep.join(parts[-3:-1])

def get_ids_from_path(data_paths, part=3):
    """
    Function to extract session id and mouse id from Open Ephys save directory name.

    Parameters:
    -----------
    data_paths : list
        List of Path objects
    part : int
        Index of path part containing the Open Ephys save directory

    Returns:
    --------
    tuple
        (session_ids, mouse_ids) - lists of extracted IDs
    """
    session_ids = []
    mouse_ids = []

    for data_path in data_paths:
        open_ephys_savedir = data_path.parts[part] # Open Ephys save path is in index 3 of full path
        session_id = open_ephys_savedir.split('_')[0]  # Session ID is before the first underscore
        mouse_id = open_ephys_savedir.split('_')[-1]  # Mouse ID is after the last underscore
        session_ids.append(session_id)
        mouse_ids.append(mouse_id)

    return session_ids, mouse_ids


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def load_ephys(ephys_file: str, num_samples: int = -1, start = 0, stop = 0, nchannels: int = 32, dtype = np.uint32, order = 'C') -> np.array:
    """
    Load and reshape binary electrophysiology data into a NumPy array.

    Parameters:
    -----------
    ephys_file : str
        Path to the binary file containing electrophysiology data
    num_samples : int
        Number of samples to read from the file
    start : int
        Number of seconds to trim from beginning (default: 0)
    stop : int
        Number of seconds to trim from end (default: 0)
    nchannels : int
        Number of channels in the ephys data (default: 32)
    dtype : numpy dtype
        Data type of binary file (default: np.uint32)
    order : str
        Reshaping order - 'C' or 'F' (default: 'C')

    Returns:
    --------
    np.array
        A 2D NumPy array of the electrophysiology data, reshaped into
        (nchannels, number_of_samples_per_channel)
    """

    # reading in binary data
    num_samples = num_samples * nchannels
    ephys_bin = np.fromfile(ephys_file, dtype=dtype, count = num_samples)

    # ensuring equal samples from each channel
    num_complete_sets = len(ephys_bin) // nchannels
    ephys_bin = ephys_bin[:num_complete_sets * nchannels]

    # reshape 1d array into nchannels x num_samples NumPy array
    ephys_data = np.reshape(ephys_bin, (nchannels, -1), order=order)

    # removing start seconds from beggining, and stop from end of signal; default is 0
    start = start * 30000
    stop = stop * 30000

    if stop == 0:
        ephys = ephys_data[:, start:]
    else:
        ephys = ephys_data[:, start: -stop]

    return ephys

def clip_outliers_with_window(data: np.ndarray, clip_mult: float = 2, window_size: int = 30000, overlap: int = 10000) -> np.ndarray:
    """
    Clips outlier values in neural data with a ±1 sample window around detected outliers.

    Parameters:
    -----------
    data : np.ndarray
        Data array of shape (n_channels, n_samples)
    clip_mult : float
        Multiplier for min/max thresholds (default: 2)
    window_size : int
        Size of sliding window for min/max calculation (default: 30000)
    overlap : int
        Overlap between windows (default: 10000)

    Returns:
    --------
    np.ndarray
        Processed data array with outliers set to zero
    """
    # Calculate number of windows
    num_windows = (data.shape[1] - window_size) // (window_size - overlap) + 1
    min_vals = np.zeros((data.shape[0], num_windows))
    max_vals = np.zeros((data.shape[0], num_windows))

    # Process each channel separately to get min/max values
    for ch in range(data.shape[0]):
        for i in range(num_windows):
            start = i * (window_size - overlap)
            end = start + window_size
            min_vals[ch,i] = np.min(data[ch,start:end])
            max_vals[ch,i] = np.max(data[ch,start:end])

    # Get mean of min and max values per channel
    avg_min_vals = np.mean(min_vals, axis=1)
    avg_max_vals = np.mean(max_vals, axis=1)

    # Apply clipping thresholds per channel
    for ch in range(data.shape[0]):
        # Create boolean masks for outlier points
        upper_outliers = data[ch,:] > clip_mult*avg_max_vals[ch]
        lower_outliers = data[ch,:] < clip_mult*avg_min_vals[ch]

        # Combine outlier masks
        outliers = upper_outliers | lower_outliers

        # Create shifted masks for ±1 window
        outliers_shifted_left = np.roll(outliers, 1)
        outliers_shifted_right = np.roll(outliers, -1)

        # Combine all masks to include the window
        final_mask = outliers | outliers_shifted_left | outliers_shifted_right

        # Set values to zero where mask is True
        data[ch, final_mask] = 0

    return data


# ============================================================================
# TTL PROCESSING
# ============================================================================

def ttl_bool(data_path: str, results_path: str, sample_hz=30000, resample_hz=1000, save=True):
    """
    Process TTL signal to boolean array.

    Parameters:
    -----------
    data_path : str
        Path to TTL .npy file
    results_path : str
        Path to save processed TTL
    sample_hz : int
        Original sampling rate (default: 30000)
    resample_hz : int
        Target sampling rate (default: 1000)
    save : bool
        If True, save to results_path

    Returns:
    --------
    np.ndarray
        Boolean TTL signal
    """
    data = np.load(data_path)

    #Resample data to 1000 Hz
    ttl_resample = data[::sample_hz//resample_hz]

    # Normalize to 0-1 range
    normalized = (ttl_resample - np.min(ttl_resample)) / (np.max(ttl_resample) - np.min(ttl_resample))

    # Rebuild ttl signal as boolean
    ttl_bool = ttl_resample > -30000

    if save:
        np.save(results_path, ttl_bool)
    return ttl_bool

def clean_camera_ttl(signal, threshold=-30000, min_frame_duration=10, min_frame_spacing=10, echo=False):
    """
    Clean camera TTL signals to extract valid frame pulses.

    Parameters:
    -----------
    signal : np.ndarray
        Raw TTL signal
    threshold : int
        Voltage threshold for detecting pulses (default: -30000)
    min_frame_duration : int
        Minimum pulse duration in samples (default: 10)
    min_frame_spacing : int
        Minimum spacing between pulses in samples (default: 10)
    echo : bool
        If True, print diagnostic information (default: False)

    Returns:
    --------
    np.ndarray
        Cleaned binary TTL signal
    """
    # Initial threshold
    binary = (signal < threshold).astype(int)
    if echo:
        print("Number of samples below threshold:", np.sum(binary))

    # Find potential frame boundaries
    transitions = np.diff(binary)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    if echo:
        print("Number of starts found:", len(starts))
        print("Number of ends found:", len(ends))
        if len(starts) > 0:
            print("First few start indices:", starts[:5])
        if len(ends) > 0:
            print("First few end indices:", ends[:5])

    # Ensure we have matching starts and ends
    if len(starts) == 0 or len(ends) == 0:
        if echo:
            print("No valid transitions found")
        return np.zeros_like(signal, dtype=int)

    if ends[0] < starts[0]:
        ends = ends[1:]
    if len(starts) > len(ends):
        starts = starts[:-1]

    # Filter based on duration and spacing
    valid_frames = []
    last_valid_end = -min_frame_spacing

    for start, end in zip(starts, ends):
        duration = end - start
        spacing = start - last_valid_end

        if duration >= min_frame_duration and spacing >= min_frame_spacing:
            valid_frames.append((start, end))
            last_valid_end = end

    # Create cleaned signal
    cleaned = np.zeros_like(signal, dtype=int)
    for start, end in valid_frames:
        cleaned[start:end] = 1

    return cleaned

def analyze_ttl_timing(signal, threshold=-25000):
    """
    Analyze TTL pulse timing and frame rates.

    Parameters:
    -----------
    signal : np.ndarray
        TTL signal to analyze
    threshold : int
        Voltage threshold for detecting pulses (default: -25000)
    """
    binary = (signal < threshold).astype(int)
    transitions = np.diff(binary)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    if len(starts) > 0 and len(ends) > 0:
        # Make sure we have matching pairs
        if ends[0] < starts[0]:
            ends = ends[1:]
        if len(starts) > len(ends):
            starts = starts[:-1]

        # Now calculate durations and spacings
        durations = ends - starts  # How long the signal is "on"
        spacings = starts[1:] - ends[:-1]  # Time between pulses

        print(f"Average pulse duration: {np.mean(durations):.2f} samples ({np.mean(durations)/1000*1000:.2f} ms)")
        print(f"Average spacing between pulses: {np.mean(spacings):.2f} samples ({np.mean(spacings)/1000*1000:.2f} ms)")
        print(f"Frame rate: {1000/np.mean(starts[1:] - starts[:-1]):.2f} fps")

        # Additional diagnostic info
        print(f"\nNumber of pulses analyzed: {len(durations)}")
        print(f"Duration range: {np.min(durations):.2f} to {np.max(durations):.2f} samples")
        print(f"Spacing range: {np.min(spacings):.2f} to {np.max(spacings):.2f} samples")

def process_ttl(ttl_path, ttl_floor=-32000, min_frame_duration=6, min_frame_spacing=6):
    """
    Process TTL signal from file - wrapper function for clean_camera_ttl.

    Parameters:
    -----------
    ttl_path : str or Path
        Path to TTL .npy file
    ttl_floor : int
        Voltage threshold for detecting pulses (default: -32000)
    min_frame_duration : int
        Minimum pulse duration in samples (default: 6)
    min_frame_spacing : int
        Minimum spacing between pulses in samples (default: 6)

    Returns:
    --------
    np.ndarray
        Cleaned boolean TTL signal
    """
    sample_hz=30000
    resample_hz=1000

    data = np.load(ttl_path)
    ttl_resample = data[::sample_hz//resample_hz]

    ttl_bool = clean_camera_ttl(ttl_resample,
                           threshold=ttl_floor,
                           min_frame_duration=min_frame_duration,
                           min_frame_spacing=min_frame_spacing)

    return ttl_bool

def crop_ttl(ttl_bool):
    """
    Crop TTL signal to valid transitions (remove start/end padding).

    Parameters:
    -----------
    ttl_bool : np.ndarray
        Boolean TTL signal

    Returns:
    --------
    tuple
        (ttl_crop, valid_transitions) - cropped TTL and transition indices
    """
    # Find all low-to-high transitions
    valid_transitions = np.where(np.diff(ttl_bool) == 1)[0]
    # Calculate duration between transitions by taking difference
    transition_durations = np.diff(valid_transitions)
    # Add the last duration by duplicating the final duration
    transition_durations = np.append(transition_durations, transition_durations[-1])
    # Keep only transitions that are at least 10 samples apart
    valid_transitions = valid_transitions[transition_durations >= 10]
    # Apply crop to ttl_bool
    ttl_crop = ttl_bool[valid_transitions[0]:valid_transitions[-1]]

    return ttl_crop, valid_transitions


# ============================================================================
# TIMESTAMP ALIGNMENT
# ============================================================================

def open_ephys_start_time(timestamps_df):
    """
    Get Open Ephys start time from sync_messages.txt file.

    Parameters:
    -----------
    timestamps_df : str or Path
        Path to sync_messages.txt file containing datetime string

    Returns:
    --------
    int
        Start timestamp in milliseconds
    """
    # Open text file and read first line
    with open(timestamps_df, 'r') as f:
        first_line = f.readline().strip()
    # Extract timestamp using regex (string of digits >= 12 characters)
    match = re.search(r'(\d{12,})', first_line)
    if match:
        timestamp_str = match.group(1)
        return int(timestamp_str)
    else:
        raise ValueError("No valid timestamp found in sync_messages.txt")

def timestamp_to_ms(timestamp_str):
    """
    Convert timestamp string to milliseconds since Unix epoch (Jan 1, 1970 UTC).

    Parameters:
    -----------
    timestamp_str : str
        Timestamp in format 'YYYY-MM-DD HH:MM:SS.fffffffff'

    Returns:
    --------
    int
        Milliseconds since midnight Jan 1, 1970 UTC
    """
    # Truncate to microseconds (6 decimal places) if longer
    if '.' in timestamp_str:
        date_part, frac_part = timestamp_str.split('.')
        frac_part = frac_part[:6]  # Keep only first 6 digits
        timestamp_str = f"{date_part}.{frac_part}"

    dt = datetime.fromisoformat(timestamp_str)
    return int(dt.timestamp() * 1000)

def align_timestamps(event_data, open_ephys_start_ms):
    """
    Align event timestamps to Open Ephys start time.

    Parameters:
    -----------
    event_data : DataFrame
        DataFrame containing 'timestamp' column with datetime strings
    open_ephys_start_ms : int
        Open Ephys start time in milliseconds since Unix epoch

    Returns:
    --------
    DataFrame
        Event data with aligned 'timestamp_ms' column in milliseconds since Open Ephys recording start
    """
    # Convert event timestamps to milliseconds since Unix epoch
    event_data['timestamp_ms'] = event_data['timestamp'].apply(timestamp_to_ms)

    # Align to Open Ephys start time
    event_data['timestamp_ms'] = event_data['timestamp_ms'] - open_ephys_start_ms

    return event_data


# ============================================================================
# BEHAVIORAL DATA PROCESSING
# ============================================================================

def process_events(idx, event_paths_a, event_paths_b, columns: dict):
    """
    Process event data from paired CSV files.

    Parameters:
    -----------
    idx : int
        Index of the session to process
    event_paths_a : list
        List of paths to eventsA CSV files
    event_paths_b : list
        List of paths to eventsB CSV files
    columns : dict
        Dictionary of variable names and datatypes

    Returns:
    --------
    DataFrame
        DataFrame containing processed event data
    """
    # Load events .csv part A (This .csv is always 7 cols wide, but we don't hardcode in case Bonsai WriteCsv improves.)
    event_data_a = pd.read_csv(event_paths_a[idx])
    col_names_a = list(columns.keys())[:len(event_data_a.columns)]
    event_data_a.columns = col_names_a
    pd.to_datetime(event_data_a['timestamp'])

    # Load events .csv part B (col number varies by experimental condition, so we need to load in col names flexibly.)
    event_data_b = pd.read_csv(event_paths_b[idx])
    col_names_b = list(columns.keys())[len(event_data_a.columns):]
    event_data_b.columns = col_names_b

    # Concatenate eventsA and eventsB dataframes
    if len(event_data_a) == len(event_data_b):
        event_data = pd.concat([event_data_a, event_data_b], axis=1)
    else:
        print("Event dataframes must contain same number of rows")
        min_length = min(len(event_data_a), len(event_data_b))
        max_length = max(len(event_data_a), len(event_data_b))
        print(f"Trimmed long dataframe by {max_length-min_length} rows.")
        event_data_a = event_data_a.iloc[:min_length]
        event_data_b = event_data_b.iloc[:min_length]
        event_data = pd.concat([event_data_a, event_data_b], axis=1)

    # Set columns to appropriate types as specified in columns dictionary
    event_data = event_data.astype(columns)

    # Calculate speed and direction columns
    event_data['speed'] = calculate_speed(event_data)
    event_data['direction'] = calculate_direction(event_data)  # Check that this is working

    # Rebuild 'reward_state' as periods between click onset and poke events
    # reward_state = True from when 'click' goes False->True until either 'poke_left' or 'poke_right' goes False->True
    reward_state = np.zeros(len(event_data), dtype=bool)

    # Find click transitions from False to True
    click_onsets = np.where((event_data['click'].shift(1) == False) &
                           (event_data['click'] == True))[0]

    # Find poke transitions from False to True (either left or right)
    poke_left_onsets = np.where((event_data['poke_left'].shift(1) == False) &
                               (event_data['poke_left'] == True))[0]
    poke_right_onsets = np.where((event_data['poke_right'].shift(1) == False) &
                                (event_data['poke_right'] == True))[0]

    # Combine and sort all poke onsets
    all_poke_onsets = np.sort(np.concatenate([poke_left_onsets, poke_right_onsets]))

    # For each click onset, find the next poke onset and mark the period as reward_state
    for click_idx in click_onsets:
        # Find the next poke onset after this click onset
        next_pokes = all_poke_onsets[all_poke_onsets > click_idx]
        if len(next_pokes) > 0:
            poke_idx = next_pokes[0]
            # Mark reward_state period from click onset to poke onset
            reward_state[click_idx:poke_idx] = True

    event_data['reward_state'] = reward_state

    # Rename centroids so that they don't get confused with SLEAP centroid
    event_data = event_data.rename(columns={'centroid_x' : 'bonsai_centroid_x'})
    event_data = event_data.rename(columns={'centroid_y' : 'bonsai_centroid_y'})

    # Add 'drinking' column (period between reward initiation poke and start of ITI)
    event_data['drinking'] = calculate_drinking(event_data)

    return event_data

def calculate_speed(data):
    """
    Calculate speed from x,y coordinates.

    Parameters:
    -----------
    data : DataFrame
        DataFrame containing 'centroid_x' and 'centroid_y' columns

    Returns:
    --------
    np.ndarray
        Speed values (0 prepended to match data length)
    """
    dx = np.diff(data['centroid_x'])
    dy = np.diff(data['centroid_y'])
    speed = np.sqrt(dx**2 + dy**2)
    # Add 0 at start to match length
    return np.concatenate(([0], speed))

def calculate_direction(data):
    """
    Calculate movement direction in radians from x,y coordinates.

    Parameters:
    -----------
    data : DataFrame
        DataFrame containing 'centroid_x' and 'centroid_y' columns

    Returns:
    --------
    np.ndarray
        Direction values in radians (0 prepended to match data length)
    """
    dx = np.diff(data['centroid_x'])
    dy = np.diff(data['centroid_y'])
    direction = np.arctan2(dy, dx)
    # Add 0 at start to match length
    return np.concatenate(([0], direction))

def calculate_drinking(data):
    """
    Calculate drinking periods between reward_state True->False and ITI False->True.

    Parameters:
    -----------
    data : DataFrame
        DataFrame containing 'reward_state' and 'iti' columns

    Returns:
    --------
    np.ndarray
        Boolean array indicating drinking periods
    """
    drinking = np.zeros(len(data), dtype=bool)

    # Find reward_state transitions from True to False
    reward_transitions = np.where((data['reward_state'].shift(1) == True) &
                                 (data['reward_state'] == False))[0]

    # Find iti transitions from False to True
    iti_transitions = np.where((data['iti'].shift(1) == False) &
                              (data['iti'] == True))[0]

    # For each reward_state True->False transition, find the next iti False->True transition
    for reward_idx in reward_transitions:
        # Find the next iti transition after this reward transition
        next_iti = iti_transitions[iti_transitions > reward_idx]
        if len(next_iti) > 0:
            iti_idx = next_iti[0]
            # Mark drinking period from reward transition to iti transition
            drinking[reward_idx:iti_idx] = True

    return drinking


# ============================================================================
# KILOSORT SPIKE SORTING
# ============================================================================

def kilosort(data_path: str, results_path: str, probe_path: str = 'probe_maps/8_tetrode_2_region_20um.mat', num_channels: int = 32, settings: dict = {}, save_preprocessed: bool = True, clean_outliers: bool = True):
    """
    Run kilosort4. Use settings dictionary to change kilosort settings for the run.

    Parameters:
    -----------
    data_path : str
        Path to neural data file (.npy or .bin)
    results_path : str
        Path to save Kilosort results
    probe_path : str
        Path to probe configuration file (default: 'probe_maps/8_tetrode_2_region_20um.mat')
    num_channels : int
        Number of channels in data (default: 32)
    settings : dict
        Kilosort settings dictionary (default: {})
    save_preprocessed : bool
        If True, save preprocessed data (default: True)
    clean_outliers : bool
        If True, clip outliers before sorting (default: True)

    Returns:
    --------
    tuple or None
        Kilosort results (ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes)
        or None if sorting failed
    """
    # Initialize paths
    data_path = Path(data_path)
    results_path = Path(results_path)

    # Multiplier for min/max thresholds. Values outside these ranges will be set to zero. Only applies if clean_outliers = True.
    clip_mult = 3

    # Handle numpy files by temporarily converting to .bin format
    if data_path.suffix == '.npy':
        # Load .npy file and save as binary
        data = np.load(data_path)
        print(f"{data_path.parent}")
        print(f"Data import shape:{data.shape}")
        data_min = data.min()
        data_max = data.max()
        data_std = data.std()

        # Apply outlier clipping
        if clean_outliers:
            data = clip_outliers_with_window(data, clip_mult)

        data = data.reshape(-1, order = 'F')
        temp_bin_path = data_path.parent / 'temp.bin'
        data.tofile(temp_bin_path)
        print(f"Created temporary binary file: {temp_bin_path}")

        # Create temporary binary file in data parent directory
        data_path = data_path.parent / 'temp.bin'

    else:
        data = np.load(data_path)
        if clean_outliers:
            data = clip_outliers_with_window(data, clip_mult)

    # Create results directory if it doesn't exist
    results_path.mkdir(parents=True, exist_ok=True)

    # Define Kilosort4 settings for the current run

    settings = settings  #{'data_dir': data_path.parent, 'n_chan_bin': num_channels, 'Th_universal': 10, 'Th_learned': 9, 'nblocks': 0, 'drift_smoothing': [0, 0, 0], 'dminx': 20, 'artifact_threshold': np.inf, 'batch_size': 60000}
    settings['data_dir'] = Path(data_path).parent

    # Run Kilosort 4
    try:
        ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
            run_kilosort(
                settings=settings,
                probe_name=Path.cwd() / probe_path,
                save_preprocessed_copy=save_preprocessed,
                do_CAR= False,
                results_dir=results_path,
                data_dtype = 'int16',
                #verbose_console=True,
                #verbose_log=True
                )

        # Delete temporary binary file from drive if it exists
        temp_bin_path = data_path.parent / 'temp.bin'
        if temp_bin_path.exists():
            temp_bin_path.unlink()

        # Write to 'good' units summary
        unit_summary(data_path, results_path, data_min, data_max, data_std, clip_mult, error=False)

        # Return results
        return ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes

    except:
        # Write error to log
        unit_summary(data_path, results_path, data_min, data_max, data_std, clip_mult, error=True)
        return None

def reset_dat_path_in_params(file_path, new_dat_path='temp_wh.dat'):
    """
    Clears and resets dat_path in params.py file to local temp_wh.dat.

    Parameters:
    -----------
    file_path : str or Path
        Path to params.py file
    new_dat_path : str
        New dat_path value (default: 'temp_wh.dat')

    Returns:
    --------
    bool
        True if successful
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match dat_path assignment
    pattern = r'(dat_path\s*=\s*)["\'].*?["\']'
    updated_content = re.sub(pattern, f'\\1"{new_dat_path}"', content)

    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(updated_content)

    return True

def unit_summary(data_path, results_path, data_min, data_max, data_std, clip_mult, error=False):
    """
    Grab the number of single units found from kilosort.log and append them to a summary txt file.

    Parameters:
    -----------
    data_path : str or Path
        Path to input data
    results_path : Path
        Path to Kilosort results directory
    data_min : float
        Minimum value in data
    data_max : float
        Maximum value in data
    data_std : float
        Standard deviation of data
    clip_mult : float
        Clipping multiplier used
    error : bool
        If True, log as error (default: False)
    """

    mouse_session = get_savedirs(data_path)
    savedir = results_path.parents[1]

    log_file = savedir / mouse_session / "kilosort4.log"
    output_file = savedir / "good_units.txt"

    with open(log_file, 'r') as file:
        content = file.read()

    # Use regex to find the number before "units"
    pattern = r'(\d{1,3}) units found with good refractory periods'
    match = re.search(pattern, content)

    if match and not error:
        # Extract the number from the first capture group
        num_units = match.group(1)

        # Append the number to the output file
        with open(output_file, 'a') as outfile:
            outfile.write(f"{mouse_session} - {num_units} units - min: {data_min} max: {data_max} std: {round(data_std, 3)}, clip_mult: {clip_mult}\n")
    elif error:
        with open(output_file, 'a') as outfile:
            outfile.write(f"{mouse_session} - Kilosort failed - min: {data_min} max: {data_max} std: {round(data_std, 3)}, clip_mult: {clip_mult}\n")
    else:
        with open(output_file, 'a') as outfile:
            outfile.write(f"{mouse_session} - No matching pattern found in the log file\n")

    print(f"Summary written to {output_file}")
