"""
Receiver Gather (version 1.6-1) bindings to ObsPy core module.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from future.builtins import *  # NOQA

from collections import namedtuple

import numpy as np

from obspy.core import Stream, Trace, Stats
from segd_read_util import _read, _open_file, _quick_merge, MyStats

HeaderCount = namedtuple('HeaderCount', 'general channel_set extended external')


@_open_file
def _read_segd(filename, headonly=False, starttime=None, endtime=None,
               merge=False, contacts_north=False, details=False, **kwargs):
    """
    Read Seg-D File Format version 2.0

    .. warning::
        This function should NOT be called directly, it registers via the
        ObsPy :func:`~obspy.core.stream.read` function, call this instead.

    :param filename: path to the sege file or a file object.
    :type filename: str, buffer
    :param headonly: If True don't read data, only main information
        contained in the headers of the trace block is read.
    :type headonly: optional, bool
    :param starttime: If not None dont read traces that start before starttime.
    :type starttime: optional, :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param endtime: If not None dont read traces that start after endtime.
    :type endtime: optional, :class:`~obspy.core.utcdatetime.UTCDateTime`
    :param merge: If True merge contiguous data blocks as they are found. For
        continuous data files having 100,000+ traces this will create
        more manageable streams.
    :type merge: bool
    :param contacts_north: If this parameter is set to True, it will map the
        components to Z (1C, 3C), N (3C), and E (3C) as well as correct
        the polarity for the vertical component.
    :type contacts_north: bool
    :param details: If True, all the information contained in the headers
        is read).
    :type details: bool
    :return: An ObsPy :class:`~obspy.core.stream.Stream` object.
        Frequencies are expressed in hertz and time is expressed in second
        (except for date).
    """
    # starttime = starttime or UTCDateTime(1970, 1, 1)
    # endtime = endtime or UTCDateTime()
    # get the number of headers/records, position of trace data
    # and record length.
    header_count = _cmp_nbr_headers(filename)
    record_count = _cmp_nbr_records(filename)

    trace_block_start = 32 * (2 + sum(header_count))
    record_length = _cmp_record_length(filename)

    # create trace data
    traces = []
    for i in range(0, record_count):
        nbr_bytes_trace_block = _cmp_jump(filename, trace_block_start)
        # trace_starttime = _read(filename, trace_block_start + 20 + 2 * 32, 8,
        #                         'binary') / 1e6
        # trace_endtime = trace_starttime + record_length
        # con1 = starttime.timestamp > trace_endtime
        # con2 = endtime.timestamp < trace_starttime
        # # determine if this block is excluded based in starttime/endtime params
        # if con1 or con2:
        #     # the block has to be excluded, increment trace_block_start
        #     #  and continue
        #     trace_block_start += nbr_bytes_trace_block
        #     continue
        trace = _make_trace(filename, trace_block_start, headonly, contacts_north, details)
        traces.append(trace)
        trace_block_start += nbr_bytes_trace_block
    if merge:
        traces = _quick_merge(traces)
    return Stream(traces=traces)


def _cmp_nbr_headers(fi):
    """
    Return a tuple containing the number of channel set headers,
    the number of extended headers and the number of external headers
    in the file.
    """
    header_count = HeaderCount(
        general=_read(fi, 11, 0.5, 'binary') - 1,
        channel_set=_read(fi, 28, 1, 'bcd'),
        extended=_read(fi, 30, 1, 'bcd'),
        external=_read(fi, 31, 1, 'bcd'),
    )
    return header_count


def _cmp_nbr_records(fi):
    """
    Return the number of records in the file (ie number of time slices
    multiplied by the number of components).
    """
    initial_header = _read_initial_headers(fi)
    nbr_records = initial_header['extended_headers']['total_nbr_record_trace']
    return nbr_records


def _cmp_record_length(fi):
    """
    Return the record length.
    """
    base_scan_interval = _read(fi, 22, 1, 'binary')
    sampling_rate = int(1000 / (base_scan_interval / 16))
    gen_head_2 = _read_initial_headers(fi)['general_header_2']
    record_length = gen_head_2['extended_record_length'] - 1 / sampling_rate
    return record_length


def _cmp_jump(fi, trace_block_start):
    """
    Return the number of bytes in a trace block.
    """
    nbr_trace_extension_block = _read(fi, trace_block_start + 9, 1, 'binary')
    nbr_bytes_header_trace = 20 + 32 * nbr_trace_extension_block
    nbr_sample_trace = _read(fi, trace_block_start + 27, 3, 'binary')
    nbr_bytes_trace_data = nbr_sample_trace * 4
    nbr_bytes_trace_block = nbr_bytes_trace_data + nbr_bytes_header_trace
    return nbr_bytes_trace_block


def _make_trace(fi, trace_block_start, headonly, standard_orientation,
                details):
    """
    Make obspy trace from a trace block (header + trace).
    """
    stats = _make_stats(fi, trace_block_start, details)
    if headonly:
        data = np.array([])
    else:  # read trace
        nbr_trace_extension_block = _read(fi, trace_block_start + 9, 1, 'binary')
        trace_start = trace_block_start + 20 + nbr_trace_extension_block * 32
        nbr_sample_trace = _read(fi, trace_block_start + 27, 3, 'binary')
        nbr_bytes_trace = 4 * nbr_sample_trace
        data = _read(fi, trace_start, nbr_bytes_trace, 'flt')
        # if stats.channel[-1] == 'Z':
        #     data = -data
        #     data = data.astype('>f4')
    return Trace(data=data, header=stats)


def _make_stats(fi, tr_block_start, details):
    """
    Make Stats object from information contained in the header of the trace.
    """
    nbr_tr_header_block = _read(fi, tr_block_start + 9, 1, 'binary')
    trace_headers = _read_trace_headers(fi, tr_block_start, nbr_tr_header_block)
    initial_headers = _read_initial_headers(fi)
    base_scan_interval = _read(fi, 22, 1, 'binary')
    sampling_rate = int(1000 / (base_scan_interval / 16))
    npts = _read(fi, tr_block_start + 27, 3, 'binary')  # number of samples per trace
    file_number = _read(fi, tr_block_start + 17, 3, 'binary') \
        if _read(fi, tr_block_start + 0, 2, 'binary') == 65535 \
        else _read(fi, tr_block_start + 0, 2, 'binary')
    rec_line_nbr = _read(fi, tr_block_start + 20, 3, 'binary')
    rec_point_nbr = _read(fi, tr_block_start + 23, 3, 'binary')
    sou_line_nbr = initial_headers['general_header_3']['source_line_number_integer']
    sou_point_nbr = initial_headers['general_header_3']['source_point_number_integer']

    trace_number = _read(fi, tr_block_start + 4, 2, 'binary')
    rec_point_index = _read(fi, tr_block_start + 26, 1, 'binary')
    stats_dict = dict(sampling_rate=sampling_rate,
                      npts=npts,
                      file_number=file_number,
                      sou_line_nbr=str(sou_line_nbr),
                      sou_point_nbr=str(sou_point_nbr),
                      sou_id=str(sou_line_nbr) + str(sou_point_nbr),
                      sou_x=initial_headers['extended_headers']['source_x'],
                      sou_y=initial_headers['extended_headers']['source_y'],
                      sou_elevation=initial_headers['extended_headers']['source_elevation'],
                      rec_line_nbr=str(rec_line_nbr),
                      rec_point_nbr=str(rec_point_nbr),
                      rec_id=str(rec_line_nbr) + str(rec_point_nbr),
                      rec_x=trace_headers['receiver_x'],
                      rec_y=trace_headers['receiver_y'],
                      rec_elevation=trace_headers['receiver_elevation'],
                      trace_number=trace_number,
                      rec_point_index=str(rec_point_index),
                      rec_resist_value=trace_headers['resist_value'],
                      nbr_stagnation_trace=initial_headers['extended_headers']['nbr_samples_in_record_trace'])

    if details:
        stats_dict['initial_headers'] = {}
        stats_initial_headers = stats_dict['initial_headers']
        stats_initial_headers.update(_read_initial_headers(fi))
        stats_dict['trace_headers'] = {}
        stats_tr_headers = stats_dict['trace_headers']
        stats_tr_headers.update(_read_trace_header(fi, tr_block_start))

        if nbr_tr_header_block > 0:
            stats_tr_headers.update(
                _read_trace_headers(fi, tr_block_start, nbr_tr_header_block))
    return stats_dict


def _read_trace_headers(fi, trace_block_start, nbr_trace_header):
    """
    Read headers in the trace block.
    """
    trace_headers = {}
    dict_func = {'1': _read_trace_header_1, '2': _read_trace_header_2,
                 '3': _read_trace_header_3, '4': _read_trace_header_4,
                 '5': _read_trace_header_5, '6': _read_trace_header_6,
                 '7': _read_trace_header_7, '8': _read_trace_header_8,
                 '9': _read_trace_header_9, '10': _read_trace_header_10}
    for i in range(1, nbr_trace_header + 1):
        trace_headers.update(dict_func[str(i)](fi, trace_block_start))
    return trace_headers


def _read_trace_header(fi, trace_block_start):
    """
    Read the 20 bytes trace header (first header in the trace block).
    """
    trace_edit_code = {0: 'unedited', 1: 'Static or stagnation before acquisition',
                       2: 'edit by acquisition system'}
    trace_edit_code_key = _read(fi, trace_block_start + 11, 1, 'binary')
    dict_header = dict(
        file_number=_read(fi, trace_block_start + 17, 3, 'binary')
        if _read(fi, trace_block_start + 0, 2, 'binary') == 65535
        else _read(fi, trace_block_start + 0, 2, 'binary'),
        trace_group_number=_read(fi, trace_block_start + 3, 1, 'bcd'),
        trace_number=_read(fi, trace_block_start + 4, 2, 'bcd'),
        extend_trace_head_number=_read(fi, trace_block_start + 9, 1, 'binary'),
        trace_edit_code=trace_edit_code[trace_edit_code_key]
    )

    return dict_header


def _read_trace_header_1(fi, trace_block_start):
    """
    Read trace header 1
    """
    pos = trace_block_start + 20

    dict_header_1 = dict(
        receive_point_index=_read(fi, pos + 6, 1, 'binary'),
        trace_sample_number=_read(fi, pos + 7, 4, 'binary'),
        extended_receiver_line_nbr=_read(fi, pos + 10, 5, 'binary'),
        extended_receiver_point_nbr=_read(fi, pos + 15, 5, 'binary'),
        sensor_type=_read(fi, pos + 20, 1, 'binary'),
        trace_count_file=_read(fi, pos + 21, 4, 'binary'),
    )
    return dict_header_1


def _read_trace_header_2(fi, trace_block_start):
    """
    Read trace header 2
    """
    pos = trace_block_start + 20 + 32

    dict_header_2 = dict(
        receiver_x=_read(fi, pos, 8, 'dbl'),
        receiver_y=_read(fi, pos + 8, 8, 'dbl'),
        receiver_elevation=_read(fi, pos + 16, 4, 'flt'),
        receiver_type_code=_read(fi, pos + 20, 1, 'binary'),
    )
    return dict_header_2


def _read_trace_header_3(fi, trace_block_start):
    """
    Read trace header 3
    """
    pos = trace_block_start + 20 + 32 * 2

    dict_header_3 = dict(
        resist_low_limit=_read(fi, pos, 4, 'flt'),
        resist_high_limit=_read(fi, pos + 4, 4, 'flt'),
        resist_value=_read(fi, pos + 8, 4, 'flt'),
        tilt_limit=_read(fi, pos + 12, 4, 'flt'),
        tilt_value=_read(fi, pos + 16, 4, 'flt'),
        resist_error=_read(fi, pos + 20, 1, 'binary'),
        tilt_error=_read(fi, pos + 21, 1, 'binary')
    )
    return dict_header_3


def _read_trace_header_4(fi, trace_block_start):
    """
    Read trace header 4
    """
    pos = trace_block_start + 20 + 32 * 3

    dict_header_4 = dict(
        capacitance_low_limit=_read(fi, pos, 4, 'flt'),
        capacitance_high_limit=_read(fi, pos + 4, 4, 'flt'),
        capacitance_value=_read(fi, pos + 8, 4, 'flt'),
        cut_off_low_limit=_read(fi, pos + 12, 4, 'flt'),
        cut_off_high_limit=_read(fi, pos + 16, 4, 'flt'),
        cut_off_value=_read(fi, pos + 20, 4, 'flt'),
        capacitance_error=_read(fi, pos + 24, 1, 'binary'),
        cut_off_error=_read(fi, pos + 25, 1, 'binary'),
    )
    return dict_header_4


def _read_trace_header_5(fi, trace_block_start):
    """
    Read trace header 5
    """
    pos = trace_block_start + 20 + 32 * 4

    dict_header_5 = dict(
        leakage_limit=_read(fi, pos, 4, 'flt'),
        leakage_value=_read(fi, pos + 4, 4, 'flt'),
        leakage_error=_read(fi, pos + 24, 1, 'binary')
    )
    return dict_header_5


def _read_trace_header_6(fi, trace_block_start):
    """
    Read trace header 6
    """
    pos = trace_block_start + 20 + 32 * 5

    dict_header_6 = dict(
        device_type=_read(fi, pos, 1, 'binary'),
        device_serial_code=_read(fi, pos + 1, 3, 'binary'),
        seismic_trace_number=_read(fi, pos + 4, 1, 'binary'),
        module_type=_read(fi, pos + 8, 1, 'binary'),
        fdu_module_serial_number=_read(fi, pos + 9, 3, 'binary'),
        position_in_fdu=_read(fi, pos + 12, 1, 'binary'),
        sub_equipment_type=_read(fi, pos + 16, 1, 'binary'),
        seismic_trace_type=_read(fi, pos + 17, 1, 'binary'),
        receiver_sensitivity=_read(fi, pos + 20, 4, 'flt')
    )
    return dict_header_6


def _read_trace_header_7(fi, trace_block_start):
    """
    Read trace header 7
    """
    pos = trace_block_start + 20 + 32 * 6

    dict_header_7 = dict(
        control_unit_type=_read(fi, pos, 1, 'binary'),
        control_unit_serial_number=_read(fi, pos + 1, 3, 'binary'),
        seismic_trace_gain_calibration=_read(fi, pos + 4, 1, 'binary'),
        filter_of_seismic_trace=_read(fi, pos + 5, 1, 'binary'),
        edit_status_of_seismic_trace=_read(fi, pos + 7, 1, 'binary'),
        millivolt_convertion_factor_for_seismic_trace_sampling=_read(fi, pos + 8, 4, 'binary'),
        times_of_noise_stack=_read(fi, pos + 12, 1, 'binary'),
        times_of_low_stack=_read(fi, pos + 13, 1, 'binary'),
        sign_code_of_seismic_trace_type=_read(fi, pos + 14, 1, 'binary'),
        processing_of_seismic_trace=_read(fi, pos + 15, 1, 'binary'),
        maximum_of_record_trace=_read(fi, pos + 16, 4, 'flt'),
        maximum_time_of_record_trace=_read(fi, pos + 20, 4, 'binary'),
        number_of_interpolations=_read(fi, pos + 24, 4, 'binary'),
        deviation_value_of_seismic_trace=_read(fi, pos + 28, 4, 'binary'),
    )
    return dict_header_7


def _read_trace_header_8(fi, trace_block_start):
    """
    Read trace header 8
    """
    pos = trace_block_start + 20 + 32 * 7

    leg_preamp_path = {
        '0': 'external input selected',
        '1': 'simulated data selected',
        '2': 'pre-amp input shorted to ground',
        '3': 'test oscillator with sensors',
        '4': 'test oscillator without sensors',
        '5': 'common mode test oscillator with sensors',
        '6': 'common mode test oscillator without sensors',
        '7': 'test oscillator on positive sensors with neg sensor grounded',
        '8': 'test oscillator on negative sensors with pos sensor grounded',
        '9': 'test oscillator on positive PA input with neg PA input ground',
        '10': 'test oscillator on negative PA input with pos PA input ground',
        '11': 'test oscillator on positive PA input with neg\
                              PA input ground, no sensors',
        '12': 'test oscillator on negative PA input with pos\
                              PA input ground, no sensors'}
    preamp_path_code = str(_read(fi, pos + 24, 4, 'binary'))

    leg_test_oscillator = {'0': 'test oscillator path open',
                           '1': 'test signal selected',
                           '2': 'DC reference selected',
                           '3': 'test oscillator path grounded',
                           '4': 'DC reference toggle selected'}
    oscillator_code = str(_read(fi, pos + 28, 4, 'binary'))

    dict_header_8 = dict(
        fairfield_test_analysis_code=_read(fi, pos, 4, 'binary'),
        first_test_oscillator_attenuation=_read(fi, pos + 4, 4, 'binary'),
        second_test_oscillator_attenuation=_read(fi, pos + 8, 4, 'binary'),
        # start delay in second
        start_delay=_read(fi, pos + 12, 4, 'binary') / 1e6,
        dc_filter_flag=_read(fi, pos + 16, 4, 'binary'),
        dc_filter_frequency=_read(fi, pos + 20, 4, 'flt'),
        preamp_path=leg_preamp_path[preamp_path_code],
        test_oscillator_signal_type=leg_test_oscillator[oscillator_code],
    )
    return dict_header_8


def _read_trace_header_9(fi, trace_block_start):
    """
    Read trace header 9
    """
    pos = trace_block_start + 20 + 32 * 8

    leg_signal_type = {'0': 'pattern is address ramp',
                       '1': 'pattern is RU address ramp',
                       '2': 'pattern is built from provided values',
                       '3': 'pattern is random numbers',
                       '4': 'pattern is a walking 1s',
                       '5': 'pattern is a walking 0s',
                       '6': 'test signal is a specified DC value',
                       '7': 'test signal is a pulse train with\
                             specified duty cycle',
                       '8': 'test signal is a sine wave',
                       '9': 'test signal is a dual tone sine',
                       '10': 'test signal is an impulse',
                       '11': 'test signal is a step function'}
    type_code = str(_read(fi, pos, 4, 'binary'))

    # test signal generator frequency 1 in hertz
    test_signal_freq_1 = _read(fi, pos + 4, 4, 'binary') / 1e3
    # test signal generator frequency 2 in hertz
    test_signal_freq_2 = _read(fi, pos + 8, 4, 'binary') / 1e3
    # test signal generator amplitude 1 in dB down from full scale -120 to 120
    test_signal_amp_1 = _read(fi, pos + 12, 4, 'binary')
    # test signal generator amplitude 2 in dB down from full scale -120 to 120
    test_signal_amp_2 = _read(fi, pos + 16, 4, 'binary')
    # test signal generator duty cycle in percentage
    duty_cycle = _read(fi, pos + 20, 4, 'flt')
    # test signal generator active duration in second
    active_duration = _read(fi, pos + 24, 4, 'binary') / 1e6
    # test signal generator activation time in second
    activation_time = _read(fi, pos + 28, 4, 'binary') / 1e6

    dict_header_9 = dict(
        test_signal_generator_signal_type=leg_signal_type[type_code],
        test_signal_generator_frequency_1=test_signal_freq_1,
        test_signal_generator_frequency_2=test_signal_freq_2,
        test_signal_generator_amplitude_1=test_signal_amp_1,
        test_signal_generator_amplitude_2=test_signal_amp_2,
        test_signal_generator_duty_cycle_percentage=duty_cycle,
        test_signal_generator_active_duration=active_duration,
        test_signal_generator_activation_time=activation_time,
    )
    return dict_header_9


def _read_trace_header_10(fi, trace_block_start):
    """
    Read trace header 10
    """
    pos = trace_block_start + 20 + 32 * 9

    dict_header_10 = dict(
        test_signal_generator_idle_level=_read(fi, pos, 4, 'binary'),
        test_signal_generator_active_level=_read(fi, pos + 4, 4, 'binary'),
        test_signal_generator_pattern_1=_read(fi, pos + 8, 4, 'binary'),
        test_signal_generator_pattern_2=_read(fi, pos + 12, 4, 'binary'),
    )
    return dict_header_10


@_open_file
def _is_rg16(filename, **kwargs):
    """
    Determine if a file is a rg16 file.

    :param filename: a path to a file or a file object
    :type filename: str, buffer
    :rtype: bool
    :return: True if the file object is a rg16 file.
    """
    try:
        sample_format = _read(filename, 2, 2, 'bcd')
        manufacturer_code = _read(filename, 16, 1, 'bcd')
        version = _read(filename, 42, 2, 'binary')
    except ValueError:  # if file too small
        return False
    con1 = version == 262 and sample_format == 8058
    return con1 and manufacturer_code == 20


@_open_file
def _read_initial_headers(filename):
    """
    Extract all the information contained in the headers located before data,
    at the beginning of the rg16 file object.

    :param filename : a path to a rg16 file or a rg16 file object.
    :type filename: str, buffer
    :return: a dictionnary containing all the information of the initial
        headers

    Frequencies are expressed in hertz and time is expressed in second
    (except for the date).
    """
    headers_content = dict(
        general_header_1=_read_general_header_1(filename),
        general_header_2=_read_general_header_2(filename),
        general_header_3=_read_general_header_3(filename),
        channel_sets_descriptor=_read_channel_sets(filename),
        extended_headers=_read_extended_headers(filename),
    )
    return headers_content


def _read_general_header_1(fi):
    """
    Extract information contained in the general header block 1
    """
    gen_head_1 = dict(
        file_number=_read(fi, 0, 2, 'bcd'),
        sample_format_code=_read(fi, 2, 2, 'bcd'),
        general_constant=_read(fi, 4, 6, 'bcd'),
        time_slice_year=_read(fi, 10, 1, 'bcd'),
        nbr_add_general_header=_read(fi, 11, 0.5, 'binary'),
        julian_day=_read(fi, 11, 1.5, 'bcd', False),
        time_slice=_read(fi, 13, 3, 'bcd'),
        manufacturer_code=_read(fi, 16, 1, 'bcd'),
        manufacturer_serial_number=_read(fi, 17, 2, 'bcd'),
        scan_byte_number=_read(fi, 19, 3, 'binary'),
        base_scan_interval=_read(fi, 22, 1, 'binary'),
        polarity_code=_read(fi, 23, 0.5, 'binary'),
        record_type=_read(fi, 25, 0.5, 'binary'),
        extend_record_length=_read(fi, 25, 1.5, 'bcd', False),
        record_scan_type=_read(fi, 27, 1, 'binary'),
        nbr_seismic_trace_group=_read(fi, 28, 1, 'bcd'),
        nbr_sampling_delay_32bytes_extend_head=_read(fi, 29, 1, 'bcd'),
        extend_head_length=_read(fi, 30, 1, 'bcd'),
        external_head_length=_read(fi, 31, 1, 'bcd')
    )
    return gen_head_1


def _read_general_header_2(fi):
    """
    Extract information contained in the general header block 2
    """
    gen_head_2 = dict(
        extended_file_number=_read(fi, 32, 3, 'binary'),
        extended_channel_sets_per_scan_type=_read(fi, 35, 2, 'binary'),
        extended_header_blocks=_read(fi, 37, 2, 'binary'),
        external_header_blocks=_read(fi, 39, 3, 'binary'),
        version_number=_read(fi, 42, 2, 'bcd'),
        nbr_general_tail_data_blocks=_read(fi, 44, 2, 'bcd'),
        # extended record length in second
        extended_record_length=_read(fi, 46, 3, 'binary') / 1e3,
        general_header_block_number=_read(fi, 50, 1, 'binary'),
    )
    return gen_head_2


def _read_general_header_3(fi):
    """
    Extract information contained in the general header block 3
    """
    gen_head_3 = dict(
        extended_file_number=_read(fi, 64, 3, 'binary'),
        source_line_number_integer=_read(fi, 67, 3, 'binary'),
        source_line_number_fraction=_read(fi, 70, 2, 'bcd'),
        source_point_number_integer=_read(fi, 72, 3, 'binary'),
        source_point_number_fraction=_read(fi, 75, 2, 'bcd'),
        source_point_index=_read(fi, 77, 1, 'binary'),
        phase_control=_read(fi, 78, 1, 'binary'),
        type_vibrator=_read(fi, 79, 1, 'binary'),
        phase_angle=_read(fi, 80, 2, 'binary'),
        general_header_block_number=_read(fi, 81, 1, 'binary'),
        source_set_number=_read(fi, 82, 1, 'binary'),
    )
    return gen_head_3


def _read_channel_sets(fi):
    """
    Extract information of all channel set descriptor blocks.
    """
    channel_sets = {}
    general = _read(fi, 11, 0.5, 'binary') - 1
    nbr_channel_set = _read(fi, 28, 1, 'bcd')
    start_byte = 64 + general * 32
    for i in range(0, nbr_channel_set):
        channel_set_name = str(i + 1)
        channel_sets[channel_set_name] = _read_channel_set(fi, start_byte)
        start_byte += 32
    return channel_sets


def _read_channel_set(fi, start_byte):
    """
    Extract information contained in the ith channel set descriptor.
    """
    nbr_32_ext = _read(fi, start_byte + 28, 0.5, 'binary', False)

    channel_set = dict(
        scan_type_number=_read(fi, start_byte, 1, 'bcd'),
        channel_set_number=_read(fi, start_byte + 1, 1, 'bcd'),
        channel_set_start_time=_read(fi, start_byte + 2, 2, 'binary') * 2e-3,
        channel_set_end_time=_read(fi, start_byte + 4, 2, 'binary') * 2e-3,
        optionnal_MP_factor=_read(fi, start_byte + 6, 1, 'bcd'),
        mp_factor_descaler_multiplier=_read(fi, start_byte + 7, 1, 'binary'),
        nbr_channels_in_channel_set=_read(fi, start_byte + 8, 2, 'bcd'),
        channel_type_code=_read(fi, start_byte + 10, 0.5, 'binary'),
        nbr_sub_scans=_read(fi, start_byte + 11, 0.5, 'bcd'),
        gain_control_method=_read(fi, start_byte + 11, 0.5, 'bcd', False),
        alias_filter_frequency=_read(fi, start_byte + 12, 2, 'bcd'),
        alias_filter_slope=_read(fi, start_byte + 14, 2, 'bcd'),
        low_cut_filter_freq=_read(fi, start_byte + 16, 2, 'bcd'),
        low_cut_filter_slope=_read(fi, start_byte + 18, 2, 'bcd'),
        notch_filter_freq=_read(fi, start_byte + 20, 2, 'bcd') / 10,
        notch_2_filter_freq=_read(fi, start_byte + 22, 2, 'bcd') / 10,
        notch_3_filter_freq=_read(fi, start_byte + 24, 2, 'bcd') / 10,
        extended_channel_set_number=_read(fi, start_byte + 26, 2, 'binary'),
        extended_header_flag=_read(fi, start_byte + 28, 0.5, 'binary'),
        nbr_32_byte_trace_header_extension=nbr_32_ext,
        vertical_stack_size=_read(fi, start_byte + 29, 1, 'binary'),
        streamer_cable_number=_read(fi, start_byte + 30, 1, 'binary'),
        array_forming=_read(fi, start_byte + 31, 1, 'binary'),

    )

    return channel_set


def _read_extended_headers(fi):
    """
    Extract information from the extended headers.
    """
    nbr_channel_set = _read(fi, 28, 1, 'bcd')
    general = _read(fi, 11, 0.5, 'binary') - 1
    start_byte = 32 + 32 + 32 * (nbr_channel_set + general)
    dict_type_test_record = {0: '正常记录', 1: '野外噪声', 2: '野外倾斜', 3: '野外串音',
                             4: '仪器噪声', 5: '仪器畸变', 6: '仪器增益/相位', 7: '仪器串音',
                             8: '仪器共模', 9: '合成', 10: '野外脉冲', 11: '仪器脉冲',
                             12: '野外畸变', 13: '仪器重力', 14: '野外漏电', 15: '野外电阻'}
    code_type_test_record = _read(fi, start_byte + 44, 4, 'binary')

    extended_header = dict(
        acquisition_length=_read(fi, start_byte, 4, 'binary'),
        acquisition_rate=_read(fi, start_byte + 4, 4, 'binary'),
        total_nbr_record_trace=_read(fi, start_byte + 8, 4, 'binary'),
        nbr_assist_trace=_read(fi, start_byte + 12, 4, 'binary'),
        nbr_seismic_record_trace=_read(fi, start_byte + 16, 4, 'binary'),
        nbr_stagnation_seismic_record_trace=_read(fi, start_byte + 20, 4, 'binary'),
        nbr_active_seismic_record_trace=_read(fi, start_byte + 24, 4, 'binary'),
        type_viberation=_read(fi, start_byte + 28, 4, 'binary'),
        nbr_samples_in_record_trace=_read(fi, start_byte + 32, 4, 'binary'),
        shot_number=_read(fi, start_byte + 36, 4, 'binary'),
        TB_time_window=_read(fi, start_byte + 40, 4, 'flt'),
        type_test_record=dict_type_test_record[code_type_test_record],
        first_receive_line_number=_read(fi, start_byte + 48, 4, 'binary'),
        first_receive_point_number=_read(fi, start_byte + 52, 4, 'binary'),
        arrangement_number=_read(fi, start_byte + 56, 4, 'binary'),
        source_x=_read(fi, start_byte + 572, 8, 'dbl'),
        source_y=_read(fi, start_byte + 580, 8, 'dbl'),
        source_elevation=_read(fi, start_byte + 588, 4, 'flt')
    )
    return extended_header


if __name__ == '__main__':
    import doctest

    doctest.testmod(exclude_empty=True)
