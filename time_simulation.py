#!/usr/bin/python3

"""Script to simulate the performance of prototype time synchronization algorithms
for Fuchsia."""

import argparse
import math
import random
import collections

import matplotlib
import matplotlib.pyplot as plt
import numpy.random

# Standard deviation of the fixed oscillator error in parts per million.
FIXED_ERROR_SIGMA_PPM = 10
# Standard deviation of the variable oscillator error in parts per million.
VARIABLE_ERROR_SIGMA_PPM = 10
# Time between selections of a new variable oscillator error value, in seconds.
VARIABLE_ERROR_PERIOD_SEC = 5

# The minimum covariance we allow within the Kalman filter to avoid over-indexing on the estimate.
MIN_COVARIANCE = 1e-6

# Conversions between time units.
MILLISECONDS_TO_SEC = 1.0/1000.0
SEC_TO_MILLISECONDS = 1000.0
MINUTES_TO_SEC = 60.0

# One million, used for readability in PPM conversions.
ONE_MILLION = 1_000_000

# All colors are taken from https://flatuicolors.com/palette/defo
PURPLE = '#9b59b6'
GREEN = '#2ecc71'
BLUE = '#3498db'
RED = '#e74c3c'
ORANGE = '#f39c12'


class Measurement:
    """A single measurement of Utc produced by some measuring instument."""
    def __init__(self, true_time, utc, sigma, rtt=None, max_p_window=None):
        self.true_time = true_time
        self.utc = utc
        self.sigma = sigma
        self.max_p_window = max_p_window
        self.rtt = rtt


class SimulationOutput:
    """A collection of time series data from a single execution of the simulation."""
    def __init__(self, description):
        self.description = description
        self.true_time = []
        self.oscillator_error = []
        self.measurement_error = []
        self.measurement_sigma = []
        self.estimate_error = []
        self.covariance = []
    
    def add_point(self, measurement, oscillator, kalman_filter):
        """Adds a new datapoint to the time series using data from the supplied measurement,
        oscillator, and the current state of the kalman filter."""
        oscillator_time = oscillator.time_at(measurement.true_time)
        self.true_time.append(measurement.true_time)
        self.oscillator_error.append(oscillator_time - measurement.true_time)
        self.measurement_error.append(measurement.utc - measurement.true_time)
        self.measurement_sigma.append(measurement.sigma)
        self.estimate_error.append(kalman_filter.estimate_0 - measurement.true_time)
        self.covariance.append(kalman_filter.covariance_00)

    def print(self):
        """Prints a table of the complete output to stdout."""
        print('True  Osc.Err  Meas.Err  Meas.SD  Filter Err  Covariance')
        print('  (s)    (ms)      (ms)     (ms)        (ms)       (s^2)')
        for i in range(0, len(self.true_time)):
            print('{:5.0f} {:+7.2f} {:+9.2f} {:8.0f} {:+12.3f}  {:10.3e}'.format(
                self.true_time[i],
                self.oscillator_error[i] * SEC_TO_MILLISECONDS,
                self.measurement_error[i] * SEC_TO_MILLISECONDS,
                self.measurement_sigma[i] * SEC_TO_MILLISECONDS,
                self.estimate_error[i] * SEC_TO_MILLISECONDS,
                self.covariance[i]
            ))

    @staticmethod
    def _accumulate_stats(prev_stats, data):
        """Updates a [minimum, maximum, count, sum_of_squares] list to include new data."""
        abs_data = [abs(d) for d in data]
        prev_stats[0] = min(prev_stats[0], min(abs_data))
        prev_stats[1] = max(prev_stats[1], max(abs_data))
        prev_stats[2] += len(abs_data)
        prev_stats[3] += sum([d**2 for d in data])

    @staticmethod
    def print_aggregate(outputs):
        """Prints a table of the aggregate statistics for an iterable of outputs."""
        datapoints = 0
        measurement_errors = [9999, 0, 0, 0]
        estimate_errors = [9999, 0, 0, 0]
        estimate_errors_80 = [9999, 0, 0, 0]
        for output in outputs:
            i_80 = int(len(output.true_time) * 0.2)
            datapoints += len(output.true_time)
            SimulationOutput._accumulate_stats(measurement_errors, output.measurement_error)
            SimulationOutput._accumulate_stats(estimate_errors, output.estimate_error)
            SimulationOutput._accumulate_stats(estimate_errors_80, output.estimate_error[i_80:])

        print('Total data points in all runs   {}'.format(datapoints))
        print()
        print('Min measurement error           {:.1f}ms'.format(
            measurement_errors[0] * SEC_TO_MILLISECONDS))
        print('Max measurement error           {:.1f}ms'.format(
            measurement_errors[1] * SEC_TO_MILLISECONDS))
        print('RMS measurement error           {:.1f}ms'.format(
            math.sqrt(measurement_errors[3]/measurement_errors[2]) * SEC_TO_MILLISECONDS))
        print()
        print('Min estimate error              {:.1f}ms'.format(
            estimate_errors[0] * SEC_TO_MILLISECONDS))
        print('Max estimate error              {:.1f}ms'.format(
            estimate_errors[1] * SEC_TO_MILLISECONDS))
        print('RMS estimate error              {:.1f}ms'.format(
            math.sqrt(estimate_errors[3]/estimate_errors[2]) * SEC_TO_MILLISECONDS))
        print()
        print('Max estimate error (final 80%)  {:.1f}ms'.format(
            estimate_errors_80[1] * SEC_TO_MILLISECONDS))
        print('RMS estimate error (final 80%)  {:.1f}ms'.format(
            math.sqrt(estimate_errors_80[3]/estimate_errors_80[2]) * SEC_TO_MILLISECONDS))
        print()

    def plot(self):
        """Plots the simulation output as a new matplotlib figure."""
        fig = plt.figure(figsize=(18, 12), dpi=100)
        fig.tight_layout()
        fig.suptitle(self.description)

        error_axes = plt.subplot(211)
        kalman_filter_axes = plt.subplot(212)
        true_time_mins = [t / 60 for t in self.true_time]

        error_axes.set_ylabel('Error (ms)')
        error_axes.errorbar(true_time_mins,
                        [e * SEC_TO_MILLISECONDS for e in self.measurement_error],
                        yerr=[s * SEC_TO_MILLISECONDS for s in self.measurement_sigma],
                        color=PURPLE, linestyle='', marker='x', label='Measurement')
 
        error_axes.plot(true_time_mins,
                        [e * SEC_TO_MILLISECONDS for e in self.oscillator_error],
                        color=RED, label='Oscillator')
        error_axes.plot(true_time_mins,
                        [e * SEC_TO_MILLISECONDS for e in self.estimate_error],
                        color=GREEN, label='Estimate')
        error_axes.axhline(y=0, color='black')
        error_axes.set_ylim(min(self.measurement_error) * SEC_TO_MILLISECONDS,
                            max(self.measurement_error) * SEC_TO_MILLISECONDS)

        kalman_filter_axes.set_ylabel('Covariance (s^2)')
        kalman_filter_axes.plot(true_time_mins,
                                self.covariance,
                                color=GREEN, label='Covariance')
        kalman_filter_axes.set_ylim(bottom=0)

        # Apply formatting that is common between the two subplots
        for axes in (error_axes, kalman_filter_axes):
            box = axes.get_position()
            axes.set_position([0.08, box.y0, 0.78, box.height])
            axes.set_xlabel('True Time (min)')
            axes.set_xlim(true_time_mins[0], true_time_mins[-1])
            axes.yaxis.set_label_coords(-0.06, 0.5)
            axes.grid(which='major', axis='both', linestyle='--', color='#aaaaaa')
            axes.legend(bbox_to_anchor=(1.13, 1.0), borderaxespad=0., frameon=False)


class Oscillator:
    """A basic model of a crystal oscillator.

    The model applies two errors to its measurement of time:
    * A fixed error, sampled from a normal distribution, representing a constant manufacturing
      error.
    * A variable error, where a new value is sampled from a normal distribution for every N
      seconds, representing a temperature-dependant frequency error on a system under variable load.
    """

    def __init__(self, fixed_error_sigma, variable_error_sigma, variation_period, initial_value=.0):
        """Creates a new Oscillator using the supplied error characteristics.

        fixed_error_sigma - standard deviation of the fixed error, in parts per million
        variable_error_sigma - standard deviation of the variable error, in parts per million
        variation_period - time between selections of new variable error value, in seconds
        initial_value - the oscillator time at true time zero, in seconds"""
        self.variation_period = variation_period
        self.fixed_error_sigma = fixed_error_sigma
        self.variable_error_sigma = variable_error_sigma
        self.initial_value = initial_value
    
    def generate_errors(self, span):
        """Regenerates a new set of random errors for this oscillator, fully defining the time it
        will return for all true times between zero and span."""
        self.fixed_error = numpy.random.normal(0.0, self.fixed_error_sigma)
        self.frequencies = []
        self.oscillator_times = []
        oscillator_time = self.initial_value
        for _ in range(0, int(span/self.variation_period)):
            step_error = self.fixed_error + numpy.random.normal(0.0, self.variable_error_sigma)
            step_frequency = (ONE_MILLION + step_error)/ONE_MILLION
            self.oscillator_times.append(oscillator_time)
            self.frequencies.append(step_frequency)
            oscillator_time += step_frequency * self.variation_period
    
    def time_at(self, true_time):
        """Returns the time a counter using this oscillator would return if queried at true time."""
        index = int(true_time / self.variation_period)
        # Returned time is oscillator time at start of this period plus frequency during the period
        # multiplied by the elapsed time within the period.
        elapsed_time_in_period = true_time - index * self.variation_period
        return self.oscillator_times[index] + self.frequencies[index] * elapsed_time_in_period


class Server:
    """A model of a time server.

    In the future the model will be able to apply some variable error to the time it reports, but
    currently it returns an exact value."""

    def __init__(self):
        """Creates a new server."""
        pass

    def report(self, true_time):
        """Returns a time that would be reported by this server when queried at a true time of
        true_time."""
        return true_time


class Network:
    """A model of the delay introduced by a network.

    Delay is modelled as a fixed minimum delay plus a random delay with Gamma distribution. The
    model does not include any correlation between the delays for multiple samples."""

    SHAPE = 2.0
    SCALE = 2.0

    def __init__(self, min_delay, modal_delay):
        """Creates a new network using the supplied minimal and modal delays in seconds."""
        self.min_delay = min_delay
        self.modal_delay = modal_delay
        self.output_scale = (modal_delay - min_delay) / Network.SCALE

    def __str__(self):
        return "min={:.0f}ms, mode={:.0f}ms".format(self.min_delay * SEC_TO_MILLISECONDS,
                                                    self.modal_delay * SEC_TO_MILLISECONDS)

    def packet_delay(self):
        """Returns the delay in milliseconds for a packet sent over the network."""
        return self.min_delay + numpy.random.gamma(Network.SHAPE, Network.SCALE) * self.output_scale


class Instrument:
    """The common properties and functionality for all time measurement instruments."""
    def __init__(self):
        self.server = None

    def with_server(self, server):
        """Uses the supplied server to provide the times that this instrument measures."""
        self.server = server

    def server_time(self, true_time):
        """Returns the time the associated server would report at true time, or returns true time
        if no server has been set."""
        return self.server.report(true_time) if self.server else true_time


class NormalInstrument(Instrument):
    """A model of an intrument to measure time as received from some server.

    The model applies an error to its measurements as uncorrelated samples from a normal
    distribution."""

    @staticmethod
    def from_args(arg_string):
        """Creates a new NormalInstrument using the supplied command line argument."""
        # Sigma should be supplied in ms but internally we use seconds.
        error_sigma = int(arg_string) * MILLISECONDS_TO_SEC
        return NormalInstrument(error_sigma)

    def __init__(self, error_sigma):
        """Creates a new NormalInstrument using the supplied standard deviation in seconds."""
        self.error_sigma = error_sigma
        Instrument.__init__(self)

    def __str__(self):
        return "Normal instrument, sigma={:.0f}ms".format(self.error_sigma * SEC_TO_MILLISECONDS)

    def measure(self, true_time):
        """Produces a measurement of time at the supplied true time."""
        server_time = super().server_time(true_time)
        utc = numpy.random.normal(server_time, self.error_sigma)
        return Measurement(true_time, utc, self.error_sigma)


class NetworkInstrument(Instrument):
    """A model of an instrument to measure time as received from some server.

    The model samples two delays from the network and assumes the time reported by the server
    occured at the midpoint of this round trip time. Sigma is approximated based on the RTT."""

    @staticmethod
    def from_args(arg_string):
        """Creates a new NetworkInstrument using the supplied command line argument."""
        params = arg_string.split(',')
        if len(params) != 2:
            raise ValueError('Network properties must contain two comma separated values')
        min_delay = int(params[0]) * MILLISECONDS_TO_SEC
        modal_delay = int(params[1]) * MILLISECONDS_TO_SEC
        return NetworkInstrument(Network(min_delay, modal_delay))

    def __init__(self, network):
        """Creates a new NetworkInstrument measuing time to the supplied Server with delays induced
        by the supplied Network."""
        self.network = network
        Instrument.__init__(self)

    def __str__(self):
        return 'Network instrument, network delay {}'.format(str(self.network))

    def measure(self, true_time):
        """Produces a measurement of time beginning at the supplied true_time."""
        # Calculate two independent delays for the inbound and outbound packets. Server processing
        # time is small in comparison so we assume it it included in the packet delay distributions.
        out_delay, in_delay = self.network.packet_delay(), self.network.packet_delay()
        # The server time was measured when the server received the packet....
        utc = self.server_time(true_time + out_delay)
        # ...but our instrument has no idea how the RTT it sees was split between send and receive
        # and so it has to assume the measurement was taken midway through RTT.
        rtt = in_delay + out_delay
        assumed_measurement_true = true_time + rtt / 2
        # sigma is coarsely estimated based on RTT by fitting a curve to the observed standard
        # deviation over a large number of runs binned by RTT. Note that sigma is zero at the
        # minimum possible RTT of 2*min_delay and that the estimate does not depend on the modal
        # delay. Sigma is capped to a minimum of 10ms.
        sigma = max(0.001, (rtt - 2*self.network.min_delay) / 4)
        return Measurement(assumed_measurement_true, utc, sigma, rtt=rtt)



class QuantizedNetworkInstrument(Instrument):
    """A model of an instrument to measure quantized time as received from some server.

    The model samples two delays from the network and assumes the time reported by the server
    occured at the midpoint of this round trip time. The actual time reported by the server
    is rounded to some time quantum, introducing an additional error. Sigma is approximated
    based on the quantum and RTT."""

    @staticmethod
    def from_args(arg_string):
        """Creates a new QuantizedNetworkInstrument using the supplied command line argument."""
        params = arg_string.split(',')
        if len(params) != 3:
            raise ValueError('QuantizedNetwork properties must contain 3 comma separated values')
        min_delay = int(params[0]) * MILLISECONDS_TO_SEC
        modal_delay = int(params[1]) * MILLISECONDS_TO_SEC
        quantum = int(params[2]) * MILLISECONDS_TO_SEC
        return QuantizedNetworkInstrument(Network(min_delay, modal_delay), quantum)

    def __init__(self, network, quantum):
        """Creates a new NetworkInstrument measuing time to the supplied Server with delays induced
        by the supplied Network and timestamps truncated to a multiple of quantum."""
        self.network = network
        self.quantum = quantum
        Instrument.__init__(self)

    def __str__(self):
        return 'Quantized network instrument, quantum={:.0f}ms, network delay {}'.format(
            self.quantum * SEC_TO_MILLISECONDS, str(self.network))

    def measure(self, true_time):
        """Produces a measurement of time beginning at the supplied true_time."""
        # Calculate two independent delays for the inbound and outbound packets. Server processing
        # time is small in comparison so we assume it it included in the packet delay distributions.
        out_delay, in_delay = self.network.packet_delay(), self.network.packet_delay()
        # Wait a random delay before sending the outbound packet so we're not always aligned at the
        # same relative position in the quantum.
        presend_delay = numpy.random.uniform(low=0.0, high=self.quantum)
        # The server time was measured when the server received the packet....
        utc = self.server_time(true_time + presend_delay + out_delay)
        # ... but the time was sent over the network with lower resolution....
        quantized_utc = utc - math.fmod(utc, self.quantum)
        # ... our instrument has to assume the actual UTC was in the middle of this quantum.
        assumed_utc = quantized_utc + self.quantum / 2
        # Our instrument also has no idea how the RTT it sees was split between send and receive
        # and so it has to assume the measurement was taken midway through RTT.
        rtt = in_delay + out_delay
        assumed_measurement_true = true_time + presend_delay + rtt / 2
        # sigma is coarsely and pessimistically estimated based on RTT and quantum by fitting a
        # curve to the observed standard deviation over a large number of runs binned by RTT. Note
        # that this formula reduces to the NetworkIntrument as quantum approaches zero.
        rtt_over_min = (rtt - 2*self.network.min_delay)
        sigma = max(0.001,
                    (rtt_over_min**2 * self.quantum * 0.03)
                    + (rtt_over_min * (0.25 - (0.13 * self.quantum)))
                    + (0.29 * self.quantum))
        return Measurement(assumed_measurement_true, assumed_utc, sigma, rtt=rtt,
                           max_p_window=self.quantum)

class CombinedQuantizedNetworkInstrument(Instrument):
    """A model of an instrument that polls a quantized network instrument multiple times to improve
    accuracy.
    """
    MIN_SEC_BETWEEN_POLLS = 10
    POLLS_PER_SAMPLE = 4

    @staticmethod
    def from_args(arg_string):
        """Creates a new CombinedQuantizedNetworkInstrument using the supplied command line argument."""
        params = arg_string.split(',')
        if len(params) != 4:
            raise ValueError('CombinedQuantizedNetwork properties must contain 4 comma separated values')
        min_delay = int(params[0]) * MILLISECONDS_TO_SEC
        modal_delay = int(params[1]) * MILLISECONDS_TO_SEC
        quantum = int(params[2]) * MILLISECONDS_TO_SEC
        mode = params[3]
        return CombinedQuantizedNetworkInstrument(Network(min_delay, modal_delay), quantum, mode)

    def __init__(self, network, quantum, mode):
        """Creates a new NetworkInstrument measuing time to the supplied Server with delays induced
        by the supplied Network and timestamps truncated to a multiple of quantum."""
        self.network = network
        self.quantum = quantum
        self.mode = mode
        Instrument.__init__(self)

    def __str__(self):
        return 'Combined Quantized network instrument, quantum={:.0f}ms, network delay {}'.format(
            self.quantum * SEC_TO_MILLISECONDS, str(self.network))

    def _measure(self, true_time):
        """Produce a bound on the real UTC time beginning at true_time"""
        # Calculate two independent delays for the inbound and outbound packets. Server processing
        # time is small in comparison so we assume it it included in the packet delay distributions.
        out_delay, in_delay = self.network.packet_delay(), self.network.packet_delay()
        # The server time was measured when the server received the packet....
        utc = self.server_time(true_time + out_delay)
        # ... but the time was sent over the network with lower resolution....
        quantized_utc = utc - math.fmod(utc, self.quantum)
        # Our instrument also has no idea how the RTT it sees was split between send and receive
        # and so it has to assume the measurement was taken midway through RTT.
        rtt = in_delay + out_delay
        delta = rtt / 2
        assumed_measurement_true = true_time + delta
        return ((assumed_measurement_true, quantized_utc - delta, quantized_utc + self.quantum + delta), delta)

    def _combine(self, earlier_bound, later_bound):
        time_diff = later_bound[0] - earlier_bound[0]
        # Calculate a worst case error due to drift.
        time_diff_err = time_diff*FIXED_ERROR_SIGMA_PPM*3.0/ONE_MILLION
        # Add time diff to each time entry.
        updated_earlier = (earlier_bound[0] + time_diff, \
            earlier_bound[1] + time_diff - time_diff_err, \
            earlier_bound[2] + time_diff + time_diff_err)
        # if intervals don't overlap we can't combine
        if later_bound[1] > updated_earlier[2] or later_bound[2] < updated_earlier[1]:
            return None
        return (later_bound[0], \
            max(later_bound[1], updated_earlier[1]), \
            min(later_bound[2], updated_earlier[2]))

    def _ideal_time(self, prev_bounds):
        if self.mode == 'random':
            return prev_bounds[0] + self.MIN_SEC_BETWEEN_POLLS + \
                numpy.random.uniform(low=0.0, high=self.quantum)
        elif self.mode == 'choose':
            # Try to choose a start time so that the bound size can be halved when combined with
            # the new bound.
            additional_subsec = self.quantum - math.fmod((prev_bounds[1] + prev_bounds[2])/2, self.quantum)
            return prev_bounds[0] + self.MIN_SEC_BETWEEN_POLLS + additional_subsec

    def measure(self, true_time):
        """Produces a measurement of time beginning at the supplied true_time."""
        # even for the initial sample offset by some random quantization
        bounds, delta = self._measure(true_time + numpy.random.uniform(low=0.0, high=self.quantum))
        deltas = [delta]
        for i in range(1, self.POLLS_PER_SAMPLE):
            ideal_time = self._ideal_time(bounds)
            # subtract estimate of delta so that our time is roughly around the desired true time
            approximated_delta = sum(deltas) / len(deltas)
            adj_true_time = ideal_time - approximated_delta
            new_bounds, new_delta = self._measure(adj_true_time)
            deltas.append(new_delta)
            bounds = self._combine(bounds, new_bounds)
        assumed_monotonic = bounds[0]
        assumed_utc = (bounds[1] + bounds[2]) / 2
        # estimate sigma based on the size of the final bound. This value was eyeballed from
        # binnederror results where bins were based on bound size.
        sigma = (bounds[2] - bounds[1]) / 5
        return Measurement(assumed_monotonic, assumed_utc, sigma)

class KalmanFilter:
    """A simple Kalman filter that consumes measurements to track time.

    The state vector is [estimated_utc, oscillator_frequency_factor] but the frequency is supplied
    at construcution and remains constant rather than responding to the measurements."""
    def __init__(self, frequency_factor, oscillator_variance):
        self.filter_time = None
        self.estimate_0 = None
        self.estimate_1 = frequency_factor
        self.covariance_00 = None
        self.oscillator_variance = oscillator_variance

    def propagate_and_correct(self, oscillator_time, measurement):
        """Propogagates the filter state forward to the supplied oscillator_time and applies the
        supplied measurement."""
        if self.estimate_0 is None:
            # Use the first measurement we receive to initialize the filter, including setting the
            # top left element in the covariance matrix to the recieved measurement variance.
            self.filter_time = oscillator_time
            self.estimate_0 = measurement.utc
            self.covariance_00 = max(measurement.sigma**2, MIN_COVARIANCE)
        else:
            # Predict by moving forward to the oscillator_time, note only the top left element in
            # the covariance matrix is non-zero hence this is the only element we calculate.
            time_step = oscillator_time - self.filter_time
            apriori_estimate_0 = self.estimate_0 + self.estimate_1 * time_step
            apriori_covariance_00 = self.covariance_00 + (time_step**2 * self.oscillator_variance)
            # Correct by incorporating the measurement data, only the first element in K is non-zero
            K_0 = apriori_covariance_00 / (apriori_covariance_00 + measurement.sigma**2)
            # Considered skipping the estimate update here if the apriori_estimate was already
            # within half the max_p_window from the measurement utc, but the simulation results
            # show poor performance, with the majority of updates being skipped and very long
            # convergence times.
            aposteriori_estimate_0 = \
                apriori_estimate_0 + K_0 * (measurement.utc - apriori_estimate_0)
            aposteriori_covariance_00 = (1 - K_0) * apriori_covariance_00
            # And update the stored filter state
            self.filter_time = oscillator_time
            self.estimate_0 = aposteriori_estimate_0
            self.covariance_00 = max(aposteriori_covariance_00, MIN_COVARIANCE)


def perform_run(oscillator, instrument, kalman_filter, span):
    """Performs a single run of the simluation using the supplied oscillator, filter, and
    instrument for the supplied time span, returning a SimulationOuput."""
    # Take and apply measurements at a fixed measurement_interval in true time. In reality a
    # system would be using oscillator time rather than true time in determining when to make its
    # measurements, but the difference between these is small and the system is not sensitive to
    # the exact time at which a measurement is made.
    measurement_interval = 120
    oscillator.generate_errors(span)
    out = SimulationOutput(str(instrument))
    for intended_time in range(0, int(span), measurement_interval):
        measurement = instrument.measure(intended_time)
        # Not all instruments can produce a measurement at the exact true time we want, use the true
        # time the instrument produced.
        true_time = measurement.true_time
        oscillator_time = oscillator.time_at(true_time)
        kalman_filter.propagate_and_correct(oscillator_time, measurement)
        out.add_point(measurement, oscillator, kalman_filter)
    return out


def run_simulation(instrument, span):
    """Run a simulation once using the supplied instrument and the standard oscillator and kalman
    filter, plotting the performance."""
    oscillator = Oscillator(FIXED_ERROR_SIGMA_PPM, VARIABLE_ERROR_SIGMA_PPM,
                            VARIABLE_ERROR_PERIOD_SEC)
    assumed_oscillator_variance = (
        (FIXED_ERROR_SIGMA_PPM + VARIABLE_ERROR_SIGMA_PPM) / ONE_MILLION)**2
    kalman_filter = KalmanFilter(1.0, assumed_oscillator_variance)
    output = perform_run(oscillator, instrument, kalman_filter, span)
    output.plot()
    plt.show()


def run_batch_simulation(instrument, span, runs):
    """Run a simulation multiple times using the supplied instrument and the standard oscillator
    and kalman filter, printing a summary of the overall performance."""
    # Initialize the elements of the simulation.
    oscillator = Oscillator(FIXED_ERROR_SIGMA_PPM, VARIABLE_ERROR_SIGMA_PPM,
                            VARIABLE_ERROR_PERIOD_SEC)
    assumed_oscillator_variance = (
        (FIXED_ERROR_SIGMA_PPM + VARIABLE_ERROR_SIGMA_PPM) / ONE_MILLION)**2
    kalman_filter = KalmanFilter(1.0, assumed_oscillator_variance)
    outputs = [perform_run(oscillator, instrument, kalman_filter, span) for _ in range(runs)]
    SimulationOutput.print_aggregate(outputs)


def run_frequency(instrument, span):
    """Run a simulation once using the supplied instrument and the standard oscillator and kalman
    filter, printing an average frequency estimate."""
    oscillator = Oscillator(FIXED_ERROR_SIGMA_PPM, VARIABLE_ERROR_SIGMA_PPM,
                            VARIABLE_ERROR_PERIOD_SEC)
    assumed_oscillator_variance = (
        (FIXED_ERROR_SIGMA_PPM + VARIABLE_ERROR_SIGMA_PPM) / ONE_MILLION)**2
    kalman_filter = KalmanFilter(1.0, assumed_oscillator_variance)
    output = perform_run(oscillator, instrument, kalman_filter, span)

    n = len(output.true_time)
    oscillator_times = [output.true_time[i] + output.oscillator_error[i] for i in range(n)]
    measurement_times = [output.true_time[i] + output.measurement_error[i] for i in range(n)]

    sum_utc = sum(measurement_times)
    sum_oscillator = sum(oscillator_times)
    sum_utc_multiply_oscillator = sum([oscillator_times[i] * measurement_times[i]
                                      for i in range(n)])
    sum_utc_squared = sum([measurement_times[i]**2 for i in range(n)])
    frequency_estimate = ((sum_utc_multiply_oscillator - (sum_utc * sum_oscillator)/n) /
                          (sum_utc_squared - (sum_utc**2)/n))
    frequency_estimate_ppm = (frequency_estimate - 1.0) * ONE_MILLION
    print("Oscillator fixed error (ppm): {:.3f}".format(oscillator.fixed_error))
    print("Estimated frequency error (ppm): {:.3f}".format(frequency_estimate_ppm))


def run_error_visualization(instrument):
    """Plots the error distribution of the supplied instrument and calculates overall sigma."""
    errors = []
    runs = 1_000_000
    for _ in range(runs):
        start_time = numpy.random.uniform(low=0.0, high=1000.0)
        measurement = instrument.measure(start_time)
        error = (measurement.utc - measurement.true_time) * SEC_TO_MILLISECONDS
        errors.append(error)
    sigma = math.sqrt(sum([e**2 for e in errors]) / len(errors))

    fig = plt.figure(figsize=(12, 8), dpi=100)
    fig.tight_layout()
    plt.title("{}     Standard deviation = {:.1f}ms".format(str(instrument), sigma))
    axes = plt.gca()
    axes.set_ylabel('Frequency')
    axes.set_xlabel('Error (ms)')
    axes.hist(errors, bins=100)
    axes.axvline(x=0, color='black')
    axes.grid(which='major', axis='both', linestyle='--', color='#aaaaaa')
    plt.show()


def run_error_binning(instrument):
    """Quantifies the error characteristics of the supplied instrument."""
    runs = 10_000_000
    rtt_bin_width = 0.05
    rtt_bin_count = 40
    max_binnable = (rtt_bin_count * rtt_bin_width)
    rtt_bins = [[0, 0] for _ in range(rtt_bin_count)]
    for _ in range(runs):
        start_time = numpy.random.uniform(low=0.0, high=1000.0)
        measurement = instrument.measure(start_time)
        error = (measurement.utc - measurement.true_time) * SEC_TO_MILLISECONDS
        if measurement.rtt and measurement.rtt < max_binnable: 
            i = int(measurement.rtt / rtt_bin_width)
            rtt_bins[i][0] += 1
            rtt_bins[i][1] += error**2
    for i in range(rtt_bin_count):
        if rtt_bins[i][0] > 0:
            print("{:.0f}-{:.0f}ms RTT {} runs, sigma={:.1f}ms".format(
                rtt_bin_width * i * SEC_TO_MILLISECONDS,
                rtt_bin_width * (i+1) * SEC_TO_MILLISECONDS,
                rtt_bins[i][0], math.sqrt(rtt_bins[i][1]/rtt_bins[i][0])))


def create_parser():
    """Creates the definition of the expected command line flags."""
    parser = argparse.ArgumentParser(
        description='Script to plot develop, explore, and validate the alorithms proposed for '
                    'Fuchsia time synchronization.',
        epilog='Jody Sankey 2020')
    parser.add_argument('--seed', action='store', type=int,
                        help="Integer to seed the random number generator.")
    parser.add_argument('--span', action='store', type=int, metavar='MIN', default=60,
                        help="Minutes of true time to simulate for each run.")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--simulation', action='store_true',
                        help="Run the simulation one time and plots the results.")
    modes.add_argument('--frequency', action='store_true',
                        help="Run the simulation one time and plots a frequency estimate over "
                             "the results.")
    modes.add_argument('--simulations', action='store', type=int, metavar='N',
                        help="Run the simulation multiple times and tabulates performance.")
    modes.add_argument('--errors', action='store_true', 
                        help="Plot the error characteristics of the instrument.")
    modes.add_argument('--binnederrors', action='store_true', 
                        help="Tabulate the error characteristics of the instrument binned by RTT.")
    instruments = parser.add_mutually_exclusive_group(required=True)
    instruments.add_argument('--normal', action='store', metavar='SIGMA_MS',
                        help="Use a instrument with error sampled from a normal distibution. Value "
                             "is the standard deviation in milliseconds.")
    instruments.add_argument('--network', action='store',  metavar='PROPS',
                        help="Use a instrument with error based on network delays. Value is min "
                             "network latency in milliseconds, comma, modal latency in "
                             "milliseconds.")
    instruments.add_argument('--quantized', action='store',  metavar='PROPS',
                        help="Use a instrument with error based on network delays and quantized "
                             "communication. Value is min network latency in milliseconds, comma, "
                             "modal latency in milliseconds, comma, quantum in milliseconds.")
    instruments.add_argument('--cquantized', action='store',  metavar='PROPS',
                        help="Use a instrument with error based on network delays and quantized "
                             "communication. Value is min network latency in milliseconds, comma, "
                             "modal latency in milliseconds, comma, quantum in milliseconds.")
    return parser


def main():
    """Main function to execute the script using command line inputs."""
    args = create_parser().parse_args()

    # Seed the RNG
    if args.seed:
        seed = args.seed
    else:
        # Seed with a known random number so we could repeat this run if needed.
        random.seed()
        seed = random.randrange(0, 0xffffffff)
    print("Seeding rng with {}".format(seed))
    random.seed(seed)

    # Initialize the instrument used to produce time measurements, including the server and network
    # incorporated in the insturment where these are relevant.
    #server = Server()
    if args.normal is not None:
        instrument = NormalInstrument.from_args(args.normal)
    elif args.network is not None:
        instrument = NetworkInstrument.from_args(args.network)
    elif args.quantized is not None:
        instrument = QuantizedNetworkInstrument.from_args(args.quantized)
    elif args.cquantized is not None:
        instrument = CombinedQuantizedNetworkInstrument.from_args(args.cquantized)

    if args.simulation:
        run_simulation(instrument, args.span * MINUTES_TO_SEC)
    if args.simulations:
        run_batch_simulation(instrument, args.span * MINUTES_TO_SEC, args.simulations)
    if args.frequency:
        run_frequency(instrument, args.span * MINUTES_TO_SEC)
    elif args.errors:
        run_error_visualization(instrument)
    elif args.binnederrors:
        run_error_binning(instrument)


if __name__ == '__main__':
    main()
    