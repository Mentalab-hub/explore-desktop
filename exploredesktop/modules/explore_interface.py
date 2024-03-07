"""Explore interface class to be used in Explore Desktop"""
import logging
import time
from collections import namedtuple
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Union
)

import explorepy.packet
from explorepy import Explore
from explorepy.settings_manager import SettingsManager
from explorepy.stream_processor import TOPICS
from explorepy.tools import bt_scan


from exploredesktop.modules.app_settings import Settings  # isort: skip


logger = logging.getLogger("explorepy." + __name__)


class ExploreInterface(Explore):
    """Interface class for Explore"""

    def __init__(self):
        super().__init__()
        self.device_chan = 8
        self.chan_dict_list = []
        self.chan_mask = "11111111"
        self.settings = None
        self.record_filename = ""
        self.filters = {}

    @property
    def sampling_rate(self) -> Optional[int]:
        """Returns the current sampling rate of the device"""
        if self.is_connected:
            return self.stream_processor.device_info['sampling_rate']

        logger.debug("Device is not connected but the sampling rate method is called.")
        return None

    @property
    def n_active_chan(self) -> Optional[int]:
        """Returns number of active channels"""
        if self.is_connected:
            if isinstance(self.chan_mask, str):
                return sum([int(i) for i in self.chan_mask])
            return sum(self.chan_mask)

        logger.debug("Device is not connected but the number of active channels method is called.")
        return None

    @property
    def is_recording(self) -> bool:
        """Returns recording status"""
        return bool(self.recorders)

    @property
    def is_pushing_lsl(self) -> bool:
        """Returns LSL pushing status"""
        return bool(self.lsl)

    @staticmethod
    def scan_devices() -> List[namedtuple]:
        """This function searches for nearby bluetooth devices and returns a list of advertising Explore devices.

        Note:
            In Windows, this function returns all the paired devices and unpaired advertising devices. The 'is_paired'
            attribute shows if the device is paired. If a device is paired,
            it will be in the returned list regardless if it is currently advertising or not.

        Returns:
                list of nearby devices
        """
        return bt_scan()

    def connect(self, device_name: str) -> bool:
        """
        Connect to explore device

        Args:
            device_name: Device name ("Explore_XXXX")

        Returns:
            True if successful, False otherwise
        """
        # Connect to the device
        try:
            super().connect(device_name=device_name)
        except ConnectionAbortedError as error:
            logger.debug("Could not connect! %s", str(error))
            return False

        # Set channel mask and default chan names
        self.settings = SettingsManager(device_name)
        self.device_chan = self.settings.get_channel_count()
        self.set_chan_mask()
        self.set_chan_dict_list()
        self.settings.settings_dict[self.settings.channel_name_key] = [
            f"ch{i + 1}" for i in range(self.device_chan)]
        # Activate all channels. Needed because of difference between hardware and software adc mask
        if self.device_chan in [4, 8] and sum(self.stream_processor.device_info['adc_mask']) != self.device_chan:
            self.set_channels("1" * self.device_chan)

        return True

    def disconnect(self):
        """Disconnect from explore device and reset variables
        """
        self.device_chan = None
        self.chan_dict_list = []
        # Save current settings before disconnecting
        if self.device_name is not None:
            SettingsManager(self.device_name).save_current_session()
        return super().disconnect()

    def set_chan_mask(self, mask=None):
        """Set channel mask.
        """
        # Initialize mask by activating all channels
        if mask is None:
            self.chan_mask = [1] * self.device_chan
            self.settings.set_adc_mask(list(reversed(self.chan_mask)))
        elif mask is not None and isinstance(mask, list):
            self.chan_mask = mask
            self.settings.set_adc_mask(list(reversed(self.chan_mask)))
        else:
            logger.error("Mask must be a list, not a %s. Current mask is %s" % (type(mask), mask))

    # TODO change to property
    def set_chan_dict_list(self, new_dict=None):
        """Set the channel status dictionary i.e. whether channels are active or inactive
        """
        if self.is_connected is False:
            return

        # Initialize dict with default values
        if new_dict is None:
            custom_names = [f"ch{i}" for i in range(1, self.device_chan + 1)]
            # NOTE Change line below to hardcode different channel names
            # custom_names = ELECTRODES_10_20 + [
            #     f"ch{i}" for i in range(len(ELECTRODES_10_20) + 1, self.device_chan + 1)]
            signal_types = ["EEG"] * self.device_chan
        else:
            custom_names = [d["name"] for d in new_dict]
            signal_types = [d["type"] for d in new_dict]

        self.chan_dict_list = [
            {
                "input": ch, "enable": bool(active), "name": name, "type": sig_type
            } for ch, active, name, sig_type in zip(
                [c.lower() for c in Settings.CHAN_LIST], self.chan_mask, custom_names, signal_types)
        ]

        # Make sure that the list has the same length as the number of channels
        self.chan_dict_list = self.chan_dict_list[:self.device_chan]

    def get_chan_dict_list(self) -> dict:
        """Retrun channel status dictionary
        """
        return self.chan_dict_list

    def _set_n_chan(self, packet: explorepy.packet.EEG) -> None:
        """Set the number of channels i.e. device type (4-ch or 8-ch)

        Args:
            packet: EEG packet
        """
        exg_fs = self.stream_processor.device_info['sampling_rate']

        # look at board ID to identify 16 chan device
        if 'board_id' in self.stream_processor.device_info.keys() and \
                self.stream_processor.device_info['board_id'] == 'PCB_305_801_XXX':
            self.device_chan = 16
            return

        # look at number of samples in packet to identify 4, 8 and 32 chan devices
        timestamp, _ = packet.get_data(exg_fs)
        if timestamp.shape[0] == 33:
            self.device_chan = 4
        elif timestamp.shape[0] == 4:
            self.device_chan = 32
        elif timestamp.shape[0] == 16:
            self.device_chan = 8

    def get_device_chan(self) -> int:
        """Returns number of channels i.e. device type (4-ch or 8-ch)"""
        return self.device_chan

    def active_chan_list(self, custom_name=False):
        """Returns list of active channels

        Args:
            custom_name (bool, optional): whether to return custom names set by user. Defaults to False.
        """
        if custom_name:
            return [one_chan_dict['name'] for one_chan_dict in self.chan_dict_list if one_chan_dict['enable']]
        return [one_chan_dict['input'] for one_chan_dict in self.chan_dict_list if one_chan_dict['enable']]

    def full_chan_list(self, custom_name) -> list:
        """Returns list of all channels

        Args:
            custom_name (bool, optional): whether to return custom names set by user. Defaults to False.
        """
        if custom_name:
            return [one_chan_dict['name'] for one_chan_dict in self.chan_dict_list]
        return [one_chan_dict['input'] for one_chan_dict in self.chan_dict_list]

    # pylint: disable=arguments-differ
    def measure_imp(self, imp_callback: Callable) -> bool:
        """Activate impedance measurement mode and subscribe to impedance topic

        Returns:
            True if successful, False otherwise"""
        try:
            self.stream_processor.imp_initialize(notch_freq=50)
        except ConnectionError:
            return False

        self.subscribe(callback=imp_callback, topic=TOPICS.imp)
        return True

    def disable_imp(self, imp_callback: Callable) -> bool:
        """Disable impedance measurement mode and unsubscribe from impedance topic

        Returns:
            True if successful, False otherwise
        """
        self.unsubscribe(callback=imp_callback, topic=TOPICS.imp)
        if self.stream_processor.disable_imp():
            return True

        logger.warning("Failed to disable impedance measurement.")
        return False

    # TODO should be property setter
    def set_sampling_rate(self, sampling_rate: int) -> bool:
        """Change the sampling rate of the device

        Returns:
            True if successful, False otherwise
        """
        # If new sampling rate is the same as the current one, do not change
        if sampling_rate == self.sampling_rate:
            return False
        try:
            changed = super().set_sampling_rate(sampling_rate=int(sampling_rate))
            if changed:
                self.settings.set_sampling_rate(int(sampling_rate))
            return changed
        except (ValueError, ConnectionAbortedError) as error:
            logger.error("Error during set sampling rate: %s", str(error))
            return False

    def subscribe(self, callback: Callable, topic: TOPICS) -> None:
        """Subscribe a callback to a topic"""
        self.stream_processor.subscribe(callback, topic)

    def unsubscribe(self, callback: Callable, topic: TOPICS) -> None:
        """Unsubscribe a callback from a topic"""
        self.stream_processor.unsubscribe(callback, topic)

    def add_filter(self, cutoff_freq: Union[float, tuple], filter_type: str) -> None:
        """Add a filter to filtered_ExG topic

        Args:
            cutoff_freq: Cut-off frequency (frequencies) for the filter
            filter_type: Filter type ['bandpass', 'lowpass', 'highpass', 'notch']
        """
        self.stream_processor.add_filter(cutoff_freq, filter_type)

    def remove_filters(self) -> None:
        """Remove all filters from filtered_ExG topic"""
        self.stream_processor.remove_filters()

    def get_bp_filter_limits(self) -> Tuple[float, float]:
        """Returns minimum low cutoff frequency of the bandpass filter"""
        nyq_freq = self.sampling_rate / 2.

        max_hc_freq = round(nyq_freq - 1, 1)
        min_lc_freq = round(0.0035 * nyq_freq, 1)

        return min_lc_freq, max_hc_freq

    def get_settings(self) -> None:
        """Get current device settings from settings manager
        """
        self.settings.load_current_settings()
