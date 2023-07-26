import logging
import time
from typing import (
    Optional,
    Union
)


from exploredesktop.modules.base_model import BaseModel  # isort: skip
from exploredesktop.modules.dialogs import FiltersDialog  # isort: skip
from exploredesktop.modules.utils import (  # isort: skip
    display_msg,
    get_filter_limits,
    verify_filters
)


logger = logging.getLogger("explorepy." + __name__)


class Filters(BaseModel):
    """Visualization filters class
    """

    def __init__(self, ui) -> None:
        super().__init__()
        self.ui = ui
        self.current_filters = None

    def reset_vars(self) -> None:
        """Reset variables"""
        self.current_filters = None
        self.explorer.filters = None

    def setup_ui_connections(self) -> None:
        """Setup connections between widgets and slots"""
        self.ui.btn_plot_filters.clicked.connect(self.popup_filters)

    def popup_filters(self) -> bool:
        """Display popup and apply filters

        Returns:
            bool: whether filters have been applied
        """
        filters = self._get_filters_popup()
        if filters is False:
            return False

        # Check if new filters are the same
        if self._check_same_filters(new_filters=filters):
            return True
        self.current_filters = filters
        self.explorer.filters = filters

        # Remove filters before applying so they dont accumulate
        if self.current_filters is not None:
            self.explorer.remove_filters()

        self.apply_filters()

        # If applying filters for the first time sleep for 1.5 seconds to reduce wavy behavior
        if self.current_filters is None:
            time.sleep(1.5)
        return True

    def _get_filters_popup(self) -> Union[bool, dict]:
        """Get filter values from popup

        Returns:
            Union[bool, dict]: False if popup was closed/cancel, dict with filter values otherwise
        """
        dialog = FiltersDialog(self.explorer.sampling_rate, self.current_filters)
        filters = dialog.exec()
        return filters

    def apply_filters(self) -> None:
        """Apply filters to explore"""
        notch_freq = self.current_filters["notch"]
        high_freq = self.current_filters["high_cutoff"]
        low_freq = self.current_filters["low_cutoff"]

        self._apply_notch_filter(notch_freq)
        self._apply_cutoff_filter(high_freq, low_freq)

        logger.info(f"Applied filters {self.current_filters}")

    def _apply_cutoff_filter(self, high_freq: Optional[float], low_freq: Optional[float]) -> None:
        """Apply filter corresponding to the given cutoff frequencies, i.e. lowpass, highpass or bandpass

        Args:
            high_freq (Optional[float]): value of the high cutoff frequency
            low_freq (Optional[float]): value of the low cutoff frequency
        """
        if high_freq is not None and low_freq is not None:
            self.explorer.add_filter(
                cutoff_freq=(low_freq, high_freq), filter_type='bandpass')
        elif high_freq is not None:
            self.explorer.add_filter(cutoff_freq=high_freq, filter_type='highpass')
        elif low_freq is not None:
            self.explorer.add_filter(cutoff_freq=low_freq, filter_type='lowpass')

    def _apply_notch_filter(self, notch_freq: Optional[int]) -> None:
        """Apply notch filter

        Args:
            notch_freq (Optional[int]): value of the notch filter
        """
        if notch_freq is not None:
            self.explorer.add_filter(cutoff_freq=notch_freq, filter_type='notch')

    def _check_same_filters(self, new_filters: dict) -> bool:
        """
        Compare new filters to existing ones

        Args:
            new_filters (dict): filters to compare

        Returns:
            bool: whether filters are the same
        """
        same = True if self.current_filters == new_filters else False
        return same

    def check_filters_sr(self, s_rate: int) -> None:
        """Check filters for new sampling rate and display warning if it has to be changed

        Args:
            s_rate (int): new sampling rate
        """
        if self.current_filters is None:
            return

        min_lc_freq, max_hc_freq = get_filter_limits(s_rate)

        warning = ""

        hc_freq_warning = (
            "High cutoff frequency cannot be larger than or equal to the nyquist frequency.\n"
            f"The high cutoff frequency has changed to {max_hc_freq:.1f} Hz!"
        )

        lc_freq_warning = (
            "Transient band for low cutoff frequency was too narrow.\n"
            f"The low cutoff frequency has changed {min_lc_freq:.1f} Hz!"
        )

        filter_ok = verify_filters(
            (self.current_filters['low_cutoff'], self.current_filters['high_cutoff']),
            s_rate)

        if filter_ok['lc_freq'] is False:
            warning += lc_freq_warning
            self.current_filters["low_cutoff"] = min_lc_freq

        if filter_ok['hc_freq'] is False:
            warning += hc_freq_warning
            self.current_filters["high_cutoff"] = max_hc_freq

        if warning != "":
            display_msg(msg_text=warning, popup_type="info")
