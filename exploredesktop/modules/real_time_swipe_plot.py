import math
import sys
import numpy as np

import explorepy
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QScrollBar
from explorepy.stream_processor import TOPICS
from explorepy.settings_manager import SettingsManager
from vispy import app
from vispy import gloo
from vispy import visuals
from vispy.util import keys

from exploredesktop.modules import Settings


class CircularBufferPadded:
    """Implements a circular buffer that is twice as long as its max allowed size. Data added to the buffer is
    inserted in two locations (at last_index and last_index + max_length) and loops back around when the end of the
    buffer (2 * max size) is reached. This ensures that the data inside the buffer is always available in one
    contiguous block of memory, allowing numpy to access the entire buffer as a view without copying it. This allows
    very fast (read) access to the data.
    """

    # TODO: Test circular buffer boundaries
    def __init__(self, max_length, dtype=np.float32):
        self.max_length = max_length
        self.buffer = np.empty(self.max_length * 2, dtype=dtype)
        self.length = 0
        self.index_first = 0
        self.index_last = 0

    def view_end(self, length, offset=0):
        """Views the last length entries in the buffer with an optional offset from the end.
        """
        # Rollover for reading isn't possible, so this doesn't have to be taken into account
        if (length + offset) < self.length:
            return self.buffer[self.index_last - length - offset:self.index_last - offset]
        elif length < self.length:
            return self.buffer[self.index_first:self.index_first + length]
        else:
            return self.buffer[self.index_first:self.index_last]

    def insert_iterating(self, chunk):
        """Inserts new data given by chunk into the buffer. Data is inserted in two locations (from last_index and
        last_index + max_size). The index of insertion loops back to the very start of the buffer at 2 * max_size.
        """
        # TODO: Slicing the chunk correctly may be faster than enumerating but increases complexity due
        for it, element in enumerate(chunk):
            loc = (self.index_last + it) % self.max_length
            self.buffer[loc] = element
            self.buffer[loc + self.max_length] = element
        self.length = min(self.length + len(chunk), self.max_length)
        self.index_last = (self.index_last + len(chunk)) % (self.max_length * 2)
        self.index_first = (self.index_last - self.length) % (self.max_length * 2)
        if self.index_first >= self.index_last:
            self.index_last = (self.index_last + self.max_length) % (self.max_length * 2)
            self.index_first = (self.index_first + self.max_length) % (self.max_length * 2)

    def get_all_info(self):
        return f"{np.array2string(self.buffer)}\n" \
               f"First index: {self.index_first}\n" \
               f"Last index: {self.index_last}\nLength: {self.length}"

    def get_last_index(self):
        return self.index_last

    def get_first_index(self):
        return self.index_first

    def get_distance(self, index):
        # get distance/length between given index and current index
        return self.index_last - (index % self.max_length)

    def get_length(self):
        return self.length

    def __str__(self):
        return f"{np.array2string(self.buffer[self.index_first:self.index_last])}"


class ExploreDataHandlerCircularBuffer:
    """Class that receives data from the Explore device and writes it to a padded circular buffer for fast read access.
    """
    def __init__(self, dev_name, interface=None):
        """Initialises an ExploreDataHandlerCircularBuffer object according to given Explore device name or Explore
        interface. If no interface is passed through, the data handler establishes its own connection and creates an
        Explore object (i.e. when testing outside of Explore Desktop).
        """
        self.dev_name = dev_name
        self.interface = interface
        self.is_connected = False
        if self.interface:
            self.explore_device = interface
            self.dev_name = self.explore_device.device_name
        else:
            self.explore_device = explorepy.Explore()
            self.explore_device.connect(self.dev_name)

        self.current_sr = 0  # TODO: current_sr needs to be adjusted when the sampling rate is changed from the settings
        self.max_channels = 0
        self.packet_length = 0
        self.max_length = 0
        self.max_length_markers = 0

        self.max_duration = 60 * 60  # in seconds

        self.moving_average_window = 200
        self.time_offset = 0

        self.settings = None
        self.channel_mask = []
        self.channels = {}

        # buffers
        self.timestamps = None
        self.markers = None
        self.timestamps_markers = None

        self.callbacks = {
            TOPICS.filtered_ExG: [self.handle_exg],
            TOPICS.raw_orn: [],
            TOPICS.marker: [self.handle_marker]
        }

        if self.explore_device.is_connected:
            self.is_connected = True
            self.dev_name = self.explore_device.device_name
            self.setup_internal_data()
            self.subscribe_packet_callbacks()

    def on_connected(self):
        self.is_connected = True
        self.dev_name = self.explore_device.device_name
        self.setup_internal_data()
        self.subscribe_packet_callbacks()

    def on_disconnected(self):
        self.is_connected = False
        self.dev_name = None
        self.unsubscribe_packet_callbacks()
        self.clear_buffers()

    def clear_buffers(self):
        self.channels = {}

        self.timestamps = None
        self.markers = None
        self.timestamps_markers = None

        self.current_sr = 0
        self.max_channels = 0
        self.packet_length = 0
        self.max_length = 0
        self.max_length_markers = 0

    def setup_internal_data(self):
        self.current_sr = int(self.explore_device.stream_processor.device_info['sampling_rate'])
        self.settings = SettingsManager(self.dev_name)
        self.max_channels = self.settings.get_channel_count()
        self.channel_mask = self.settings.get_adc_mask()
        self.channel_mask.reverse()

        self.packet_length = self.get_packet_length()  # requires max_channels to be set

        self.max_length = self.current_sr * self.max_duration
        self.max_length_markers = self.max_duration * 30  # 30 markers per second

        self.setup_buffers()

    def setup_buffers(self):
        for i in range(self.max_channels):
            self.channels[i] = CircularBufferPadded(self.max_length, dtype=np.float32)

        self.timestamps = CircularBufferPadded(self.max_length, dtype=np.float32)
        self.markers = CircularBufferPadded(self.max_length_markers, dtype='<U10')
        self.timestamps_markers = CircularBufferPadded(self.max_length_markers, dtype=np.float32)

    def change_settings(self):
        self.change_channel_mask()

    def change_channel_mask(self):
        self.channel_mask = self.settings.get_adc_mask()
        self.channel_mask.reverse()

    def subscribe_packet_callbacks(self):
        self.sub_unsub_packet_callbacks(sp_callback=self.explore_device.stream_processor.subscribe)

    def unsubscribe_packet_callbacks(self):
        self.sub_unsub_packet_callbacks(sp_callback=self.explore_device.stream_processor.unsubscribe)

    def sub_unsub_packet_callbacks(self, sp_callback):
        """Subscribes or unsubscribes the device's/interface's stream processor (given by sp_callback) to all internal
        packet callbacks.
        """
        for topic, callbacks in self.callbacks.items():
            for callback in callbacks:
                sp_callback(callback=callback, topic=topic)

    def handle_exg(self, packet):
        """Inserts incoming exg data and timestamps from a packet into internal buffers.
        """
        timestamps, channels = packet.get_data(exg_fs=self.current_sr)
        if self.time_offset == 0:
            self.time_offset = timestamps[0]

        timestamps = timestamps - self.time_offset
        for i in range(len(channels)):
            self.channels[i].insert_iterating(channels[i])
        self.timestamps.insert_iterating(timestamps)

    def handle_orn(self, packet):
        raise NotImplementedError

    def handle_marker(self, packet):
        """Inserts marker label and timestamp from a packet into internal buffers.
        """
        timestamp, marker_string = packet.get_data()
        if self.time_offset == 0:
            self.time_offset = timestamp
        timestamp = timestamp - self.time_offset
        if timestamp <= self.timestamps_markers.view_end(0):
            print(f"Marker timestamp in the past, last: {self.markers.view_end(0)}, new: {timestamp}")
        else:
            self.markers.insert_iterating(marker_string)
            self.timestamps_markers.insert_iterating(timestamp)

    def get_all_channels_and_time(self, duration, offset=0):
        """Returns a list of numpy views of the last duration-many seconds (with a potential offset from the end) of the
        internal ExG channel buffers and a numpy view of the timestamp. The amount of data returned is determined by
        the given duration and the current sampling rate.
        """
        num_values = duration * self.current_sr
        all_channels = [None for _ in range(len(self.channels))]
        if offset == 0:
            current_index = self.channels[0].get_last_index() % num_values
        else:
            current_index = 0

        timestamps = self.timestamps.view_end(num_values, offset=offset)
        for i in range(len(self.channel_mask)):
            if self.channel_mask[i] == 1:
                all_channels[i] = self.channels[i].view_end(num_values, offset=offset)

        return all_channels, timestamps, current_index

    def get_packet_length(self):
        return 4 if (self.max_channels == 32 or self.max_channels == 16) \
            else 16 if self.max_channels == 8 \
            else 33

    def get_last_index(self, channel):
        return self.channels[channel].get_last_index()

    def get_max_length(self):
        return self.max_length

    def get_length(self, channel):
        return self.channels[channel].get_length()

    def get_distance(self, channel, from_index):
        return self.channels[channel].get_distance(from_index)

    def get_markers(self):
        """Returns a numpy view of all marker labels and marker timestamps.
        """
        return self.markers.view_end(self.max_length_markers), self.timestamps_markers.view_end(self.max_length_markers)

    def get_current_sr(self):
        return self.current_sr

    def get_num_plots(self):
        return self.max_channels

    def get_channel_mask(self):
        return self.channel_mask


# *** VERTEX SHADERS ***

vertex_default = """
    /* Default vertex shader that sets x and y coordinate according to incoming 2D pos vector without transformations.
    */
    
    #version 120

    attribute vec2 pos;
    
    void main() {
        gl_Position = vec4(pos.x, pos.y, 0.0f, 1.0f);
    }
"""

vertex_channel = """
    /* Vertex shader used by the channels.
    It determines y position according to...
     - incoming voltage, current baseline and y_scale (pos_y, baseline, half of y_range)
     - vertical paddings (vertical_padding, top_padding, bottom padding)
     - plot index and maximum plots to draw (plot_index, num_plots)
     
     It determines x position according to...
     - incoming timestamp, current duration and leftmost timestamp (pos_x, x_length, x_min)
     - mode of drawing (is_scrolling, is_swipe_plot)
     - horizontal paddings (horizontal_padding, left_padding, right_padding)
    
     It additionally takes a line colour and passes it to the fragment shader. */

    #version 120
    
    uniform float vertical_padding;
    uniform float left_padding;
    uniform float right_padding;
    uniform float horizontal_padding;
    uniform float top_padding;
    uniform float bottom_padding;

    uniform vec4 line_colour;
    uniform vec4 line_colour_highlighted;

    uniform float plot_index;
    uniform float num_plots;

    uniform bool is_scrolling;
    uniform bool is_swipe_plot;

    uniform float x_length;
    uniform float y_range;

    uniform float baseline;
    uniform float x_min;

    attribute float pos_x;
    attribute float pos_y;

    varying vec4 v_col;

    void main() {
        float available_y_range = (2.0 - 2.0*vertical_padding - top_padding - bottom_padding);
        float available_plot_range = available_y_range / num_plots;
        float offset = ((num_plots-plot_index)-0.5)*(1.0 / num_plots) * available_y_range;
        
        float y = pos_y - baseline;
        y = y / (y_range / 2.0f); // all values in [-1;1], max_y = 1.0, min_y = -1.0
        y = y * available_plot_range * 0.5;
        y = y + offset - 1 + bottom_padding + vertical_padding;
        
        
        // calculate new x
        float x;
        if(!is_scrolling && is_swipe_plot) {
            x = mod(pos_x, x_length);
        } else {
            x = (pos_x - x_min);
        }
        float available_x_range = 2.0 - 2.0*horizontal_padding - left_padding - right_padding;
        x = (x / x_length) * available_x_range - 1.0 + left_padding + horizontal_padding;
        
        v_col = line_colour;
        //uncomment for rainbow lines!
        //float r = pow(x, 2.0);
        //float g = -3*pow(x+0.33, 2.0)+1;
        //float b = -3*pow(x-0.33, 2.0)+1;
        
        float val = (2.0f * plot_index / num_plots) - 1.0f;
        float r = pow(val, 2.0);
        float g = -3*pow(val+0.33, 2.0)+1;
        float b = -3*pow(val-0.33, 2.0)+1;
        
        v_col = vec4(vec3(r, g, b) * 0.5f + 0.3f * vec3(1.0f, 1.0f, 1.0f), 1.0);
        gl_Position = vec4(x, y, 0.0, 1.0);
    }
"""

vertex_vertical_line = """
    /* Vertex shader that is used by any program drawing vertical lines. It determines the x position of the line from
    an incoming timestamp (pos.x) in the same way it is determined for channels in the vertex_channel VS.
    The y coordinate (pos.y) is calculated outside the VS and only passed along.*/
    
    #version 120

    uniform float x_length;
    uniform float horizontal_padding;
    uniform float left_padding;
    uniform float right_padding;

    uniform float x_min;

    uniform bool is_scrolling;

    attribute vec2 pos;
    
    void main(void) {
        float x;
        if (!is_scrolling){
            x = mod(pos.x, x_length);
        } else {
            x = pos.x - x_min;
        }
        float available_x_range = 2.0 - 2.0*horizontal_padding - left_padding - right_padding;
        x = (x / x_length) * available_x_range - 1.0 + left_padding + horizontal_padding;

        gl_Position = vec4(x, pos.y, 0.0, 1.0);
    }
"""

vertex_padding = """
    /* Vertex shader used for drawing the ticks on the y_axis. x and y coordinates are calculated outside of the VS
    without padding (so for x in [-1;1] and y in [-1;1]), this VS only adds padding to the coordinates.*/
    
    #version 120
    
    attribute vec2 pos;
    
    uniform float vertical_padding;
    uniform float horizontal_padding;
    uniform float top_padding;
    uniform float bottom_padding;
    uniform float left_padding;
    uniform float right_padding;
    
    void main(void) {
        float available_y_range = 2.0 - 2.0*vertical_padding - top_padding - bottom_padding;
        float available_x_range = 2.0 - 2.0*horizontal_padding - left_padding - right_padding;
        float y = pos.y * available_y_range - 1.0 + bottom_padding + vertical_padding;
        float x = pos.x * available_x_range - 1.0 + left_padding + horizontal_padding;
        gl_Position = vec4(x, y, 0.0, 1.0);
    }
"""

# *** FRAGMENT SHADERS ***

fragment_explore_swipe = """
    /* Fragment shader that gets a colour from the VS and sets it as fragment colour. */
    
    #version 120
    
    varying vec4 v_col;

    void main() {
        gl_FragColor = v_col;
    }
    """

frag_explore_marker = """
    /* Fragment shader that sets the fragment colour to a bright red. */
    
    #version 120
    
    void main()
    {
        gl_FragColor = vec4(0.9, 0.0, 0.0, 1.0);
    }
"""

fragment_explore_swipe_line = """
    /* Fragment shader that sets the fragment colour to a blueish grey. */
    
    #version 120
    
    void main()
    {
        gl_FragColor = vec4(0.3, 0.3, 0.4, 1.0);
    }
"""

fragment_axes = """
    /* Fragment shader that sets the fragment colour to a blueish grey. */
    
    #version 120
    
    void main()
    {
        gl_FragColor = vec4(0.4, 0.4, 0.5, 1.0);
    }
"""


def s_to_time_string(time_s):
    time_string = f"{time_s}s"
    return time_string


class SwipePlotExploreCanvas(app.Canvas):
    """ Class that handles drawing ExG channels to a vispy canvas. Data that doesn't change every draw call (i.e.
    paddings, number of plots to draw, duration to visualise, scales etc.) is set at initialisation, data that changes
    every draw call is retrieved and set in the on_timer function.
    """
    def __init__(self, explore_data_handler, y_scale=100, x_scale=10):
        super(SwipePlotExploreCanvas, self).__init__(autoswap=False)
        #self.measure_fps()

        self.is_swipe_plot = True
        self.is_active = False

        self.current_second = 0

        self.timestamp_scale = 10000

        self.background_colour = (0.0, 0.0, 0.0, 1.0)
        self.line_colour = (0.6, 0.6, 0.8, 1.0)
        self.line_colour_highlighted = (0.8, 0.2, 0.2, 1.0)
        self.text_color = self.line_colour
        self.font_size = 6

        self.vertical_padding = 0.0
        self.top_padding = 0.0
        self.bottom_padding = 0.1

        self.horizontal_padding = 0.0
        self.left_padding = 0.1
        self.right_padding = 0.0

        self.half_tick_length = 0.01

        self.vertical_marker_space = 8  # number of vertical slots for marker labels

        self.max_visible_plots = 8  # maximum number of channels visible on the screen
        self.channel_mask = []
        self.currently_visible_plots = []
        # If channels are en- or disabled, the setting has to be synced between on_timer and on_draw, hence the flag
        self.update_visible_plots_in_timer = False
        self.plot_offset = 0  # determines index of the top channel to draw

        self.duration = x_scale  # in s
        self.y_range = y_scale * 2  # range for y in uV
        self.x_resolution = 1  # x tick resolution in s

        self.min_time_window = 0.5
        self.min_y_scale = 1

        self.scroll_activated_at = -1
        self.translate_back = 0
        self.is_scrolling = False

        # Program for drawing the y- and x-axis (without ticks or labels)
        self.axis_program = gloo.Program()
        self.axis_program.set_shaders(vert=vertex_padding, frag=fragment_axes)
        self.axis_program.bind(gloo.VertexBuffer(self.create_axes()))

        self.axis_program['vertical_padding'] = self.vertical_padding
        self.axis_program['horizontal_padding'] = self.horizontal_padding
        self.axis_program['top_padding'] = self.top_padding
        self.axis_program['left_padding'] = self.left_padding
        self.axis_program['right_padding'] = self.right_padding
        self.axis_program['bottom_padding'] = self.bottom_padding

        # Program for drawing ticks on the x-axis
        self.x_ticks_program = gloo.Program()
        self.x_ticks_program.set_shaders(vert=vertex_default, frag=fragment_axes)
        self.x_ticks_program.bind(gloo.VertexBuffer(self.create_x_ticks()))

        # Initialisation for variables and fields that are unknown until after connecting
        self.channel_labels = None
        self.time_labels = None
        self.marker_labels = None

        self.programs = []

        self.swipe_line_program = None
        self.y_ticks_program = None
        self.marker_program = None

        self.num_plots = 0
        self.x_coords = np.array(0, dtype=np.uint32)
        self.indices = self.x_coords

        self.timer = app.Timer("auto", self.on_timer, start=False)

        self.is_visible = False

        self.explore_data_handler = explore_data_handler
        if self.explore_data_handler.is_connected:
            self.setup_programs_and_plots()
            self.timer.start()

    def on_connected(self):
        self.setup_programs_and_plots()
        self.timer.start()

    def on_disconnected(self):
        self.clear_programs_and_plots()
        self.timer.stop()
        self.is_visible = False

    def setup_programs_and_plots(self):
        self.num_plots = self.explore_data_handler.get_num_plots()
        self.channel_mask = self.explore_data_handler.get_channel_mask()

        # ***** PROGRAM FOR MARKERS *****
        self.marker_program = gloo.Program()
        self.marker_program.set_shaders(vert=vertex_vertical_line, frag=frag_explore_marker)
        self.marker_program['x_length'] = self.duration
        self.marker_program['horizontal_padding'] = self.horizontal_padding
        self.marker_program['left_padding'] = self.left_padding
        self.marker_program['right_padding'] = self.right_padding

        # ***** PROGRAM FOR CURRENT POSITION LINE (LATEST VALUE) *****
        self.swipe_line_program = gloo.Program(vert=vertex_vertical_line, frag=fragment_explore_swipe_line)
        self.swipe_line_program['x_length'] = self.duration
        self.swipe_line_program['horizontal_padding'] = self.horizontal_padding
        self.swipe_line_program['left_padding'] = self.left_padding
        self.swipe_line_program['right_padding'] = self.right_padding
        self.swipe_line_program['is_scrolling'] = False
        self.swipe_line_program['x_min'] = 0.0

        # ***** PROGRAM FOR Y TICKS *****
        self.y_ticks_program = gloo.Program()
        self.y_ticks_program.set_shaders(vert=vertex_padding, frag=fragment_axes)
        self.y_ticks_program.bind(gloo.VertexBuffer(self.create_y_ticks()))
        self.y_ticks_program['vertical_padding'] = self.vertical_padding
        self.y_ticks_program['horizontal_padding'] = self.horizontal_padding
        self.y_ticks_program['left_padding'] = self.left_padding
        self.y_ticks_program['right_padding'] = self.right_padding
        self.y_ticks_program['top_padding'] = self.top_padding
        self.y_ticks_program['bottom_padding'] = self.bottom_padding

        # Indices for the index buffer ([0, 1, ..., num_values])
        self.x_coords = np.arange(self.duration * self.explore_data_handler.current_sr).astype(np.uint32)
        self.indices = self.x_coords

        self.channel_labels = self.create_y_labels()
        self.time_labels = self.create_x_labels()
        self.marker_labels = self.create_marker_labels()

        # ***** PROGRAMS FOR EXG PLOTTING *****
        for i in range(self.num_plots):
            self.programs.append(gloo.Program(vert=vertex_channel, frag=fragment_explore_swipe))

            self.programs[i]['line_colour'] = self.line_colour
            self.programs[i]['line_colour_highlighted'] = self.line_colour_highlighted
            self.programs[i]['y_range'] = self.y_range
            self.programs[i]['is_scrolling'] = False
            self.programs[i]['is_swipe_plot'] = self.is_swipe_plot
            self.programs[i]['x_length'] = self.duration
            self.programs[i]['x_min'] = 0
            self.programs[i]['vertical_padding'] = self.vertical_padding
            self.programs[i]['horizontal_padding'] = self.horizontal_padding
            self.programs[i]['left_padding'] = self.left_padding
            self.programs[i]['right_padding'] = self.right_padding
            self.programs[i]['top_padding'] = self.top_padding
            self.programs[i]['bottom_padding'] = self.bottom_padding
            self.programs[i]['plot_index'] = i
            self.programs[i]['baseline'] = 0
            self.programs[i]['num_plots'] = self.num_plots

        self.update_visible_plots()

    def clear_programs_and_plots(self):
        self.num_plots = 0
        self.x_coords = np.array(0, dtype=np.uint32)
        self.channel_labels = None
        self.swipe_line_program = None
        self.y_ticks_program = None
        self.programs = []

    def set_y_scale(self, y_scale):
        """Sets the y scale to be used for drawing the plots.
        :param y_scale: The new y scale in uV, the new plot range will be between -y_scale and +y_scale
        """
        if y_scale < self.min_y_scale:
            return
        self.y_range = 2 * y_scale
        for i in range(self.num_plots):
            self.programs[i]['y_range'] = self.y_range

    def set_x_scale(self, x_scale):
        """Sets the x scale (time axis) to be used for drawing the plots.
        :param x_scale: The new time window in seconds to be visualised on screen.
        """
        if x_scale < self.min_time_window:
            return
        self.duration = x_scale
        self.x_coords = np.arange(self.duration * self.explore_data_handler.current_sr).astype(np.uint32)
        for i in range(self.num_plots):
            self.programs[i]['x_length'] = self.duration
        self.marker_program['x_range'] = self.duration
        self.swipe_line_program['x_length'] = self.duration
        #self.update_x_positions()

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)

    def on_draw(self, event):
        """Draws all available programs and visuals.
        """
        gloo.clear(self.background_colour)
        if not self.is_visible or not self.is_active:
            return
        self.axis_program.draw('lines')
        if self.y_ticks_program is not None:
            self.y_ticks_program.draw('lines')
        self.x_ticks_program.draw('lines')
        for i in range(len(self.currently_visible_plots)):
            if self.currently_visible_plots[i] == 0:
                continue
            if self.is_scrolling or not self.is_swipe_plot:
                self.programs[i].draw('line_strip')
            else:
                # TODO: figure out an *easy* way to cut the lines in two if there is a gap in the middle of the graph
                # Ideas:
                # * primitive restart (not sure how to enable this)
                # * two programs per line?! (one program per line is already unnecessary)
                # * use "lines" not "line_strip" (needs an index buffer of twice the current size
                # * use gl.drawArrays twice and specify offset (need to rewrite some stuff for this)
                # most efficient (probably):
                #  - use one buffer for all lines
                #  - then draw using an index buffer that reflects this or multiple draw calls with offsets
                self.programs[i].draw('line_strip', self.indices)
        self.marker_program.draw('lines')
        self.marker_labels.draw()
        if not self.is_scrolling and self.swipe_line_program is not None and self.is_swipe_plot:
            self.swipe_line_program.draw('lines')
        if self.channel_labels is not None:
            self.channel_labels.draw()
        if self.time_labels is not None:
            self.time_labels.draw()
        self.swap_buffers()  # swaps the buffer we just drew every program/visual to with the current buffer

    def on_mouse_wheel(self, event):
        """Handles scrolling back in time when the mouse wheel is activated on the canvas.
        """
        channel = 0
        self.translate_back += int(event.delta[1] * 100)
        if self.scroll_activated_at >= 0:
            self.translate_back += self.explore_data_handler.get_distance(channel, self.scroll_activated_at)
        self.translate_back = min(self.translate_back,
                                  self.explore_data_handler.get_length(
                                      channel) - self.duration * self.explore_data_handler.current_sr)
        self.translate_back = max(self.translate_back, 0)

        if self.translate_back == 0:
            self.scroll_activated_at = -1
        else:
            self.scroll_activated_at = self.explore_data_handler.get_last_index(channel)

    def on_timer(self, event):
        """Handles all buffer and value updates for the next draw call and calls update on the canvas at the end.
        """
        if not self.is_active:
            return

        # Determine whether we are scrolling and how far away from the last ExG index we are
        additional_offset = 0
        if self.scroll_activated_at >= 0:
            additional_offset = self.explore_data_handler.get_distance(0, self.scroll_activated_at)

        self.is_scrolling = (additional_offset + self.translate_back) > 0

        # Request the buffer views for all channels (y) and the timestamps (x) from the data handler
        y, x, current_index = \
            self.explore_data_handler.get_all_channels_and_time(self.duration,
                                                                offset=self.translate_back + additional_offset)

        # Determine the first actual timestamp for plotting according to the latest timestamp and cut off older values
        start_ts = x[-1] - self.duration
        first_ts = np.searchsorted(x, start_ts)  # assume the same ts index for all channels
        x = x[first_ts:]

        # If we are drawing a swipe plot, determine the index of the current timestamp for rolling the index buffer
        # accordingly
        if self.is_swipe_plot:
            midpoint_ts = (x[-1] // self.duration) * self.duration
            midpoint_index = np.searchsorted(x, midpoint_ts)
            self.indices = gloo.IndexBuffer(np.roll(self.x_coords[:len(x)], len(x) - midpoint_index))

        # Create the vertex buffer for the timestamps
        vbo = gloo.VertexBuffer(x)

        # Update the plots if an update was requested from outside (i.e. changed settings to disable a channel etc.)
        if self.update_visible_plots_in_timer:
            self.update_visible_plots()
            self.update_visible_plots_in_timer = False

        # Bind values that (might) have changed to the plot programs
        index = 0
        for i in range(len(self.currently_visible_plots)):
            if self.currently_visible_plots[i] == 1:
                self.programs[i]['is_scrolling'] = self.is_scrolling
                self.programs[i]['pos_x'] = vbo
                self.programs[i]['x_min'] = x[0]
                self.programs[i]['pos_y'] = gloo.VertexBuffer(y[i][first_ts:])
                index += 1

        self.update_x_axis(x[0], x[-1])

        # Get marker data, slice it to only include the visible time range and create vertices accordingly
        marker_labels, timestamps_markers = self.explore_data_handler.get_markers()
        start_index, stop_index = np.searchsorted(timestamps_markers, [x[0], x[-1]])
        found_timestamps = timestamps_markers[start_index:stop_index]
        timestamp_coordinates = []
        for t in found_timestamps:
            timestamp_coordinates.append([t, 1.0])
            timestamp_coordinates.append([t, -1.0])
        timestamp_buffer = np.zeros(len(timestamp_coordinates), [('pos', np.float32, 2)])
        if len(timestamp_coordinates) > 0:
            timestamp_buffer['pos'] = timestamp_coordinates
        self.update_marker_labels(marker_labels[start_index:stop_index], found_timestamps, x[0],
                                  start_index=start_index)

        # Bind the marker coordinates and additionally needed info to the marker program
        self.marker_program.bind(gloo.VertexBuffer(timestamp_buffer))
        self.marker_program['is_scrolling'] = self.is_scrolling
        self.marker_program['x_min'] = x[0]

        # Create and bind the swipe line coordinates to the swipe line buffer
        swipe_line_buffer = np.zeros(2, [('pos', np.float32, 2)])
        swipe_line_buffer['pos'] = [[x[-1], 1.0], [x[-1], -1.0]]
        self.swipe_line_program.bind(gloo.VertexBuffer(swipe_line_buffer))

        if not self.is_visible:
            self.is_visible = True
            self.show()
        if self.is_active:
            self.update()

    def create_axes(self):
        axes_positions = np.array([
            # Axes
            [0.0, 1.0],  # y-axis (top)
            [0.0, 0.0],  # y-axis (bottom)
            [0.0, 0.0],  # x-axis (left)
            [1.0, 0.0]  # x-axis (right)
        ])
        axes = np.zeros(4, [('pos', np.float32, 2)])
        axes['pos'] = axes_positions
        return axes

    def create_y_ticks(self):
        """Returns coordinates for the y tick vertices according the number of plots to draw and length of the tick.
        """
        ticks_y_positions = []
        offset = 1.0 / self.num_plots
        for i in range(self.num_plots):
            y = ((self.num_plots - i) - 0.5) * offset
            ticks_y_positions.append([-self.half_tick_length, y])
            ticks_y_positions.append([self.half_tick_length, y])
        ticks_y = np.zeros(self.num_plots * 2, [('pos', np.float32, 2)])
        ticks_y['pos'] = ticks_y_positions
        return ticks_y

    def create_x_ticks(self):
        """Returns coordinates for the initial x tick vertices according to tick resolution, duration and tick length.
        """
        ticks_x_positions = []
        for i in range(self.duration // self.x_resolution + 1):
            x = i * self.x_resolution / self.duration
            ticks_x_positions.append([x, -self.half_tick_length])
            ticks_x_positions.append([x, self.half_tick_length])
        ticks_x = np.zeros((self.duration // self.x_resolution + 1) * 2, [('pos', np.float32, 2)])
        ticks_x['pos'] = ticks_x_positions
        return ticks_x

    def create_y_labels(self):
        """Returns the text visual holding channel labels for visible channels as well as their positions.
        """
        channel_text = []
        channel_positions = []
        tr_sys = visuals.transforms.TransformSystem()
        tr_sys.configure(canvas=self)
        # This uses the same y position calculation as the y ticks
        y_range = 2.0 - 2.0 * self.vertical_padding - self.top_padding - self.bottom_padding
        offset = 1.0 / self.num_plots
        x = self.left_padding + self.horizontal_padding - 2 * self.half_tick_length - 1
        for i in range(self.num_plots):
            y = ((self.num_plots - i) - 0.5) * offset
            y = y * y_range - 1.0 + self.bottom_padding + self.vertical_padding
            channel_text.append(f"ch{i + 1}")
            channel_positions.append((x, y))
        channel_labels = visuals.TextVisual(channel_text, color=self.text_color, pos=channel_positions,
                                            font_size=self.font_size, anchor_y='center', anchor_x='right')
        channel_labels.transforms = tr_sys
        return channel_labels

    def create_x_labels(self):
        """Returns the text visual holding time labels for the ticks on the x-axis as well as their positions.
        """
        start = (self.current_second // self.duration) * self.duration
        time_text = []
        time_positions = []
        tr_sys = visuals.transforms.TransformSystem()
        tr_sys.configure(canvas=self)
        y = self.bottom_padding + self.vertical_padding - 2 * self.half_tick_length - 1
        x_range = 2 - 2 * self.horizontal_padding - self.left_padding - self.right_padding
        for i in range(self.duration // self.x_resolution + 1):
            x = (i * self.x_resolution / self.duration) * x_range - 1 + self.left_padding + self.horizontal_padding
            t = start + i
            if t > self.current_second > self.duration:
                t -= self.duration
            time_text.append(s_to_time_string(t))
            time_positions.append((x, y))
        time_labels = visuals.TextVisual(time_text, color=self.text_color, pos=time_positions,
                                         font_size=self.font_size, anchor_y='bottom', anchor_x='right')
        time_labels.transforms = tr_sys
        return time_labels

    def create_marker_labels(self):
        """Returns an (empty) text visual to be used for the marker labels
        """
        tr_sys = visuals.transforms.TransformSystem()
        tr_sys.configure(canvas=self)
        marker_labels = visuals.TextVisual(color=self.text_color, font_size=self.font_size, anchor_y='bottom', anchor_x='left')
        marker_labels.transforms = tr_sys
        return marker_labels

    def update_marker_labels(self, labels, pos_buffer, start, start_index=0):
        """Writes the marker labels for the currently visible markers as well as their positions to the marker_labels
        text visual.
        """
        marker_labels_pos = []
        if len(labels) != len(pos_buffer):
            raise ValueError(f"Number of marker labels doesn't equal number of positions:\n"
                             f"#labels: {len(labels)}, #pos_buffer: {pos_buffer}")
        if len(labels) == 0:
            self.marker_labels.text = None
        else:
            y_range = 2 - 2 * self.vertical_padding - self.top_padding - self.bottom_padding - 0.1
            y_top = 1 - self.vertical_padding - self.top_padding
            for i in range(len(labels)):
                pos_x = self.time_to_x_position(pos_buffer[i], start)
                pos_y = y_top - (((start_index + i) % self.vertical_marker_space) / self.vertical_marker_space) * y_range - 0.1
                marker_labels_pos.append((pos_x, pos_y))
            self.marker_labels.text = labels
            self.marker_labels.pos = marker_labels_pos

    def update_x_axis(self, start, stop):
        """Binds new x-axis tick coordinates to the x_ticks_pogram according to first and last timestamp in the plot.
        Additionally, the tick labels and their positions are written to the time_labels text visual.
        """
        time_text = []
        time_positions = []
        tick_coordinates = []
        y = self.bottom_padding + self.vertical_padding - 2 * self.half_tick_length - 1
        if math.fmod(start, self.x_resolution) < 0.001:
            current_tick = round(start, 3)
        else:
            current_tick = round(start // self.x_resolution * self.x_resolution + self.x_resolution, 3)

        while current_tick <= stop:
            tick_x = self.time_to_x_position(current_tick, start)
            time_text.append(s_to_time_string(current_tick))
            time_positions.append((tick_x, y))
            tick_coordinates.append([tick_x, y+self.half_tick_length])
            tick_coordinates.append([tick_x, y+3*self.half_tick_length])
            current_tick += self.x_resolution

        ticks_x = np.zeros(len(tick_coordinates), [('pos', np.float32, 2)])
        ticks_x['pos'] = tick_coordinates
        self.x_ticks_program.bind(gloo.VertexBuffer(ticks_x))
        self.time_labels.text = time_text
        self.time_labels.pos = time_positions

    def update_y_axis(self):
        """Binds the y-axis tick coordinates to y_ticks_program according to the number of visible plots and tick length.
        """
        ticks_y_positions = []
        num_plots = sum(self.currently_visible_plots)
        offset = 1.0 / num_plots
        for i in range(num_plots):
            y = ((num_plots - i) - 0.5) * offset
            ticks_y_positions.append([-self.half_tick_length, y])
            ticks_y_positions.append([self.half_tick_length, y])
        ticks_y = np.zeros(num_plots * 2, [('pos', np.float32, 2)])
        ticks_y['pos'] = ticks_y_positions
        self.y_ticks_program.bind(gloo.VertexBuffer(ticks_y))

    def update_y_labels(self):
        """Writes the labels for the currently visible channels as well as their positions to the channel_labels text
        visual.
        """
        channel_labels_text = []
        channel_labels_pos = []
        y_range = 2.0 - 2.0 * self.vertical_padding - self.top_padding - self.bottom_padding
        num_plots = sum(self.currently_visible_plots)
        offset = 1.0 / num_plots
        x = self.left_padding + self.horizontal_padding - 2 * self.half_tick_length - 1

        iterator = 0
        for i in range(len(self.currently_visible_plots)):
            if self.currently_visible_plots[i] == 1:
                channel_labels_text.append(f"ch{i+1}")
                y = ((num_plots - iterator) - 0.5) * offset
                y = y * y_range - 1.0 + self.bottom_padding + self.vertical_padding
                channel_labels_pos.append((x, y))
                iterator += 1
        self.channel_labels.text = channel_labels_text
        self.channel_labels.pos = channel_labels_pos

    def time_to_x_position(self, time, start):
        """Calculates the x coordinate (in screen space) from a given time in seconds, the first timestamp in the plot
        and the information whether the plot is currently in swipe mode.
        """
        length = self.duration
        if not self.is_scrolling and self.is_swipe_plot:
            x = math.fmod(time, length)
        else:
            x = time - start
        available_x_range = 2 - 2*self.horizontal_padding - self.left_padding - self.right_padding
        x = x/length * available_x_range - 1 + self.left_padding + self.horizontal_padding
        return x

    def update_programs(self):
        iterator = 0
        for i in range(len(self.currently_visible_plots)):
            if self.currently_visible_plots[i] == 1:
                self.programs[i]['plot_index'] = iterator
                self.programs[i]['num_plots'] = sum(self.currently_visible_plots)

                iterator += 1

    def change_settings(self):
        """Method called from outside to inform the canvas of changes concerning the Explore device.
        """
        self.change_channel_mask()

    def change_channel_mask(self):
        """Sets the plot's channel mask and a flag telling the timer function to update plots.
        """
        self.channel_mask = self.explore_data_handler.get_channel_mask()
        self.update_visible_plots_in_timer = True

    def set_currently_visible_plots(self):
        self.currently_visible_plots = [0 for _ in self.channel_mask]
        iterator = 0
        for i in range(len(self.channel_mask)):
            if self.channel_mask[i] == 1:
                if self.plot_offset <= iterator < (self.max_visible_plots + self.plot_offset):
                    self.currently_visible_plots[i] = 1
                iterator += 1

    def set_max_visible_plots(self, new_max):
        self.max_visible_plots = new_max
        self.update_visible_plots_in_timer = True

    def update_visible_plots(self):
        self.set_currently_visible_plots()
        if sum(self.currently_visible_plots) >= 1:
            self.update_programs()
            self.update_y_axis()
            self.update_y_labels()

    def set_plot_offset(self, new_offset):
        self.plot_offset = max(min(new_offset, sum(self.channel_mask)-self.max_visible_plots), 0)
        self.update_visible_plots_in_timer = True

    def set_active(self, is_active):
        self.is_active = is_active


class EXGPlotVispy:
    """This class is a helper class to communicate signals and inputs from the Qt environment to the data handler
    (ExploreDataHandlerCircularBuffer) and the vispy canvas (SwipePlotExploreCanvas).
    """
    def __init__(self, ui, explore_interface) -> None:
        self.ui = ui
        self.explore_handler = ExploreDataHandlerCircularBuffer(dev_name=None, interface=explore_interface)
        self.c = SwipePlotExploreCanvas(self.explore_handler, y_scale=Settings.DEFAULT_SCALE,
                                        x_scale=Settings.WIN_LENGTH)
        self.ui.horizontalLayout_21.addWidget(self.c.native)
        self.vertical_scroll_bar = QScrollBar()
        self.vertical_scroll_bar.setMinimum(0)
        self.set_vertical_scrollbar()
        self.ui.horizontalLayout_21.addWidget(self.vertical_scroll_bar)

    def set_active(self, is_active):
        """This sets the canvas as inactive or active. The timer inside the function doesn't update any values or
        buffers if it is inactive, which prevents unnecessary buffer binding and slow-downs when the canvas isn't
        visible.
        """
        self.c.set_active(is_active)

    def on_connected(self):
        self.explore_handler.on_connected()
        self.c.on_connected()
        self.set_vertical_scrollbar()

    def on_disconnected(self):
        self.explore_handler.on_disconnected()
        self.c.on_disconnected()

    def change_settings(self):
        self.explore_handler.change_settings()
        self.c.change_settings()
        self.set_vertical_scrollbar()

    def change_scale(self, new_val):
        self.c.set_y_scale(Settings.SCALE_MENU_VISPY[new_val])

    def change_timescale(self, new_val):
        self.c.set_x_scale(int(Settings.TIME_RANGE_MENU[new_val]))

    def set_plot_offset(self):
        """Updates the plot offset with the current value of the scrollbar when it is activated.
        """
        self.c.set_plot_offset(self.vertical_scroll_bar.value())

    def set_vertical_scrollbar(self):
        """Sets maximum and bar size of the vertical scrollbar for the plots.
        """
        plot_count = sum(self.explore_handler.get_channel_mask())
        page_step = min(self.c.max_visible_plots, plot_count)
        maximum = max(0, plot_count-self.c.max_visible_plots)
        self.vertical_scroll_bar.setPageStep(page_step)
        self.vertical_scroll_bar.setMaximum(maximum)

    def setup_ui_connections(self):
        self.ui.value_yAxis.currentTextChanged.connect(self.change_scale)
        self.ui.value_timeScale.currentTextChanged.connect(self.change_timescale)
        self.vertical_scroll_bar.valueChanged.connect(self.set_plot_offset)