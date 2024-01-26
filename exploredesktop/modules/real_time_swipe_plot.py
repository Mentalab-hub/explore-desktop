import sys
import numpy as np

import explorepy
from explorepy.stream_processor import TOPICS
from explorepy.settings_manager import SettingsManager
from vispy import app
from vispy import gloo
from vispy import visuals
from vispy.util import keys

from exploredesktop.modules import Settings

class CircularBufferPadded:
    def __init__(self, max_length, dtype=np.float32):
        self.max_length = max_length
        self.buffer = np.empty(self.max_length * 2, dtype=dtype)
        self.length = 0
        self.index_first = 0
        self.index_last = 0

    def view_end(self, length, offset=0):
        # Rollover for reading isn't possible, so this doesn't have to be taken into account
        if (length + offset) < self.length:
            return self.buffer[self.index_last - length - offset:self.index_last - offset]
        elif length < self.length:
            return self.buffer[self.index_first:self.index_first + length]
        else:
            return self.buffer[self.index_first:self.index_last]

    def insert_iterating(self, chunk):
        # TODO: Slicing the chunk correctly may be faster than enumerating but increases complexity
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
    _valid_channels = [4, 8, 16, 32]

    def __init__(self, dev_name, interface=None):
        self.dev_name = dev_name
        self.interface = interface
        self.is_connected = False
        if self.interface:
            self.explore_device = interface
            self.dev_name = self.explore_device.device_name
        else:
            self.explore_device = explorepy.Explore()
            self.explore_device.connect(self.dev_name)

        self.current_sr = 0
        self.max_channels = 0
        self.packet_length = 0
        self.max_length = 0
        self.max_length_markers = 0

        self.max_duration = 60 * 60  # in seconds

        self.moving_average_window = 200
        self.time_offset = 0

        self.channels = {}
        self.baselines = {}

        self.timestamps = None
        self.markers = None
        self.timestamps_markers = None

        self.callbacks = {
            # TOPICS.raw_ExG: [self.handle_exg],
            TOPICS.filtered_ExG: [self.handle_exg],
            TOPICS.raw_orn: [],
            TOPICS.marker: [self.handle_marker]
        }

        if self.explore_device.is_connected:
            self.is_connected = True
            self.dev_name = self.explore_device.device_name
            self.setup_buffers()
            self.subscribe_packet_callbacks()

    def on_connected(self):
        self.is_connected = True
        self.dev_name = self.explore_device.device_name
        self.setup_buffers()
        self.subscribe_packet_callbacks()

    def on_disconnected(self):
        self.is_connected = False
        self.dev_name = None
        self.unsubscribe_packet_callbacks()
        self.clear_buffers()

    def clear_buffers(self):
        self.channels = {}
        self.baselines = {}

        self.timestamps = None
        self.markers = None
        self.timestamps_markers = None

        self.current_sr = 0
        self.max_channels = 0
        self.packet_length = 0
        self.max_length = 0
        self.max_length_markers = 0

    def setup_buffers(self):
        self.current_sr = int(self.explore_device.stream_processor.device_info['sampling_rate'])
        self.max_channels = SettingsManager(self.dev_name).get_channel_count()
        self.packet_length = self.get_packet_length()  # requires max_channels to be set

        self.max_length = self.current_sr * self.max_duration
        self.max_length_markers = self.max_duration * 30  # 30 markers per second

        for i in range(self.max_channels):
            self.channels[i] = CircularBufferPadded(self.max_length, dtype=np.float32)
            self.baselines[i] = 0

        self.timestamps = CircularBufferPadded(self.max_length, dtype=np.float32)
        self.markers = CircularBufferPadded(self.max_length_markers, dtype='<U10')
        self.timestamps_markers = CircularBufferPadded(self.max_length_markers, dtype=np.float32)

    def subscribe_packet_callbacks(self):
        self.sub_unsub_packet_callbacks(sp_callback=self.explore_device.stream_processor.subscribe)

    def unsubscribe_packet_callbacks(self):
        self.sub_unsub_packet_callbacks(sp_callback=self.explore_device.stream_processor.unsubscribe)

    def sub_unsub_packet_callbacks(self, sp_callback):
        for topic, callbacks in self.callbacks.items():
            for callback in callbacks:
                sp_callback(callback=callback, topic=topic)

    def handle_exg(self, packet):
        timestamps, channels = packet.get_data(exg_fs=self.current_sr)
        if self.time_offset == 0:
            self.time_offset = timestamps[0]

        timestamps = timestamps - self.time_offset
        for i in range(len(channels)):
            self.baselines[i] = self.baselines[i] * 0.8 + sum(channels[i]) / len(channels[i]) * 0.2
            self.channels[i].insert_iterating(channels[i])
        self.timestamps.insert_iterating(timestamps)

    def handle_orn(self, packet):
        raise NotImplementedError

    def handle_marker(self, packet):
        timestamp, marker_string = packet.get_data()
        print(f"Got marker with ts: {timestamp}")
        if self.time_offset == 0:
            self.time_offset = timestamp
        timestamp = timestamp - self.time_offset
        print(f"Timestamp with offset subtracted is: {timestamp}")
        if timestamp <= self.timestamps_markers.view_end(0):
            print(f"Marker timestamp in the past, last: {self.markers.view_end(0)}, new: {timestamp}")
        else:
            self.markers.insert_iterating(marker_string)
            self.timestamps_markers.insert_iterating(timestamp)

    def get_channel_and_time(self, channel, duration, offset=0):
        # TODO: write a version that gets *all* channels*
        num_values = duration * self.current_sr
        if offset == 0:
            current_index = self.channels[channel].get_last_index() % num_values
        else:
            current_index = 0
        return self.channels[channel].view_end(num_values, offset=offset), \
               self.timestamps.view_end(num_values, offset=offset), \
               self.baselines[channel], \
               current_index

    def get_all_channels_and_time(self, duration, offset=0):
        num_values = duration * self.current_sr
        all_channels = [None for _ in range(len(self.channels))]
        if offset == 0:
            current_index = self.channels[0].get_last_index() % num_values
        else:
            current_index = 0

        timestamps = self.timestamps.view_end(num_values, offset=offset)
        for i in range(len(self.channels)):
            all_channels[i] = self.channels[i].view_end(num_values, offset=offset)

        return all_channels, timestamps, self.baselines, current_index

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
        return self.markers.view_end(self.max_length_markers), self.timestamps_markers.view_end(self.max_length_markers)

    def get_current_sr(self):
        return self.current_sr

    def get_num_plots(self):
        return self.max_channels


vertex_channel = """
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
        gl_Position = vec4(x, y, 0.0, 1.0);
    }
"""

fragment_explore_swipe = """
    varying vec4 v_col;

    void main() {
        gl_FragColor = v_col;
    }
    """

frag_explore_marker = """
#version 120

void main()
{
    gl_FragColor = vec4(0.9, 0.0, 0.0, 1.0);
}
"""


vertex_vertical_line = """
    #version 120

    uniform float x_length;
    uniform float horizontal_padding;
    uniform float left_padding;
    uniform float right_padding;

    attribute vec2 pos;
    void main(void) {
        float x = mod(pos.x, x_length);
        float available_x_range = 2.0 - 2.0*horizontal_padding - left_padding - right_padding;
        x = (x / x_length) * available_x_range - 1.0 + left_padding + horizontal_padding;

        gl_Position = vec4(x, pos.y, 0.0, 1.0);
    }
"""

fragment_explore_swipe_line = """
#version 120

void main()
{
    gl_FragColor = vec4(0.3, 0.3, 0.4, 1.0);
}
"""

fragment_axes = """
#version 120

void main()
{
    gl_FragColor = vec4(0.4, 0.4, 0.5, 1.0);
}
"""

vertex_padding = """
#version 120
in vec2 pos;
in float vertical_padding;
in float horizontal_padding;
in float top_padding;
in float bottom_padding;
in float left_padding;
in float right_padding;

void main(void) {
    float available_y_range = 2.0 - 2.0*vertical_padding - top_padding - bottom_padding;
    float available_x_range = 2.0 - 2.0*horizontal_padding - left_padding - right_padding;
    float y = pos.y * available_y_range - 1.0 + bottom_padding + vertical_padding;
    float x = pos.x * available_x_range - 1.0 + left_padding + horizontal_padding;
    gl_Position = vec4(x, y, 0.0, 1.0);
}
"""


def s_to_time_string(time_s):
    time_string = f"{time_s}s"
    return time_string


class SwipePlotExploreCanvas(app.Canvas):
    def __init__(self, explore_data_handler, y_scale=100, x_scale=10):
        # TODO: Test circular buffer boundaries
        # TODO: Make sure datahandler returns...
        #  * potentially the index buffer
        # TODO: Make code more efficient and readable (i.e. reuse shader if possible)
        super(SwipePlotExploreCanvas, self).__init__(autoswap=False)
        self.measure_fps()

        self.is_swipe_plot = True
        self.is_active = False

        self.ts_last_print = 0

        self.current_second = 0

        self.timestamp_scale = 10000

        self.background_colour = (0.2, 0.2, 0.3, 1.0)
        self.line_colour = (0.6, 0.6, 0.8, 1.0)
        self.line_colour_highlighted = (0.8, 0.2, 0.2, 1.0)
        self.text_color = self.line_colour
        self.font_size = 6

        self.avg_refresh_time = 0
        self.avg_refresh_time_iterator = 0

        self.timer_iterator = 0

        self.duration = x_scale  # in s
        self.y_range = y_scale * 2  # range for y in uV
        self.x_resolution = 1

        self.scroll_activated_at = -1

        self.min_time_window = 0.5
        self.min_y_scale = 1

        self.translate_back = 0
        self.is_scrolling = False

        self.vertical_padding = 0.0
        self.top_padding = 0.0
        self.bottom_padding = 0.1

        self.horizontal_padding = 0.0
        self.left_padding = 0.1
        self.right_padding = 0.0

        self.half_tick_length = 0.01

        self.axis_program = gloo.Program()
        self.axis_program.set_shaders(vert=vertex_padding, frag=fragment_axes)
        self.axis_program.bind(gloo.VertexBuffer(self.create_axes()))

        self.axis_program['vertical_padding'] = self.vertical_padding
        self.axis_program['horizontal_padding'] = self.horizontal_padding
        self.axis_program['top_padding'] = self.top_padding
        self.axis_program['left_padding'] = self.left_padding
        self.axis_program['right_padding'] = self.right_padding
        self.axis_program['bottom_padding'] = self.bottom_padding

        self.x_ticks_program = gloo.Program()
        self.x_ticks_program.set_shaders(vert=vertex_padding, frag=fragment_axes)
        self.x_ticks_program.bind(gloo.VertexBuffer(self.create_x_ticks()))
        self.x_ticks_program['vertical_padding'] = self.vertical_padding
        self.x_ticks_program['horizontal_padding'] = self.horizontal_padding
        self.x_ticks_program['top_padding'] = self.top_padding
        self.x_ticks_program['left_padding'] = self.left_padding
        self.x_ticks_program['right_padding'] = self.right_padding
        self.x_ticks_program['bottom_padding'] = self.bottom_padding

        self.channel_labels = None
        self.time_labels = None

        self.programs = []

        self.swipe_line_program = None
        self.y_ticks_program = None
        self.marker_program = None

        self.timer = app.Timer("auto", self.on_timer, start=False)

        self.is_visible = False

        self.num_plots = 0
        self.x_coords = np.array(0, dtype=np.uint32)
        self.indices = self.x_coords

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

        self.x_coords = np.arange(self.duration * self.explore_data_handler.current_sr).astype(np.uint32)
        self.indices = self.x_coords

        self.channel_labels = self.create_y_labels()
        self.time_labels = self.create_x_labels()

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

    def clear_programs_and_plots(self):
        self.num_plots = 0
        self.x_coords = np.array(0, dtype=np.uint32)
        self.channel_labels = None
        self.swipe_line_program = None
        self.y_ticks_program = None
        self.programs = []

    def set_y_scale(self, y_scale):
        '''
        Sets the y scale to be used for drawing the plots.
        :param y_scale: The new y scale in uV, the new plot range will be between -y_scale and +y_scale
        '''
        if y_scale < self.min_y_scale:
            return
        self.y_range = 2 * y_scale
        print(f"New y range is {self.y_range}")
        for i in range(self.num_plots):
            self.programs[i]['y_range'] = self.y_range

    def set_x_scale(self, x_scale):
        '''
        Sets the x scale (time axis) to be used for drawing the plots.
        :param x_scale: The new time window in seconds to be visualised on screen.
        '''
        if x_scale < self.min_time_window:
            return
        self.duration = x_scale
        print(f"New x scale is {self.duration}")
        self.x_coords = np.arange(self.duration * self.explore_data_handler.current_sr).astype(np.uint32)
        for i in range(self.num_plots):
            self.programs[i]['x_length'] = self.duration
        self.marker_program['x_range'] = self.duration
        self.swipe_line_program['x_length'] = self.duration
        self.x_ticks_program.bind(gloo.VertexBuffer(self.create_x_ticks()))
        self.update_x_positions()

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)

    def on_draw(self, event):
        gloo.clear(self.background_colour)
        if not self.is_visible or not self.is_active:
            return
        self.axis_program.draw('lines')
        if self.y_ticks_program is not None:
            self.y_ticks_program.draw('lines')
        if not self.is_scrolling:
            self.x_ticks_program.draw('lines')
        for i in range(self.num_plots):
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
        if not self.is_scrolling and self.swipe_line_program is not None and self.is_swipe_plot:
            self.swipe_line_program.draw('lines')
        if self.channel_labels:
            self.channel_labels.draw()
        if self.time_labels and not self.is_scrolling:
            self.time_labels.draw()
        self.swap_buffers()

    def on_key_press(self, event):
        # TODO: decide what to do with this
        # return  # switched off for now
        if event.key == keys.LEFT:
            self.set_x_scale(x_scale=(self.duration + 5))
        elif event.key == keys.RIGHT:
            self.set_x_scale(x_scale=(self.duration - 5))
        elif event.key == keys.UP:
            self.set_y_scale(y_scale=((self.y_range / 2) + 10))
        elif event.key == keys.DOWN:
            self.set_y_scale(y_scale=((self.y_range / 2) - 10))
        self.update()

    def on_mouse_wheel(self, event):
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
        if not self.is_active:
            return
        additional_offset = 0
        if self.scroll_activated_at >= 0:
            additional_offset = self.explore_data_handler.get_distance(0, self.scroll_activated_at)

        self.is_scrolling = (additional_offset + self.translate_back) > 0

        y, x, baseline, current_index = \
            self.explore_data_handler.get_all_channels_and_time(self.duration,
                                                                offset=self.translate_back + additional_offset)
        start_ts = x[-1] - self.duration
        first_ts = np.searchsorted(x, start_ts)  # assume the same ts index for all channels
        x = x[first_ts:]

        if self.is_swipe_plot:
            midpoint_ts = (x[-1] // self.duration) * self.duration
            midpoint_index = np.searchsorted(x, midpoint_ts)
            self.indices = gloo.IndexBuffer(np.roll(self.x_coords[:len(x)], len(x) - midpoint_index))

        vbo = gloo.VertexBuffer(x)
        for i in range(len(y)):
            self.programs[i]['is_scrolling'] = self.is_scrolling
            self.programs[i]['pos_x'] = vbo
            self.programs[i]['x_min'] = x[0]
            self.programs[i]['pos_y'] = gloo.VertexBuffer(y[i][first_ts:])

        if int(x[-1]) > self.current_second:
            self.current_second = int(x[-1])
            self.update_x_labels()

        _, timestamps_markers = self.explore_data_handler.get_markers()  # TODO: use the strings (placeholder _ currently)
        start_index, stop_index = np.searchsorted(timestamps_markers, [x[0], x[-1]])
        found_timestamps = timestamps_markers[start_index:stop_index]
        timestamp_coordinates = []
        for t in found_timestamps:
            timestamp_coordinates.append([t, 1.0])
            timestamp_coordinates.append([t, -1.0])
        timestamp_buffer = np.zeros(len(timestamp_coordinates), [('pos', np.float32, 2)])
        if len(timestamp_coordinates) > 0:
            timestamp_buffer['pos'] = timestamp_coordinates
        self.marker_program.bind(gloo.VertexBuffer(timestamp_buffer))
        #self.marker_program['is_scrolling'] = self.is_scrolling
        #self.marker_program['x_min'] = x[0]
        #self.swipe_line_program['pos_x'] = np.array([x[-1]]).astype(np.float32)
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
        ticks_x_positions = []
        for i in range(self.duration // self.x_resolution + 1):
            x = i * self.x_resolution / self.duration
            ticks_x_positions.append([x, -self.half_tick_length])
            ticks_x_positions.append([x, self.half_tick_length])
        ticks_x = np.zeros((self.duration // self.x_resolution + 1) * 2, [('pos', np.float32, 2)])
        ticks_x['pos'] = ticks_x_positions
        return ticks_x

    def create_y_labels(self):
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

    def update_x_labels(self, clear=False):
        if clear:
            self.time_labels.text = []
            return
        time_text = []
        start = (self.current_second // self.duration) * self.duration
        for i in range(self.duration // self.x_resolution + 1):
            t = start + i
            if t > self.current_second > self.duration:
                t -= self.duration
            time_text.append(s_to_time_string(t))
        self.time_labels.text = time_text

    def update_x_positions(self):
        time_positions = []
        y = self.bottom_padding + self.vertical_padding - 2 * self.half_tick_length - 1
        x_range = 2 - 2 * self.horizontal_padding - self.left_padding - self.right_padding
        for i in range(self.duration // self.x_resolution + 1):
            x = (i * self.x_resolution / self.duration) * x_range - 1 + self.left_padding + self.horizontal_padding
            time_positions.append((x, y))
        self.time_labels.pos = time_positions

    def set_active(self, is_active):
        self.is_active = is_active


class EXGPlotVispy:
    def __init__(self, ui, explore_interface) -> None:
        self.ui = ui
        self.explore_handler = ExploreDataHandlerCircularBuffer(dev_name=None, interface=explore_interface)
        self.c = SwipePlotExploreCanvas(self.explore_handler, y_scale=Settings.DEFAULT_SCALE,
                                        x_scale=Settings.WIN_LENGTH)
        self.ui.horizontalLayout_21.addWidget(self.c.native)

    def set_active(self, is_active):
        self.c.set_active(is_active)

    def on_connected(self):
        self.explore_handler.on_connected()
        self.c.on_connected()

    def on_disconnected(self):
        self.explore_handler.on_disconnected()
        self.c.on_disconnected()

    def change_scale(self, new_val):
        self.c.set_y_scale(Settings.SCALE_MENU_VISPY[new_val])

    def change_timescale(self, new_val):
        self.c.set_x_scale(int(Settings.TIME_RANGE_MENU[new_val]))

    def setup_ui_connections(self):
        self.ui.value_yAxis.currentTextChanged.connect(self.change_scale)
        self.ui.value_timeScale.currentTextChanged.connect(self.change_timescale)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        device_name = sys.argv[1]
    else:
        device_name = "Explore_8539"
    explore_handler = ExploreDataHandlerCircularBuffer(dev_name=device_name)
    c = SwipePlotExploreCanvas(explore_data_handler=explore_handler)
    app.run()
