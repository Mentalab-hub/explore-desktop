import sys
import time

import numpy as np

import explorepy
from explorepy.stream_processor import TOPICS
from vispy import app
from vispy import gloo
from vispy.util import keys

gloo.gl.use_gl('gl+')


class CircularBufferPadded:
    def __init__(self, max_length, dtype=np.float32):
        self.max_length = max_length
        self.buffer = np.empty(self.max_length*2, dtype=dtype)
        self.length = 0
        self.index_first = 0
        self.index_last = 0

    def view_end(self, length, offset=0):
        # Rollover for reading isn't possible, so this doesn't have to be taken into account
        if (length + offset) < self.length:
            return self.buffer[self.index_last-length-offset:self.index_last-offset]
        elif length < self.length:
            return self.buffer[self.index_first:self.index_first + length]
        else:
            return self.buffer[self.index_first:self.index_last]

    def insert_iterating(self, chunk):
        # Slicing the chunk correctly may be faster than enumerating but increases complexity
        for it, element in enumerate(chunk):
            loc = (self.index_last + it) % self.max_length
            self.buffer[loc] = element
            self.buffer[loc+self.max_length] = element
        self.length = min(self.length + len(chunk), self.max_length)
        self.index_last = (self.index_last + len(chunk)) % (self.max_length*2)
        self.index_first = (self.index_last - self.length) % (self.max_length*2)
        if self.index_first >= self.index_last:
            self.index_last = (self.index_last + self.max_length) % (self.max_length*2)
            self.index_first = (self.index_first + self.max_length) % (self.max_length*2)

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
    def __init__(self, dev_name):

        self.callbacks = {
            TOPICS.raw_ExG: [self.handle_exg],
            TOPICS.raw_orn: [],
            TOPICS.marker: [self.handle_marker]
        }

        self.dev_name = dev_name
        self.explore_device = explorepy.Explore()
        self.explore_device.connect(self.dev_name)

        self.max_duration = 60 * 60  # in seconds
        self.max_channels = 32  # hardcoded for now

        self.current_sr = 250  # hardcoded for now
        self.packet_length = 4  # hardcoded for now

        self.max_length = self.current_sr * self.max_duration
        self.max_length_markers = self.max_duration * 30

        self.moving_average_window = 200

        self.timestamp_iterator = 0

        self.time_offset = 0

        self.channels = {}
        self.baselines = {}
        for i in range(self.max_channels):
            self.channels[i] = CircularBufferPadded(self.max_length, dtype=np.float32)
            self.baselines[i] = 0
        self.timestamps = CircularBufferPadded(self.max_length, dtype=np.float32)
        self.markers = CircularBufferPadded(self.max_length_markers, dtype='<U10')
        self.timestamps_markers = CircularBufferPadded(self.max_length_markers, dtype=np.float32)

        self.subscribe_packet_callbacks()

    def subscribe_packet_callbacks(self):
        self.sub_unsub_packet_callbacks(sp_callback=self.explore_device.stream_processor.subscribe)

    def unsubscribe_packet_callbacks(self):
        self.sub_unsub_packet_callbacks(sp_callback=self.explore_device.stream_processor.unsubscribe)

    def sub_unsub_packet_callbacks(self, sp_callback):
        for topic, callbacks in self.callbacks.items():
            for callback in callbacks:
                sp_callback(callback=callback, topic=topic)

    def handle_exg(self, packet):
        now = time.time() - self.time_offset
        if self.time_offset == 0:
            self.time_offset = now
            now = 0.0
        for i in range(self.max_channels):
            ch = packet.data[i]
            self.baselines[i] = self.baselines[i] * 0.8 + sum(ch) / len(ch) * 0.2
            self.channels[i].insert_iterating(ch)
        #timestamps = np.linspace(now - 0.012, now, 16)
        timestamps = np.linspace(now - 1/self.current_sr*(self.packet_length-1), now, self.packet_length)
        self.timestamps.insert_iterating(timestamps)

    def handle_orn(self, packet):
        raise NotImplementedError

    def handle_marker(self, packet):
        now = time.time() - self.time_offset
        if self.time_offset == 0:
            self.time_offset = now
            now = 0.0
        timestamp = [now]
        _, marker_string = packet.get_data()
        self.markers.insert_iterating(marker_string)
        self.timestamps_markers.insert_iterating(timestamp)

    def get_channel_and_time(self, channel, duration, offset=0):
        # TODO: write a version that gets *all* channels*
        num_values = duration * self.current_sr
        #current_index = self.channels[channel].get_length() % num_values
        if offset == 0:
            current_index = self.channels[channel].get_last_index() % num_values
        else:
            current_index = 0
        return self.channels[channel].view_end(num_values, offset=offset), \
               self.timestamps.view_end(num_values, offset=offset), \
               self.baselines[channel], \
               current_index

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


vertex_explore_swipe = """
    // Remember: positions (x and y) have to be in [-1;1]

    uniform float vertical_padding;
    
    uniform vec4 line_colour;
    uniform vec4 line_colour_highlighted;

    uniform float plot_index;
    uniform float num_plots;

    uniform float x_length;
    uniform float y_range;

    uniform float baseline;

    attribute float pos_x;
    attribute float pos_y;

    varying vec4 v_col;

    void main() {
        float new_x = (pos_x / x_length) * 2.0 - 1.0;

        float plot_range = (2.0f - 2.0f * vertical_padding) / num_plots;  // height available per plot

        float new_y = pos_y - baseline;
        new_y = new_y / y_range / 2.0f;  // all values in [-1;1], max_y = 1.0, min_y = -1.0

        // Move y coordinate to the correct location according to plot index, amount of plots and vertical padding
        new_y = new_y * plot_range * 0.5f;
        new_y = new_y + 1.0f;
        new_y = new_y - 0.5f * plot_range;
        new_y = new_y - vertical_padding;
        new_y = new_y - plot_index * plot_range;

        v_col = line_colour;
        gl_Position = vec4(new_x, new_y, 0.0, 1.0);
    }
    """

fragment_explore_swipe = """
    varying vec4 v_col;

    void main() {
        gl_FragColor = v_col;
    }
    """

vertex_explore_marker = """
    #version 330

    in float x_range;

    in float pos_x;

    void main(void) {
        float new_x = ((pos_x - (int(pos_x / x_range) * x_range)) / x_range - 0.5f) * 2.0f;
        gl_Position = vec4(new_x, 0.0, 0.0, 1.0);
    }
    """

frag_explore_marker = """
#version 330

out vec4 frag_color;

void main()
{
    gl_FragColor = vec4(0.9, 0.0, 0.0, 1.0);
}
"""

geometry_explore_marker = """
#version 330

layout (points) in;
layout (line_strip, max_vertices=2) out;

void main(void) {
    vec4 p = gl_in[0].gl_Position;

    gl_Position = vec4(p.x, 1.0, 0, 1);
    EmitVertex();
    gl_Position = vec4(p.x, -1.0, 0, 1);
    EmitVertex();
    EndPrimitive();
}
"""

vertex_explore_swipe_line = """
    #version 330

    in float x_range;

    in float pos_x;

    void main(void) {
        //float new_x = ((pos_x - (int(pos_x / x_range) * x_range)) / x_range - 0.5f) * 2.0f;
        float new_x = (pos_x / x_range - 0.5f) * 2.0f;
        gl_Position = vec4(new_x, 0.0, 0.0, 1.0);
    }
    """

fragment_explore_swipe_line = """
#version 330

out vec4 frag_color;

void main()
{
    gl_FragColor = vec4(0.6, 0.6, 0.6, 1.0);
}
"""

geometry_explore_swipe_line = """
#version 330

layout (points) in;
layout (line_strip, max_vertices=2) out;

void main(void) {
    vec4 p = gl_in[0].gl_Position;

    gl_Position = vec4(p.x, 1.0, 0, 1);
    EmitVertex();
    gl_Position = vec4(p.x, -1.0, 0, 1);
    EmitVertex();
    EndPrimitive();
}
"""


class SwipePlotExploreCanvas(app.Canvas):
    def __init__(self, explore_data_handler):
        super(SwipePlotExploreCanvas, self).__init__()

        self.background_colour = (0.2, 0.2, 0.3, 1.0)
        self.line_colour = (0.6, 0.6, 0.8, 1.0)
        self.line_colour_highlighted = (0.8, 0.2, 0.2, 1.0)

        self.avg_refresh_time = 0
        self.avg_refresh_time_iterator = 0

        self.explore_data_handler = explore_data_handler
        self.timer_iterator = 0

        self.duration = 10  # in s
        self.y_range = 200  # range for y in uV

        self.scroll_activated_at = -1

        self.num_plots = self.explore_data_handler.get_num_plots()

        self.min_time_window = 0.5
        self.min_y_scale = 1

        self.translate_back = 0

        self.programs = []
        for i in range(self.num_plots):
            self.programs.append(gloo.Program(vertex_explore_swipe, fragment_explore_swipe))

        self.marker_program = gloo.Program()
        self.marker_program.set_shaders(vert=vertex_explore_marker, frag=frag_explore_marker,
                                        geom=geometry_explore_marker)
        self.marker_program['x_range'] = self.duration

        self.swipe_line_program = gloo.Program()
        self.swipe_line_program.set_shaders(vert=vertex_explore_swipe_line, frag=fragment_explore_swipe_line,
                                            geom=geometry_explore_swipe_line)
        self.swipe_line_program['x_range'] = self.duration * self.explore_data_handler.get_current_sr()

        self.x_coords = np.arange(self.duration * self.explore_data_handler.get_current_sr()).astype(np.float32)

        for i in range(self.num_plots):
            self.programs[i]['line_colour'] = self.line_colour
            self.programs[i]['line_colour_highlighted'] = self.line_colour_highlighted
            self.programs[i]['y_range'] = self.y_range
            self.programs[i]['x_length'] = self.duration * self.explore_data_handler.get_current_sr()
            self.programs[i]['vertical_padding'] = 0.05
            self.programs[i]['plot_index'] = i
            self.programs[i]['num_plots'] = self.num_plots

        self.timer = app.Timer("auto", self.on_timer, start=True)

        self.is_visible = False

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
        self.x_coords = np.arange(self.duration * self.explore_data_handler.current_sr).astype(np.float32)
        for i in range(self.num_plots):
            self.programs[i]['x_length'] = self.duration * self.explore_data_handler.current_sr
        self.marker_program['x_range'] = self.duration
        self.swipe_line_program['x_range'] = self.duration * self.explore_data_handler.current_sr

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)

    def on_draw(self, event):
        gloo.clear(self.background_colour)
        for i in range(self.num_plots):
            self.programs[i].draw('line_strip')
        self.marker_program.draw('points')
        self.swipe_line_program.draw('points')

    def on_key_press(self, event):
        if event.key == keys.LEFT:
            self.set_x_scale(x_scale=(self.duration+5))
        elif event.key == keys.RIGHT:
            self.set_x_scale(x_scale=(self.duration-5))
        elif event.key == keys.UP:
            self.set_y_scale(y_scale=((self.y_range/2)+10))
        elif event.key == keys.DOWN:
            self.set_y_scale(y_scale=((self.y_range/2)-10))
        self.update()

    def on_mouse_wheel(self, event):
        channel = 0
        self.translate_back += int(event.delta[1] * 100)
        if self.scroll_activated_at >= 0:
            self.translate_back += self.explore_data_handler.get_distance(channel, self.scroll_activated_at)
        self.translate_back = min(self.translate_back,
                                  self.explore_data_handler.get_length(channel) - self.duration * self.explore_data_handler.current_sr)
        self.translate_back = max(self.translate_back, 0)
        if self.translate_back == 0:
            self.scroll_activated_at = -1
        else:
            self.scroll_activated_at = self.explore_data_handler.get_last_index(channel)

    def on_timer(self, event):
        for i in range(self.num_plots):
            additional_offset = 0
            if self.scroll_activated_at >= 0:
                additional_offset = self.explore_data_handler.get_distance(0, self.scroll_activated_at)
            y, x, baseline, current_index =\
                self.explore_data_handler.get_channel_and_time(i, self.duration, offset=self.translate_back+additional_offset)
            if len(x) < 2:
                return

            new_y = np.roll(y, current_index)
            if len(x) < self.duration * self.explore_data_handler.current_sr:
                self.programs[i]['pos_x'] = np.arange(len(x)).astype(np.float32)
            else:
                self.programs[i]['pos_x'] = self.x_coords
            self.programs[i]['pos_y'] = new_y
            self.programs[i]['baseline'] = baseline
        _, timestamps_markers = self.explore_data_handler.get_markers()
        start_index = np.searchsorted(timestamps_markers, x[0])
        found_timestamps = timestamps_markers[start_index:]
        self.marker_program['pos_x'] = found_timestamps
        self.swipe_line_program['pos_x'] = np.array([current_index]).astype(np.float32)
        if not self.is_visible:
            self.is_visible = True
            self.show()
        self.update()

    def on_timer_measure_time(self, event):
        before = time.time()
        self.on_timer(event)
        after = time.time() - before
        self.avg_refresh_time = self.avg_refresh_time * self.avg_refresh_time_iterator + after
        self.avg_refresh_time_iterator += 1
        self.avg_refresh_time /= self.avg_refresh_time_iterator
        if self.avg_refresh_time_iterator % 30 == 0:
            print(f"Average elapsed time for on_timer: {self.avg_refresh_time*1000}ms")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        device_name = sys.argv[1]
    explore_handler = ExploreDataHandlerCircularBuffer(dev_name=device_name)
    c = SwipePlotExploreCanvas(explore_data_handler=explore_handler)
    app.run()
