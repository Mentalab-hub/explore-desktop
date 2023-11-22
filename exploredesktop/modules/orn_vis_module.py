import numpy as np

from PySide6.QtWidgets import QVBoxLayout, QPushButton

from vispy import app, gloo
from vispy.gloo import Program, VertexBuffer, IndexBuffer
from vispy.util.transforms import perspective, translate
from vispy.geometry import create_box


class OrientationVisualiser:
    '''Adds a view to the orientation tab which contains a cube visualising the device orientation,
    requires calibration data to work (there is currently no way to calibrate the device from Explore Desktop!).
    '''
    def __init__(self, ui) -> None:
        self.ui = ui
        c = OrientationVisualisationCanvas()
        self.callback = c.on_mapped_orn_received
        self.vertical_layout = QVBoxLayout()
        self.ui.horizontal_orientation_layout.addLayout(self.vertical_layout)
        self.vertical_layout.addWidget(c.native)

        self.reset_orn_button = QPushButton(text="Reset orientation")
        self.reset_orn_button.clicked.connect(c.reset_orn)

        self.calibrate_orn_button = QPushButton(text="Calibrate orientation")
        self.calibrate_orn_button.clicked.connect(self.calibrate_orientation)

        self.vertical_layout.addWidget(self.reset_orn_button)
        self.vertical_layout.addWidget(self.calibrate_orn_button)

        #self.ui.horizontal_orientation_layout.addWidget(c.native)

    def calibrate_orientation(self):
        raise NotImplementedError


vertex = """
// uniforms passed from outside to the program, shared across vertices
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 normal_matrix;

uniform bool u_use_cube_color;
uniform vec4 u_cube_color;

// attributes passed from vertex buffer
attribute vec3 position;
attribute vec2 texcoord;
attribute vec3 normal;
attribute vec4 color;

// outputs to the fragment shader
out vec4 v_color;
out vec3 v_normal;
out vec3 v_world_pos;

void main()
{
    gl_Position = projection * view * model * vec4(position,1.0); // see gl_Position from OpenGL, sets vertex position
    if(u_use_cube_color){
        v_color = u_cube_color;
    } else {
        v_color = color; // uses position-based coloring - default from vispy's create_<geometry>
    }
    v_normal = mat3(normal_matrix) * normal; // vertex normal in world space
    v_world_pos = vec3(model * vec4(position,1.0)); // vertex position in world space
}
"""

fragment = """
// inputs from the vertex shader
in vec4 v_color;
in vec3 v_normal;
in vec3 v_world_pos;

// ambient light values
uniform vec3 u_ambient_light_color;
uniform float u_ambient_light_intensity;

// aiffuse light values
uniform vec3 u_diff_light_color;
uniform vec3 u_diff_light_position;

// specular light values
uniform vec3 view_pos;
uniform float u_specular_light_intensity;
uniform float u_specular_light_shininess;
uniform vec3 u_specular_light_color;

// implements Phong shading (ideally), currently not working properly
void main()
{
    vec3 norm = normalize(v_normal);

    // ambient light calculation
    vec3 ambient = u_ambient_light_intensity * u_ambient_light_color;

    // diffuse light calculation
    vec3 light_dir = normalize(u_diff_light_position - v_world_pos);
    float diff_coeff = max(dot(norm, light_dir), 0.0);
    vec3 diff = diff_coeff * u_diff_light_color;

    // specular light calculation
    vec3 view_dir = normalize(view_pos - v_world_pos);
    //vec3 view_dir = normalize(vec3(0.0, 0.0, -5.0) - v_world_pos);
    vec3 reflect_dir = reflect(-light_dir, norm);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), u_specular_light_shininess);
    vec3 specular = u_specular_light_intensity * spec * u_specular_light_color;

    // specular lighting not working yet, use only ambient and diffuse for now
    
    //vec4 col = vec4((ambient + diff + specular), 1.0) * v_color;
    
    vec4 col = vec4(ambient + diff, 1.0) * v_color;
    
    gl_FragColor = col; // set fragment colour, see gl_FragColor from OpenGL
}
"""


class OrientationVisualisationCanvas(app.Canvas):
    '''Orientation visualiser class using vispy. Based on vispy's rotating cube examples, using the matrix in the mapped
    orientation packets from explorepy to set the model matrix of the visualised box. Accuracy of the visualisation
    depends on this matrix.
    '''
    def __init__(self):
        app.Canvas.__init__(self)

        # Build cube data
        V, I, _ = create_box(1.0, 1.0, 1.0)
        vertices = VertexBuffer(V)
        self.indices = IndexBuffer(I)

        # Build program
        self.program = Program(vertex, fragment)
        self.program.bind(vertices)

        # Build view, model, projection & normal
        view_pos = (0, 0, -5)
        view = translate(view_pos)
        model = np.eye(4, dtype=np.float32)
        self.program['model'] = model
        self.default_orientation = model
        self.program['view'] = view
        self.program['view_pos'] = view_pos
        self.program['normal_matrix'] = np.linalg.inv(model).T

        self.program['u_use_cube_color'] = True
        self.program['u_cube_color'] = (0.5, 0.5, 0.5, 1.0)

        # Pass ambient values to shaders
        self.program['u_ambient_light_color'] = (1.0, 1.0, 1.0)
        self.program['u_ambient_light_intensity'] = 0.7

        # Pass diffuse values to shaders
        self.program['u_diff_light_color'] = (1.0, 1.0, 1.0)
        self.program['u_diff_light_position'] = (4.0, 4.0, 4.0)

        # Pass specular values to shaders, not used currently
        self.program['u_specular_light_color'] = (0.0, 0.0, 1.0)
        self.program['u_specular_light_intensity'] = 0.8
        self.program['u_specular_light_shininess'] = 64

        self.phi, self.theta = 0, 0
        gloo.set_state(clear_color=(0.30, 0.30, 0.35, 1.00), depth_test=True)

        self.activate_zoom()

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.program.draw('triangles', self.indices)

    def on_resize(self, event):
        self.activate_zoom()

    def activate_zoom(self):
        gloo.set_viewport(0, 0, *self.physical_size)
        projection = perspective(45.0, self.size[0] / float(self.size[1]),
                                 2.0, 10.0)
        self.program['projection'] = projection

    def on_mapped_orn_received(self, packet):
        new_model = np.matmul(self.default_orientation, packet.matrix)
        self.program['model'] = new_model
        self.program['normal_matrix'] = np.linalg.inv(new_model).T
        self.update()

    def reset_orn(self):
        self.default_orientation = np.matmul(np.linalg.inv(np.reshape(self.program['model'], (4, 4))),
                                             self.default_orientation)
