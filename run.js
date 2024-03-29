import {defs, tiny} from './examples/common.js';

const {
    Vector, Vector3, vec, vec3, vec4, color, hex_color, Shader, Matrix, Mat4, Light, Shape, Material, Scene,
} = tiny;

class Square extends Shape {
    constructor() {
        super("position", "normal",);
        // Loop 3 times (for each axis), and inside loop twice (for opposing cube sides):
        this.arrays.position = Vector3.cast(
            [-1, 0, 1], [1, 0, 1], [1, 0, -1], [-1, 0, -1]);
        this.arrays.normal = Vector3.cast(
            [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]);
        // Arrange the vertices into a square shape in texture space too:
        //this.indices.push(0, 1, 2, 1, 3, 2, 4, 5, 6, 5, 7, 6, 8, 9, 10, 9, 11, 10, 12, 13,
         //   14, 13, 15, 14, 16, 17, 18, 17, 19, 18, 20, 21, 22, 21, 23, 22);
        this.indices.push(0, 3, 1, 1, 3, 2)
    }
}

class Step extends Shape{
    constructor() {
        super("position", "normal",);
        this.arrays.position = Vector3.cast()
        this.arrays.normal = Vector3.cast()
    }
}

export class Run extends Scene {
    constructor() {
        // constructor(): Scenes begin by populating initial values like the Shapes and Materials they'll need.
        super();

        // Hall config
        this.hall_width = 3
        this.step_depth = 10

        this.max_depth = 60
        this.min_depth = -this.step_depth

        // Dynamics
        this.speed = 4.0 // in units / second
        //this.speed = 10.0 // in units / second
        //this.speed = 0.0 // Freeze scene

        this.pause = false;
        this.game = true;

        // At the beginning of our program, load one of each of these shape definitions onto the GPU.
        this.shapes = {
            torus: new defs.Torus(15, 15),
            torus2: new defs.Torus(3, 15),
            sphere: new defs.Subdivision_Sphere(4),
            circle: new defs.Regular_2D_Polygon(1, 15),
            // TODO:  Fill in as many additional shape instances as needed in this key/value table.
            //        (Requirement 1)

            // ** character model **
            body: new defs.Subdivision_Sphere(4),
            left_leg: new defs.Capped_Cylinder(100, 100),
            right_leg: new defs.Capped_Cylinder(100, 100),
            left_arm: new defs.Capped_Cylinder(100, 100),
            right_arm: new defs.Capped_Cylinder(100, 100),

            square: new Square()
        };

        // *** Materials
        this.materials = {
            test: new Material(new defs.Phong_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#ffffff")}),
            test2: new Material(new Gouraud_Shader(),
                {ambient: .4, diffusivity: .6, color: hex_color("#992828")}),
        }

        this.initial_camera_location = Mat4.look_at(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
        // hall width is the same as height
        this.initial_camera_location = Mat4.look_at(vec3(0, this.hall_width/3, 0), vec3(0, this.hall_width/3, -1), vec3(0, 1, 0));
        //this.initial_camera_location = Mat4.look_at(vec3(0, this.hall_width/3, 3), vec3(0, 0, 0), vec3(0, 1, 0));

        this.hall = new Hall(this.hall_width, this.step_depth, this.min_depth, this.max_depth);


        // TODO: put this into a Character data structure and call new Character() instead
        this.body = Mat4.identity()
        .times(Mat4.translation(0, 0.3, -3))
        .times(Mat4.scale(0.2, 0.2, 0.2));

        //matrix for position
        this.position = 0.0;

        this.jump_flag = false;
        this.prev = 0;
        this.jump_velocity = 4.5;
        this.y_0 = 0.3;
        this.gravity = 1.66;
        this.t = 0;
    }
    
    move_left() {
        if ((this.position + -0.6) <= -6.0) {
            console.log("Edge");
        } else {
            this.body = this.body.times(Mat4.translation(-0.6, 0, 0))
            this.position += -0.6;
        }
        //console.log(this.body)
        //console.log(this.position);
        //console.log(this.hall.active_steps[0][0])
    }

    move_right() {
        if ((this.position + 0.6) >= 6.0) {
            console.log("Edge");
        } else {
            this.body = this.body.times(Mat4.translation(0.6, 0, 0))
            this.position += 0.6;
        }
        //console.log(this.body)
        //console.log(this.position);
        //console.log(this.hall.active_steps[0][0])
    }

    jump() {
        const f = 0.03 * (this.y_0 + this.jump_velocity * this.t - (0.5 * this.gravity * this.t**2));
        if(this.body[1][3] <= 0.3 && this.prev >= this.body[1][3]) {
            this.body[1][3] = 0.3;
            this.jump_flag = false;
            this.prev = 0;
            this.t = 0;
        }
        else {
            this.body = this.body.times(Mat4.translation(0, f, 0));
            this.prev = this.body[1][3];
            this.t += 0.1;
        }
        console.log(this.body)
    }

    rotate_cw() {
        if(!this.jump_flag){
            return;
        }
        this.rotated = false;
        this.hall.rotate('cw');
        this.rotated = true;
    }

    rotate_ccw() {
        if(!this.jump_flag){
            return;
        }
        this.rotated = false;
        this.hall.rotate('ccw');
        this.rotated = true;
    }

    make_control_panel() {
  /*      // Draw the scene's buttons, setup their actions and keyboard shortcuts, and monitor live measurements.
        this.key_triggppered_button("View solar system", ["Control", "0"], () => this.attached = () => null);
        this.new_line();
        this.key_triggered_button("Attach to planet 1", ["Control", "1"], () => this.attached = () => this.planet_1);
        this.key_triggered_button("Attach to planet 2", ["Control", "2"], () => this.attached = () => this.planet_2);
        this.new_line();
        this.key_triggered_button("Attach to planet 3", ["Control", "3"], () => this.attached = () => this.planet_3);
        this.key_triggered_button("Attach to planet 4", ["Control", "4"], () => this.attached = () => this.planet_4);
        this.new_line();
        this.key_triggered_button("Attach to moon", ["Control", "m"], () => this.attached = () => this.moon);
*/
        this.key_triggered_button("Left", ["ArrowLeft"], this.move_left);
        this.key_triggered_button("Right", ["ArrowRight"], this.move_right);

        // does nothing right now
        this.key_triggered_button("Jump", [" "], () => this.jump_flag = true);

        this.key_triggered_button("Rotate clockwise", ["r"], this.rotate_cw);
        this.key_triggered_button("Rotate counterclockwise", ["w"], this.rotate_ccw);
        this.key_triggered_button("Pause", ["p"], () => this.pause = !this.pause);
    }


    draw_hall(context, program_state){
        //console.log("start draw hall")

        const angle = this.hall.current_angle
        const n_steps = this.hall.active_steps.length

        let rotation_transform = Mat4.identity()
        rotation_transform = rotation_transform.times(Mat4.translation(0, this.hall.width/2, 0))
        rotation_transform = rotation_transform.times(Mat4.rotation(angle, 0, 0, 1))
        rotation_transform = rotation_transform.times(Mat4.translation(0, -this.hall.width/2, 0))
        let printed = false
        let m = Mat4.rotation(angle, 0, 0, 1)
        let v1, v2
        let x1 = -1.5
        let x2 = -1.5
        let platform = false
        let fullStep = false
        for (let i=0; i<n_steps; i++){
            const step_depth = this.hall.active_steps[i][0]
            const push_back_transform = Mat4.translation(0, 0, -step_depth)
            const step = this.hall.active_steps[i][1]
            step.draw(context, program_state, push_back_transform.times(rotation_transform), this.materials.test)

            if(step_depth < 5 && step_depth > -5){

                if(this.hall.active_steps[i][1].arrays.position.length == 16){
                    fullStep = true
                    continue;
                }
                    //console.log('new step')
                    v1 = vec4(
                        this.hall.active_steps[i][1].arrays.position[0][0],
                        this.hall.active_steps[i][1].arrays.position[0][1]-1.5,
                        this.hall.active_steps[i][1].arrays.position[0][2],
                        1)

                    v1 = m.times(v1)

                    v2 = vec4(
                        this.hall.active_steps[i][1].arrays.position[1][0],
                        this.hall.active_steps[i][1].arrays.position[1][1]-1.5,
                        this.hall.active_steps[i][1].arrays.position[1][2],
                        1)

                    v2 = m.times(v2)
                    console.log(this.body)
                    console.log(v1[0], v2[0], v1[1], v2[1])
                    if (v2[1]< -1.45 || v1[1] < -1.45){
                        console.log('platform at', v1[0], v2[0], v1[1], v2[1])
                        x1 = v1[0]
                        x2 = v2[0]
                        platform = true
                        //console.log(platform)
                    }
                    //console.log(v1)
                    //console.log(v2)
            }
            //step.draw(context, program_state, push_back_transform.times(rotation_transform), this.materials.test)
        }
        if (!this.jump_flag){
            if (Math.abs(this.hall.current_angle - this.hall.desired_angle) < 0.05) {
                if ((!platform || (!(this.body[0][3] > x1 && this.body[0][3] < x2))) && !fullStep){
                    console.log("no platform")
                    console.log(this.body)
                    console.log(x1, x2)
                    this.game = false;
                }
            }
        }
    }
    rotated = false;

    draw_character(context, program_state) {
        let t = program_state.animation_time / 1000;
        let body = Mat4.identity();
        body = body
        .times(Mat4.translation(0, 0.3, -3))
        .times(Mat4.scale(0.2, 0.2, 0.2))
        
        let right_leg = this.body;
        right_leg = right_leg
        .times(Mat4.translation(0.3, 0, 0))
        .times(Mat4.rotation(0.5 * Math.sin(Math.PI * t), 1, 0, 0))
        .times(Mat4.rotation(-Math.PI / 2, 1, 0, 0))
        .times(Mat4.translation(0, 0, -1))
        .times(Mat4.scale(0.2, 0.2, 1))

        let left_leg = this.body;
        left_leg = left_leg
        .times(Mat4.translation(-0.3, 0, 0))
        .times(Mat4.rotation(0.5 * Math.sin(-Math.PI * t), 1, 0, 0))
        .times(Mat4.rotation(-Math.PI / 2, 1, 0, 0))
        .times(Mat4.translation(0, 0, -1))
        .times(Mat4.scale(0.2, 0.2, 1))

        let right_arm = this.body;
        right_arm = right_arm
        .times(Mat4.translation(0.3, 0.8, 0))
        .times(Mat4.rotation(0.5 * Math.sin(-Math.PI * t), 1, 0, 0))
        .times(Mat4.rotation(-Math.PI / 2, 1, 1, 0))
        .times(Mat4.translation(0, 0, -1))
        .times(Mat4.scale(0.2, 0.2, 1))

        let left_arm = this.body;
        left_arm = left_arm
        .times(Mat4.rotation(Math.PI, 0, 1, 0))
        .times(Mat4.translation(0.3, 0.8, 0))
        .times(Mat4.rotation(0.5 * Math.sin(-Math.PI * t), 1, 0, 0))
        .times(Mat4.rotation(Math.PI / 2, 1, 1, 0))
        .times(Mat4.rotation(Math.PI, 0, 1, 0))
        .times(Mat4.translation(0, 0, -1))
        .times(Mat4.scale(0.2, 0.2, 1))

        if(this.jump_flag)
            this.jump(t);

        body = this.body;

        this.shapes.body.draw(context, program_state, body, this.materials.test2);
        this.shapes.right_leg.draw(context, program_state, right_leg, this.materials.test2);
        this.shapes.left_leg.draw(context, program_state, left_leg, this.materials.test2);
        this.shapes.right_arm.draw(context, program_state, right_arm, this.materials.test2);
        this.shapes.left_arm.draw(context, program_state, left_arm, this.materials.test2);

        this.body = body;
    }

    display(context, program_state) {
        // display():  Called once per frame of animation.
        // Setup -- This part sets up the scene's overall camera matrix, projection matrix, and lights:
        if (!context.scratchpad.controls) {
            //this.children.push(context.scratchpad.controls = new defs.Movement_Controls());
            // Define the global camera and projection matrices, which are stored in program_state.
            program_state.set_camera(this.initial_camera_location);
        }


        program_state.projection_transform = Mat4.perspective(
            Math.PI / 4, context.width / context.height, .1, 1000);


        const light_position = vec4(0, 5, 5, 1);
        // The parameters of the Light are: position, color, size
        //program_state.lights = [new Light(light_position, color(1, 1, 1, 1), 1000)];

        // TODO:  Fill in matrix operations and drawing code to draw the solar system scene (Requirements 3 and 4)
        const t = program_state.animation_time / 1000, dt = program_state.animation_delta_time / 1000;
        //this.shapes.torus.draw(context, program_state, model_transform, this.materials.test.override({color: yellow}));
        let model_transform = Mat4.identity();

        program_state.lights = [new Light(light_position, color(1, 1, 1, 1), 10**3)];

        //this.shapes.square.draw(context, program_state, model_transform, this.materials.test)

        //Draw hall
        if(this.game && !this.pause){
            const pull_distance = dt * this.speed
            this.hall.pull_hall(pull_distance)

            this.hall.clip_steps();
            this.hall.make_steps();

            // this parameter should be between 0 and 1. 0.07 seems ok.
            this.hall.update_current_angle(0.07)
        }



        // The following code seems complicated but is only to automate a rotation every 2 seconds
        // The hall can be rotated with this.hall.rotate('cw') or this.hall.rotate('ccw').
        // update_current_angle is to allow for smooth transition.

        /*
        if (Math.floor(t)%2 == 0){
            if (!this.rotated) {
                this.hall.rotate('cw')
                this.rotated = true
            }
        } else {
            this.rotated = false
        }*/

        this.draw_hall(context, program_state)

        this.draw_character(context, program_state);
    }
}

class StepFactory{
    constructor(width, depth) {
        this.width = width;
        this.depth = depth;
    }
    make_full_step(){

        let s = new Step()
        let m = Mat4.identity()
        m = m.times(this.make_base())

        //Square.insert_transformed_copy_into(s, [], m)
        Square.insert_transformed_copy_into(s, [], this.make_base())
        Square.insert_transformed_copy_into(s, [], this.make_left())
        Square.insert_transformed_copy_into(s, [], this.make_right())
        Square.insert_transformed_copy_into(s, [], this.make_top())

        return s
    }
    make_base(){
        return Mat4.scale(this.width/2,1,this.depth/2);
    }
    make_left(){
        let mat = Mat4.identity()
        mat = mat.times(Mat4.rotation(-Math.PI/2, 0, 0, 1))
        mat = mat.times(this.make_base())
        mat = mat.times(Mat4.translation(-1,-this.width/2, 0))
        return mat
    }
    make_right(){
        let mat = Mat4.identity()
        mat = mat.times(Mat4.rotation(+Math.PI/2, 0, 0, 1))
        mat = mat.times(this.make_base())
        mat = mat.times(Mat4.translation(+1,-this.width/2, 0))
        return mat
    }
    make_top(){
        let mat = Mat4.identity()
        mat = mat.times(Mat4.rotation(Math.PI, 0, 0, 1))
        mat = mat.times(this.make_base())
        mat = mat.times(Mat4.translation(0,-this.width, 0))
        return mat
    }


    make_square_step(){
        let s = new Step()
        let m = Mat4.identity()
        Square.insert_transformed_copy_into(s, [],m)
        return s
    }

    make_platform_step(randomize_position=true, angle=0.0){
        let s = new Step()
        let m = Mat4.identity()

        m = m.times(Mat4.translation(0 , this.width/2, 0))
        m = m.times(Mat4.rotation(angle, 0, 0, 1))
        m = m.times(Mat4.translation(0 , -this.width/2, 0))

        if (randomize_position) {
            const random_position = (Math.random() - 0.5) * (this.width - 1)
            m = m.times(Mat4.translation(random_position, 0, 0))
        }

        m = m.times(Mat4.scale(0.5, 1, this.depth/2))

        Square.insert_transformed_copy_into(s, [],m)

        return s
    }

}

class Hall{
    constructor(width, step_depth, min_depth, max_depth, ) {
        this.width = width
        this.min_depth = min_depth
        this.max_depth = max_depth
        this.step_depth = step_depth

        this.desired_angle=0
        this.current_angle=0

        this.step_factory = new StepFactory(width, step_depth)
        // each element of acive steps consists on a step depth and Step object.
        this.active_steps = []

        this.init_steps()
    }
    pull_hall(d){
        for(let i=0; i<this.active_steps.length; i++){
            this.active_steps[i][0] -= d;
        }
        this.last_step_depth -= d
    }
    rotate(side){
        if (side=='cw'){
            this.desired_angle -= Math.PI/2
        } else if (side=='ccw'){
            this.desired_angle += Math.PI/2
        }
    }

    update_current_angle(transition_speed)
    {
        // Transition speed has to go be between 0 (excluded) and 1 (instant transition)
        this.current_angle = this.current_angle * (1-transition_speed) + this.desired_angle * transition_speed
    }

    clip_steps(){
        let active_steps = []
        for(let i=0; i<this.active_steps.length; i++){
            if (this.active_steps[i][0] >= this.min_depth){
                active_steps.push(this.active_steps[i])
            }
        }
        this.active_steps = active_steps;
    }
    make_steps(){
        //console.log(this.last_step_depth, this.max_depth)
        while(this.last_step_depth < this.max_depth){
            //console.log('new step')
            this.active_steps.push(
                [this.last_step_depth + this.step_depth*3, this.step_factory.make_full_step()]
            )

            let angle= Math.floor(Math.random() * 3) * Math.PI / 2
            this.active_steps.push(
                [this.last_step_depth + this.step_depth*4, this.step_factory.make_platform_step(true, angle)]
            )
            this.active_steps.push(
                [this.last_step_depth + this.step_depth*5, this.step_factory.make_platform_step(true, angle)]
            )
            angle= angle + Math.floor(Math.random() * 2 + 1) * Math.PI / 2
            this.active_steps.push(
                [this.last_step_depth + this.step_depth*4, this.step_factory.make_platform_step(true, angle)]
            )
            if (Math.random() > 0.5) {
                this.active_steps.push(
                    [this.last_step_depth + this.step_depth * 5, this.step_factory.make_platform_step(true, angle)]
                )
            }
            this.last_step_depth = this.last_step_depth + this.step_depth*3
        }
    }
    init_steps(){
        const n_init_full=3 // Number of blocks with no holes at the beginning
        for (let n=0; n<n_init_full; n++){
            this.active_steps.push(
                [n*this.step_depth, this.step_factory.make_full_step()]
            )
        }
        this.active_steps.push(
            [(n_init_full)*this.step_depth, this.step_factory.make_platform_step(true, 0)]
        )
        this.active_steps.push(
            [(n_init_full+1)*this.step_depth, this.step_factory.make_platform_step(true, 0)]
        )

        this.last_step_depth = (n_init_full-1)*this.step_depth;
    }
}






class Gouraud_Shader extends Shader {
    // This is a Shader using Phong_Shader as template
    // TODO: Modify the glsl coder here to create a Gouraud Shader (Planet 2)

    constructor(num_lights = 2) {
        super();
        this.num_lights = num_lights;
    }

    shared_glsl_code() {
        // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
        return ` 
        precision mediump float;
        const int N_LIGHTS = ` + this.num_lights + `;
        uniform float ambient, diffusivity, specularity, smoothness;
        uniform vec4 light_positions_or_vectors[N_LIGHTS], light_colors[N_LIGHTS];
        uniform float light_attenuation_factors[N_LIGHTS];
        uniform vec4 shape_color;
        uniform vec3 squared_scale, camera_center;
        
        // Specifier "varying" means a variable's final value will be passed from the vertex shader
        // on to the next phase (fragment shader), then interpolated per-fragment, weighted by the
        // pixel fragment's proximity to each of the 3 vertices (barycentric interpolation).
        varying vec3 N, vertex_worldspace;
        varying vec4 vertex_color;
        // ***** PHONG SHADING HAPPENS HERE: *****                                       
        vec3 phong_model_lights( vec3 N, vec3 vertex_worldspace ){                                        
            // phong_model_lights():  Add up the lights' contributions.
            vec3 E = normalize( camera_center - vertex_worldspace );
            vec3 result = vec3( 0.0 );
            for(int i = 0; i < N_LIGHTS; i++){
                // Lights store homogeneous coords - either a position or vector.  If w is 0, the 
                // light will appear directional (uniform direction from all points), and we 
                // simply obtain a vector towards the light by directly using the stored value.
                // Otherwise if w is 1 it will appear as a point light -- compute the vector to 
                // the point light's location from the current surface point.  In either case, 
                // fade (attenuate) the light as the vector needed to reach it gets longer.  
                vec3 surface_to_light_vector = light_positions_or_vectors[i].xyz - 
                                               light_positions_or_vectors[i].w * vertex_worldspace;                                             
                float distance_to_light = length( surface_to_light_vector );

                vec3 L = normalize( surface_to_light_vector );
                vec3 H = normalize( L + E );
                // Compute the diffuse and specular components from the Phong
                // Reflection Model, using Blinn's "halfway vector" method:
                float diffuse  =      max( dot( N, L ), 0.0 );
                float specular = pow( max( dot( N, H ), 0.0 ), smoothness );
                float attenuation = 1.0 / (1.0 + light_attenuation_factors[i] * distance_to_light * distance_to_light );
                
                vec3 light_contribution = shape_color.xyz * light_colors[i].xyz * diffusivity * diffuse
                                                          + light_colors[i].xyz * specularity * specular;
                result += attenuation * light_contribution;
            }
            return result;
        } `;
    }
    vertex_glsl_code() {
        // ********* VERTEX SHADER *********
        return this.shared_glsl_code() + `
            attribute vec3 position, normal;                            
            // Position is expressed in object coordinates.
            
            uniform mat4 model_transform;
            uniform mat4 projection_camera_model_transform;
    
            void main(){                                                                   
                // The vertex's final resting place (in NDCS):
                gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
                // The final normal vector in screen space.
                N = normalize( mat3( model_transform ) * normal / squared_scale);
                vertex_worldspace = ( model_transform * vec4( position, 1.0 ) ).xyz;
                
                
                vertex_color = vec4( shape_color.xyz * ambient, shape_color.w );
                vertex_color.xyz += phong_model_lights( normalize( N ), vertex_worldspace );
            
            } `;
    }

    /*
    attribute vec3 position, normal;

            uniform mat4 model_transform;
            uniform mat4 projection_camera_model_transform;

            void main(){
                gl_Position = projection_camera_model_transform * vec4(position, 1.0);
                vec3 N = normalize(mat3(model_transform) * normal / squared_scale);
                vec3 vertex_worldspace = (model_transform * vec4 (position, 1.0)).xyz;

                vertex_color = color_based_on_normal(N, vertex_worldspace);
            }
     */
    /*
    out vec3 vertex_color;
            void main(){
               vec3 v = vec3(gl_ModelViewMatrix * gl_Vertex);
               vec3 N = normalize(gl_NormalMatrix * gl_Normal);

               vec3 L = normalize(gl_LightSource[0].position.xyz - v);
               vec3 E = normalize(-v);
               vec3 R = normalize(-reflect(L, N));

               vec4 Iamb = gl_FrontLightProduct[0].ambient;

               vec4 Idiff = gl_FrontLightProduct[0].diffuse * max(dot(N, L), 0.0);

               vec4 Ispec = gl_FrontLightProduct[0].specular *
                            pow(max(dot(R,E), 0.0),0.3*gl_FrontMaterial.shininess);

               vertex_color = gl_FrontLightModelProduct.sceneColor + Iamb + Idiff + Ispec;

               gl_Position = gl_ModelViewProjectionMatrix + gl_Vertex;
                           }
     */
    fragment_glsl_code() {
        // ********* FRAGMENT SHADER *********
        // A fragment is a pixel that's overlapped by the current triangle.
        // Fragments affect the final image or get discarded due to depth.
        return this.shared_glsl_code() + `
            void main(){                                                           
                // Compute an initial (ambient) color:
                //gl_FragColor = vec4( shape_color.xyz * ambient, shape_color.w );
                // Compute the final color with contributions from lights:
                // gl_FragColor.xyz += phong_model_lights( normalize( N ), vertex_worldspace );
                gl_FragColor = vertex_color;
                
            } `;
    }

    send_material(gl, gpu, material) {
        // send_material(): Send the desired shape-wide material qualities to the
        // graphics card, where they will tweak the Phong lighting formula.
        gl.uniform4fv(gpu.shape_color, material.color);
        gl.uniform1f(gpu.ambient, material.ambient);
        gl.uniform1f(gpu.diffusivity, material.diffusivity);
        gl.uniform1f(gpu.specularity, material.specularity);
        gl.uniform1f(gpu.smoothness, material.smoothness);
    }

    send_gpu_state(gl, gpu, gpu_state, model_transform) {
        // send_gpu_state():  Send the state of our whole drawing context to the GPU.
        const O = vec4(0, 0, 0, 1), camera_center = gpu_state.camera_transform.times(O).to3();
        gl.uniform3fv(gpu.camera_center, camera_center);
        // Use the squared scale trick from "Eric's blog" instead of inverse transpose matrix:
        const squared_scale = model_transform.reduce(
            (acc, r) => {
                return acc.plus(vec4(...r).times_pairwise(r))
            }, vec4(0, 0, 0, 0)).to3();
        gl.uniform3fv(gpu.squared_scale, squared_scale);
        // Send the current matrices to the shader.  Go ahead and pre-compute
        // the products we'll need of the of the three special matrices and just
        // cache and send those.  They will be the same throughout this draw
        // call, and thus across each instance of the vertex shader.
        // Transpose them since the GPU expects matrices as column-major arrays.
        const PCM = gpu_state.projection_transform.times(gpu_state.camera_inverse).times(model_transform);
        gl.uniformMatrix4fv(gpu.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        gl.uniformMatrix4fv(gpu.projection_camera_model_transform, false, Matrix.flatten_2D_to_1D(PCM.transposed()));

        // Omitting lights will show only the material color, scaled by the ambient term:
        if (!gpu_state.lights.length)
            return;

        const light_positions_flattened = [], light_colors_flattened = [];
        for (let i = 0; i < 4 * gpu_state.lights.length; i++) {
            light_positions_flattened.push(gpu_state.lights[Math.floor(i / 4)].position[i % 4]);
            light_colors_flattened.push(gpu_state.lights[Math.floor(i / 4)].color[i % 4]);
        }
        gl.uniform4fv(gpu.light_positions_or_vectors, light_positions_flattened);
        gl.uniform4fv(gpu.light_colors, light_colors_flattened);
        gl.uniform1fv(gpu.light_attenuation_factors, gpu_state.lights.map(l => l.attenuation));
    }

    update_GPU(context, gpu_addresses, gpu_state, model_transform, material) {
        // update_GPU(): Define how to synchronize our JavaScript's variables to the GPU's.  This is where the shader
        // recieves ALL of its inputs.  Every value the GPU wants is divided into two categories:  Values that belong
        // to individual objects being drawn (which we call "Material") and values belonging to the whole scene or
        // program (which we call the "Program_State").  Send both a material and a program state to the shaders
        // within this function, one data field at a time, to fully initialize the shader for a draw.

        // Fill in any missing fields in the Material object with custom defaults for this shader:
        const defaults = {color: color(0, 0, 0, 1), ambient: 0, diffusivity: 1, specularity: 1, smoothness: 40};
        material = Object.assign({}, defaults, material);

        this.send_material(context, gpu_addresses, material);
        this.send_gpu_state(context, gpu_addresses, gpu_state, model_transform);
    }
}

class Ring_Shader extends Shader {
    update_GPU(context, gpu_addresses, graphics_state, model_transform, material) {
        // update_GPU():  Defining how to synchronize our JavaScript's variables to the GPU's:
        const [P, C, M] = [graphics_state.projection_transform, graphics_state.camera_inverse, model_transform],
            PCM = P.times(C).times(M);
        context.uniformMatrix4fv(gpu_addresses.model_transform, false, Matrix.flatten_2D_to_1D(model_transform.transposed()));
        context.uniformMatrix4fv(gpu_addresses.projection_camera_model_transform, false,
            Matrix.flatten_2D_to_1D(PCM.transposed()));
    }

    shared_glsl_code() {
        // ********* SHARED CODE, INCLUDED IN BOTH SHADERS *********
        return `
        precision mediump float;
        varying vec4 point_position;
        varying vec4 center;
        `;
    }

    vertex_glsl_code() {
        // ********* VERTEX SHADER *********
        // TODO:  Complete the main function of the vertex shader (Extra Credit Part II).
        return this.shared_glsl_code() + `
        attribute vec3 position;
        uniform mat4 model_transform;
        uniform mat4 projection_camera_model_transform;
        
        void main(){
          gl_Position = projection_camera_model_transform * vec4( position, 1.0 );
          
          center = model_transform * vec4(0, 0, 0, 1);
          point_position = model_transform * vec4( position, 1.0 );
          
        }`;
    }

    fragment_glsl_code() {
        // ********* FRAGMENT SHADER *********
        // TODO:  Complete the main function of the fragment shader (Extra Credit Part II).
        return this.shared_glsl_code() + `
        void main(){
          
          vec3 vertex_color = normalize(center.xyz);
          
          float dist = distance(center, point_position);
          
          float x = sin(dist*21.0);
          
          vec3 color = vec3(0.685, 0.5, 0.25) * x;
          
          // for some reason, sin(5 * dist) crashes --> precision issues?
          gl_FragColor = vec4(color, 1.0);
          
        }`;
    }
}

