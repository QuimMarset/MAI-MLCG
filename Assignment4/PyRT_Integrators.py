from GaussianProcess import *
from PyRT_Common import *


# -------------------------------------------------Integrator Classes
# the integrators also act like a scene class in that-
# it stores all the primitives that are to be ray traced.
class Integrator(ABC):
    # Initializer - creates object list
    def __init__(self, filename_, experiment_name=''):
        # self.primitives = []
        self.filename = filename_ + experiment_name
        # self.env_map = None  # not initialized
        self.scene = None

    @abstractmethod
    def compute_color(self, ray):
        pass

    # def add_environment_map(self, env_map_path):
    #    self.env_map = EnvironmentMap(env_map_path)
    def add_scene(self, scene):
        self.scene = scene

    def get_filename(self):
        return self.filename

    # Simple render loop: launches 1 ray per pixel
    def render(self):
        # YOU MUST CHANGE THIS METHOD IN ASSIGNMENTS 1.1 and 1.2:
        cam = self.scene.camera  # camera object
        #ray = Ray()
        print('Rendering Image: ' + self.get_filename())
        print(cam.width, cam.height)
        for x in range(0, cam.width):
            for y in range(0, cam.height):
                
                # Assignment 1.1
                # pixel = RGBColor(x/cam.width, y/cam.height, 0)

                # Assignment 1.2
                ray = Ray(direction=cam.get_direction(x, y))
                pixel = self.compute_color(ray)
                
                self.scene.set_pixel(pixel, x, y)  # save pixel to pixel array
            progress = (x / cam.width) * 100
            print('\r\tProgress: ' + str(progress) + '%', end='')
        # save image to file
        print('\r\tProgress: 100% \n\t', end='')
        full_filename = self.get_filename()
        self.scene.save_image(full_filename)


class LazyIntegrator(Integrator):
    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray):
        return BLACK


class IntersectionIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Intersection')

    def compute_color(self, ray):
        # ASSIGNMENT 1.2: PUT YOUR CODE HERE
        if self.scene.any_hit(ray):
            return RED
        else:
            return BLACK


class DepthIntegrator(Integrator):

    def __init__(self, filename_, max_depth_=10):
        super().__init__(filename_ + '_Depth')
        self.max_depth = max_depth_

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        hit_data = self.scene.closest_hit(ray)
        color = BLACK
        if hit_data.has_hit:
            hit_distance = hit_data.hit_distance
            color_comp = max(1 - (hit_distance/self.max_depth), 0)
            color = RGBColor(color_comp, color_comp, color_comp)
        return color


class NormalIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Normal')

    def compute_color(self, ray):
        # ASSIGNMENT 1.3: PUT YOUR CODE HERE
        hit_data = self.scene.closest_hit(ray)
        color = BLACK
        if hit_data.has_hit:
            normal = hit_data.normal
            color = (normal + Vector3D(1, 1, 1))/2
            color = RGBColor(color.x, color.y, color.z)
        return color


class PhongIntegrator(Integrator):

    def __init__(self, filename_):
        super().__init__(filename_ + '_Phong')

    def compute_color(self, ray):
        # ASSIGNMENT 1.4: PUT YOUR CODE HERE
        color = BLACK
        
        hit_data = self.scene.closest_hit(ray)
        
        if hit_data.has_hit:

            normal = hit_data.normal
            point = hit_data.hit_point
            primitive_idx = hit_data.primitive_index
            primitive = self.scene.object_list[primitive_idx]

            for light_source in self.scene.pointLights:                
                intensity = light_source.intensity
                light_pos = light_source.pos

                x = np.array([light_pos.x, light_pos.y, light_pos.z])
                y = np.array([point.x, point.y, point.z])
                distance = np.linalg.norm(x - y)
                
                light_vector = (light_pos - point)/distance

                ray2 = Ray(point, light_vector, distance)
                if self.scene.any_hit(ray2):
                    continue

                cam_ray_unit_vector = ray.d * (-1) / hit_data.hit_distance
                color += primitive.BRDF.get_value(light_vector, cam_ray_unit_vector, normal).multiply(intensity / (distance**2))

            color += self.scene.i_a.multiply(primitive.BRDF.kd)

        return color


class CMCIntegrator(Integrator):  # Classic Monte Carlo Integrator

    def __init__(self, n, filename_, experiment_name=''):
        filename_mc = filename_ + '_MC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_mc)
        self.n_samples = n
        self.pdf = UniformPDF()

    def compute_color(self, ray):
        color = BLACK
        
        hit_data = self.scene.closest_hit(ray)
        point = hit_data.hit_point
        
        if hit_data.has_hit:
            normal = hit_data.normal
            object_hit = self.scene.object_list[hit_data.primitive_index]

            sample_dir, sample_prob = sample_set_hemisphere(self.n_samples, self.pdf)

            illum_integral_estimate = BLACK

            for direction, prob in zip(sample_dir, sample_prob):
                w_j_rotated = center_around_normal(direction, normal)

                ray2 = Ray(point, w_j_rotated)
                hit_2_data = self.scene.closest_hit(ray2)

                if hit_2_data.has_hit:
                    object_hit_2 = self.scene.object_list[hit_2_data.primitive_index]
                    incident_radiance = object_hit_2.emission
                else:
                    incident_radiance = self.scene.env_map.getValue(w_j_rotated)

                brdf_value = object_hit.BRDF.get_value(w_j_rotated, Vector3D(0, 0, 0), normal)
                integrand_sample = incident_radiance.multiply(brdf_value)
                illum_integral_estimate += integrand_sample/prob

            color = illum_integral_estimate/self.n_samples

        else:
            color = self.scene.env_map.getValue(ray.d)

        return color


def create_gaussian_process(num_samples, pdf, p_func):
    gaussian_process = GP(SobolevCov(), p_func)
    samples_pos, _ = sample_set_hemisphere(num_samples, pdf)
    gaussian_process.add_sample_pos(samples_pos)
    return gaussian_process


class BayesianMonteCarloIntegrator(Integrator):
    def __init__(self, n, filename_, num_gp=5, experiment_name=''):
        filename_bmc = filename_ + '_BMC_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_bmc)
        self.n_samples = n
        self.pdf = UniformPDF()
        self.myGPs = [create_gaussian_process(n, self.pdf, Constant(1)) for _ in range(num_gp)]


    def compute_color(self, ray):
        color = BLACK
        hit_data = self.scene.closest_hit(ray)
        point = hit_data.hit_point
        
        if hit_data.has_hit:
            normal = hit_data.normal
            object_hit = self.scene.object_list[hit_data.primitive_index]

            sample_integrands = []

            # Removing artifacts
            index = np.random.choice(len(self.myGPs))
            selected_gp = self.myGPs[index]
            sample_directions = selected_gp.samples_pos

            for sample_dir in sample_directions:
                # Removing artifacts
                y_rotated_dir = rotate_around_y(np.random.randint(1, 360), sample_dir)
                w_j_rotated = center_around_normal(y_rotated_dir, normal)

                ray2 = Ray(point, w_j_rotated)
                hit_2_data = self.scene.closest_hit(ray2)

                if hit_2_data.has_hit:
                    object_hit_2 = self.scene.object_list[hit_2_data.primitive_index]
                    incident_radiance = object_hit_2.emission
                else:
                    incident_radiance = self.scene.env_map.getValue(w_j_rotated)

                brdf_value = object_hit.BRDF.get_value(w_j_rotated, Vector3D(0, 0, 0), normal)
                sample_integrand = incident_radiance.multiply(brdf_value)
                sample_integrands.append(sample_integrand)

            selected_gp.add_sample_val(sample_integrands)
            bmc_integral_estimate = selected_gp.compute_integral_BMC()
            color = bmc_integral_estimate

        else:
            color = self.scene.env_map.getValue(ray.d)
        
        return color


class CMCISIntegrator(CMCIntegrator):

    def __init__(self, n, filename_, experiment_name=''):
        filename_mc_is = filename_ + '_MC_IS_' + str(n) + '_samples' + experiment_name
        Integrator.__init__(self, filename_mc_is)
        self.n_samples = n
        self.pdf = CosinePDF(1)


def create_gaussian_process_IS(num_samples, pdf):
    gaussian_process = GP_IS(SobolevCov())
    samples_pos, _ = sample_set_hemisphere(num_samples, pdf)
    gaussian_process.add_sample_pos(samples_pos)
    return gaussian_process


class BayesianMCISIntegrator(Integrator):

    def __init__(self, n, filename_, num_gp=5, experiment_name=''):
        filename_bmc = filename_ + '_BMC_IS_' + str(n) + '_samples' + experiment_name
        super().__init__(filename_bmc)
        self.n_samples = n
        self.pdf = CosinePDF(1)
        self.myGPs = [create_gaussian_process_IS(n, self.pdf) for _ in range(num_gp)]

    
    def compute_color(self, ray):
        color = BLACK
        hit_data = self.scene.closest_hit(ray)
        point = hit_data.hit_point
        
        if hit_data.has_hit:
            normal = hit_data.normal
            object_hit = self.scene.object_list[hit_data.primitive_index]

            # Removing artifacts
            index = np.random.choice(len(self.myGPs))
            selected_gp = self.myGPs[index]
            sample_directions = selected_gp.samples_pos

            incident_radiance_values = np.zeros((self.n_samples, 3))

            for (i, sample_dir) in enumerate(sample_directions):
                # Removing artifacts
                y_rotated_dir = rotate_around_y(np.random.randint(1, 360), sample_dir)
                w_j_rotated = center_around_normal(y_rotated_dir, normal)

                ray2 = Ray(point, w_j_rotated)
                hit_2_data = self.scene.closest_hit(ray2)

                if hit_2_data.has_hit:
                    object_hit_2 = self.scene.object_list[hit_2_data.primitive_index]
                    incident_radiance = object_hit_2.emission
                else:
                    incident_radiance = self.scene.env_map.getValue(w_j_rotated)

                incident_radiance_values[i] = [incident_radiance.r, incident_radiance.g, incident_radiance.b]

            selected_gp.add_sample_val(incident_radiance_values)
            bmc_integral_estimate = selected_gp.compute_integral_BMC(object_hit.BRDF, normal)
            color = bmc_integral_estimate

        else:
            color = self.scene.env_map.getValue(ray.d)
        
        return color
